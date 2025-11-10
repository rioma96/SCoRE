import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
import pickle as pickle
import numpy as np
import pandas as pd
import glob
from codecarbon import EmissionsTracker
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random



# to include the Repo code without installing in the environment
import sys
sys.path.append('../')



def build_balanced_class_batches_from_df(
    df,
    input_col,              # column with arrays / tensors
    label_col,              # column with multi-hot labels
    k_per_class=5,          # samples per class in each final batch
    seed=300,
    batch_shuffle_buffer=256,   # how many full batches to keep in a shuffle buffer
    prefetch=True
):
    """
    Returns a tf.data.Dataset that yields batches of size k_per_class * n_classes.
    Each batch contains exactly k_per_class examples sampled (with repetition) from each class bucket.
    """
    # Set seeds for determinism
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Materialize arrays from the DataFrame
    X = np.stack(df[input_col].to_numpy(), axis=0)
    Y = np.stack(df[label_col].to_numpy(), axis=0).astype(np.int32)

    n_classes = Y.shape[1]
    assert Y.ndim == 2, "label_col must be multi-hot (2D) with shape [n_samples, n_classes]"

    # Per-class datasets (examples may appear in multiple classes if multi-label)
    class_datasets = []
    for c in range(n_classes):
        idx = np.where(Y[:, c] == 1)[0]
        if idx.size == 0:
            # If a class has no samples, we can't build balanced batches that include it.
            # You can: (a) skip it, (b) synthesize data, or (c) raise.
            # Here we skip it gracefully by continuing.
            continue

        Xc = X[idx]
        Yc = Y[idx]

        # Build per-class dataset
        ds_c = tf.data.Dataset.from_tensor_slices((Xc, Yc))
        # Shuffle within class; repeat to provide an infinite stream (upsampling rare classes)
        ds_c = ds_c.shuffle(buffer_size=max(len(idx), 1), seed=seed, reshuffle_each_iteration=True).repeat()
        # Take k_per_class at a time from this class
        ds_c = ds_c.batch(k_per_class, drop_remainder=True)
        class_datasets.append(ds_c)

    if not class_datasets:
        raise ValueError("No classes with samples were found.")

    n_effective_classes = len(class_datasets)

    # Zip one chunk from each class, then concat along the batch dimension
    def _merge_batches(*class_batches):
        # class_batches is a tuple of (x_batch, y_batch) from each class
        xs = [xb for xb, _ in class_batches]
        ys = [yb for _, yb in class_batches]
        x = tf.concat(xs, axis=0)  # [k_per_class * n_effective_classes, ...]
        y = tf.concat(ys, axis=0)  # [k_per_class * n_effective_classes, n_classes]
        return x, y

    balanced = tf.data.Dataset.zip(tuple(class_datasets)).map(
        _merge_batches,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Optional: shuffle at the batch level (keeps class balance within each batch)
    if batch_shuffle_buffer and batch_shuffle_buffer > 0:
        balanced = balanced.shuffle(batch_shuffle_buffer, seed=seed, reshuffle_each_iteration=True)

    if prefetch:
        balanced = balanced.prefetch(tf.data.AUTOTUNE)

    return balanced, (k_per_class * n_effective_classes)




# Small utility to tee prints to both terminal and a logfile
class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
            except Exception:
                pass
        self.flush()

    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass

from CBKGE.NN_creation_and_dependencies import *
from CBKGE.NN_preproc import *
from CBKGE.utilities_validation import *


def significant_correlation_matrix(one_hot_labels, link_dict, alpha=0.05):
    """
    Plots a heatmap of significant correlations between classes.

    Parameters:
    - one_hot_labels: array-like, shape (n_samples, n_classes)
      One-hot encoded labels.
    - link_dict: dict
      Dictionary with class names as keys.
    - alpha: float, optional (default=0.05)
      Significance level for correlations.
    """

    # Convert the one-hot array to a DataFrame for easier manipulation
    df = pd.DataFrame(one_hot_labels, columns=link_dict.keys())

    # Calculate the correlation matrix and the p-value matrix
    correlation_matrix = df.corr()
    p_value_matrix = pd.DataFrame(np.zeros(correlation_matrix.shape), columns=correlation_matrix.columns, index=correlation_matrix.index)

    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            if i != j:
                _, p_value_matrix.iat[i, j] = pearsonr(df.iloc[:, i], df.iloc[:, j])

    # Create a mask for significant correlations
    significant_mask = p_value_matrix < alpha
    # Set diagonal of correlation_matrix_test to value 0
    np.fill_diagonal(correlation_matrix.values, 0.)
    return correlation_matrix, significant_mask





configuration = {'input_shape': 1536,#2304, # 1536, 3072
                 'architecture':'MLP', #MLP, CNN, COMAL_NO_SHARED, COMAL_SHARED, MLP_DROPOUT
                 'distance':'euclidean',#'cosine',#'euclidean',
                 'activation':'swish', #'swish',#tf.math.sin,
                 'learning_rate':0.0005,
                 'temperature':1,
                 'base_temperature':1,
                 'coarse_labels_temperature':0.1,
                 'optimizer':tf.keras.optimizers.serialize(tf.keras.optimizers.AdamW()),#tf.keras.optimizers.legacy.Adam(),
                 'output_dimensions':15,
                 'depth':5,#5
                 'depth_encoder':5,
                 'pert': 0.01,
                 'epochs':50,
                 'batch_size':256,
                 'similarity_to_class':'kNN_chunks',#'w_average'#'kNN_chunks', #'centroid','w_average','kNN_chunks'
                 'val_batch':200000,
                 'kNN':50,
                 'n_jobs':-1,
                 'top_perf':1000,

                 'classic_kNN':True,
                 
                 'eval_method':'multi', #NEW 'multi', 'unroll', 'isin', model/performance evaluation method
                 'threshold':0.6,  #NEW threshold to be used in 'multi' to go from relation probabilities to relation occurrences
                 
                 'loss':'similarity_adaptive', ##NEW 'jaccard' 'similarity'  'similarity_multiMLP'  'hierarchical_similarity'  
                 'jaccard_threshold':0.3, ##NEW threshold for 'jaccard' loss
                 'lambda_coarse':0.5,

                 'balance_triples':False,
                 'quantile_cutoff_balancing':0.95,
                 'balance_labels':False, ## NEW! No balance, triple balance, triple+labels balance
                 
                 'bag_threshold':0.,   ##NEW (when using 'multi') percentage cutoff for bag label appearence
                 'get_thresholds':False,
                 'return_headtail':False,
                 'other_top_perf':[100,200,300],
                 'calibrated':False,
                 'harmonic_score':True,
                 'cls':False,
                 'q':None,
                 'add_NA_class':False,
                 'branch_idx':None,
                 'dropout_rate':0.2,


                 #Adaptive loss parameters
                 'T_end':0.05,
                 'T_warmup_epochs':0,
                 'T_hold_epochs':0,
                 'T_per_batch':True,
                 'T_mode':'cosine', #linear, cosine, exponential
                 'k_per_class':5 #number of samples per class in each batch if using balanced batches
                }


if configuration['eval_method']=='multi':
    configuration['return_headtail']=False



## want to extract dataset name from sh command line
dataset=sys.argv[1]
print(f'Running on dataset {dataset}')



dict_dataset={}
dict_dataset = {
    'nyt10d': {'train': 'Final/NYT10D/Train/', 
               'test': 'Final/NYT10D/Test/', 
               'val': None},
    'nyt10m': {'train': 'Final/NYT10m/Train/',
               'test': 'Final/NYT10m/Test/',
               'val': 'Final/NYT10m/Val/'},
    'wiki20m': {'train': 'Final/Wiki20m/Train/',
                'test': 'Final/Wiki20m/Test/',
                'val': 'Final/Wiki20m/Val/'},
    'disrex': {'train': 'Final/DisRex/Train/',
               'test': 'Final/DisRex/Test/',
               'val': 'Final/DisRex/Val/'},
    'wiki20distant': {'train': 'Final/Wiki20mDistant/Train/',
                      'test': 'Final/Wiki20m/Test/',
                      'val': 'Final/Wiki20mDistant/Val/'}
}


path_train=dict_dataset[dataset]['train']
path_test=dict_dataset[dataset]['test']
if dict_dataset[dataset]['val'] is not None:
    path_val=dict_dataset[dataset]['val']
    validation=True
else:
    validation=False

#Configuration parameters for the datasets (see on section VI.B of the paper)
config_params = {
    'nyt10m': {'c': 0.4, 'k': 20},
    'nyt10d': {'c': 0.7, 'k': 100},
    'disrex': {'c': 0.5, 'k': 50},
    'wiki20m': {'c': 0.5, 'k': 100},
    'wiki20distant': {'c': 0.7, 'k': 150}
}

#Select appropriate configuration parameters for the dataset
configuration['kNN'] = config_params[dataset]['k']
configuration['threshold'] = config_params[dataset]['c']



file_list_train=glob.glob(path_train+"*.pkl")
file_list_test=glob.glob(path_test+"*.pkl")
if validation:
    file_list_val=glob.glob(path_val+"*.pkl")

#Create a dir wirth the dataset name if it does not exist to save results

if not os.path.exists(dataset+"_results"):
    os.makedirs(dataset+"_results", exist_ok=True)

list_results=[]
indices = []

for embedding_type in ['average']:


    tracker1 = EmissionsTracker()
    tracker1.start()

    if embedding_type=='boundary' or embedding_type=='mean_max':
        configuration['input_shape'] = 3072
    else:
        configuration['input_shape'] = 1536

    ## create link dictionary and training and test datasets 
    configuration['class_probabilities'],link_dict, training_dataset, train_df= from_file_list_to_tfdataset(file_list_train,  #, coarse_relation_dict 
                                                            balance_triples=configuration['balance_triples'], 
                                                            quantile_cutoff=configuration['quantile_cutoff_balancing'],
                                                            balance_labels=configuration['balance_labels'],
                                                            cls=configuration['cls'],
                                                            return_df=True,
                                                            get_thresholds=True,
                                                            embedding_type=embedding_type)
    _, test_dataset, test_df = from_file_list_to_tfdataset(file_list_test, 
                                                relation_dict=link_dict,
                                                return_df=True,
                                                return_headtail=configuration['return_headtail'],
                                                cls=configuration['cls'],
                                                embedding_type=embedding_type)
    if validation:
        _, validation_dataset,val_df = from_file_list_to_tfdataset(file_list_val, 
                                                            relation_dict=link_dict, 
                                                            balance_triples=False,  
                                                            quantile_cutoff=configuration['quantile_cutoff_balancing'],
                                                            balance_labels=False,
                                                            cls=configuration['cls'],
                                                            return_df=True,
                                                            embedding_type=embedding_type)

    emissions: float = tracker1.stop()  # returns emissions in kg CO2eq
    
    # Get energy consumption in kWh
    kwh_consumed_dataset_loading = tracker1._total_energy.kWh  # total energy (kWh)
    seeds=[42,71,88,101,300]

    #Aggiungere modalità selezione embedding

    #Select a number of runs to perform the evaluation on the same dataset
    for seed in seeds:

        tracker2 = EmissionsTracker()
        tracker2.start()

        random.seed(seed)   
        tf.random.set_seed(seed)
        np.random.seed(seed)

        training_dataset_even, per_batch_size = build_balanced_class_batches_from_df(
            df=train_df,                   # your pandas DF
            input_col="Concatenated",             # <-- change to your column
            label_col="link_name",  
            seed=seed,          # <-- change to your column
            k_per_class=configuration.get('k_per_class', 5)
        )
            

        configuration['number_of_classes'] = len(link_dict)

        # prepare per-run logfile named with dataset, seed and embedding type
        log_dir = f"{dataset}_results"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{dataset}_{seed}_{embedding_type}.log")
        log_file = open(log_path, 'w', encoding='utf-8')

        # redirect stdout and stderr to both terminal and logfile
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)



        # buffer_size=len(training_dataset)
        # configuration['val_batch'] = len(training_dataset)
        # training_dataset=training_dataset_unshuffled.shuffle(buffer_size, reshuffle_each_iteration=True, seed = seed)
        # test_dataset=test_dataset_unshuffled.shuffle(buffer_size, reshuffle_each_iteration=True, seed = seed)
        # if validation:
        #     validation_dataset=validation_dataset_unshuffled.shuffle(buffer_size, reshuffle_each_iteration=True, seed = seed)

        tf.keras.backend.clear_session()

        model = CreateModel(training_configuration = configuration)
        model.summary()

        temp_cb = TemperatureSchedulerAuto(
            T_start=configuration['temperature'],
            T_end=configuration['T_end'],
            mode=configuration['T_mode'],  # 'cosine'|'exp'|'linear'
            per_batch=configuration['T_per_batch'],
            warmup_epochs=configuration['T_warmup_epochs'],
            hold_epochs=configuration['T_hold_epochs'])

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if validation else 'loss',
            min_delta=0 if validation else 1e-4,
            patience=10,
            verbose=0,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0
        )

        if validation:
            history = model.fit(
                training_dataset_even,                  # already batched & infinite
                epochs=configuration['epochs'],    # interpret as "how many rounds"
                validation_data=validation_dataset.batch(configuration['batch_size']),
                steps_per_epoch=256,
                callbacks=[callback, temp_cb]
            )
        else:
            history = model.fit(
                training_dataset_even,
                epochs=configuration['epochs'],    # number of rounds
                steps_per_epoch=256,
                callbacks=[callback, temp_cb]
            )

        #Uncomment if you want save the weights of the model after training
        #model.save_weights(f'{dataset}_model_{seed}.h5')   


        
        row_keys=[]
        for kNN in [70]:
            configuration['kNN']=kNN
            for threshold in [0.4]:
                configuration['threshold']=threshold
                for harmonic_score in [True]:
                    bool_bayesian=False
                    calibrated=False
                    row_results=[]
                    configuration['get_thresholds']=bool_bayesian
                    configuration['calibrated']=calibrated
                    configuration['harmonic_score']=harmonic_score

                    test_y_true, test_y_pred, test_y_score, results_perf, best_indices, test_y_pred_at_R,  norm_w_batch,test_batch_weight,total_distances,total_indices = evaluation_and_performance(test_configuration=configuration,
                                                                                    training_dataset=training_dataset,
                                                                                    test_dataset=test_dataset,
                                                                                    model=model
                                                                                    )
                    
                    if bool_bayesian==True and calibrated==True and harmonic_score==True:
                        print('\n\n RESULT USING CALIBRATED BAYESIAN FORMULATION HARMONIC')
                        row_keys.append('BCH')
                    elif bool_bayesian==True and calibrated==True and harmonic_score==False:
                        print('\n\n RESULT USING CALIBRATED BAYESIAN FORMULATION MEAN')
                        row_keys.append('BCM')
                    elif bool_bayesian==True and calibrated==False and harmonic_score==True:
                        print('\n\n RESULT USING BAYESIAN FORMULATION HARMONIC')
                        row_keys.append('BH')
                    elif bool_bayesian==True and calibrated==False and harmonic_score==False:
                        print('\n\n RESULT USING BAYESIAN FORMULATION MEAN')
                        row_keys.append('BM')
                    elif bool_bayesian==False and calibrated==True and harmonic_score==True:
                        print('\n\n RESULT USING CALIBRATED LIKELIHOOD FORMULATION HARMONIC')
                        row_keys.append('LCH')
                    elif bool_bayesian==False and calibrated==True and harmonic_score==False:
                        print('\n\n RESULT USING CALIBRATED LIKELIHOOD FORMULATION MEAN')
                        row_keys.append('LCM')
                    elif bool_bayesian==False and calibrated==False and harmonic_score==True:
                        print('\n\n RESULT USING LIKELIHOOD FORMULATION HARMONIC')
                        row_keys.append('LH')
                    else: 
                        print('\n\n RESULT USING LIKELIHOOD FORMULATION MEAN')
                        row_keys.append('LM')
                    print('Performance on the whole validation set:\n')
                    for key in results_perf['total']:
                        if key!='list_f1':
                            perf=results_perf['total'][key]
                            print(f'{key}: {perf}')
                            row_results.append(perf)

                    for atX in configuration['other_top_perf']:
                        top_string='@'+str(atX)
                        print(f'\n\Performance {top_string} on validation set:\n')
                        for key in results_perf[f'@'+str(atX)]:
                            if key!='list_f1':
                                perf=results_perf[f'@'+str(atX)][key]
                                print(f'{key}: {perf}')
                                row_results.append(perf)

                    top_string='@'+str(configuration['top_perf'])
                    print(f'\n\Performance {top_string} on validation set:\n')
                    for key in results_perf[f'@'+str(configuration['top_perf'])]:
                        if key!='list_f1':
                            perf=results_perf[f'@'+str(configuration['top_perf'])][key]
                            print(f'{key}: {perf}')
                            row_results.append(perf)

                    PatR=precision_at_R_ml(test_y_true, test_y_pred_at_R)
                    print(f'Precision @R: {PatR}')
                    
                    row_results.append(PatR)
                    row_results.append(configuration['kNN'])

                    #Calculate correlation matrix and p-value matrix

                    # Sample one-hot encoded data (replace this with your actual data)
                    one_hot_labels = test_y_true

                    # Convert the one-hot array to a DataFrame for easier manipulation
                    df = pd.DataFrame(one_hot_labels, columns=link_dict.keys())

                    # Calculate the correlation matrix and the p-value matrix
                    correlation_matrix_test = df.corr()
                    p_value_matrix = pd.DataFrame(np.zeros(correlation_matrix_test.shape), columns=correlation_matrix_test.columns, index=correlation_matrix_test.index)

                    for i in range(len(correlation_matrix_test.columns)):
                        for j in range(len(correlation_matrix_test.columns)):
                            if i != j:
                                _, p_value_matrix.iat[i, j] = pearsonr(df.iloc[:, i], df.iloc[:, j])

                    # Set a significance level
                    alpha = 0.05

                    # Create a mask for significant correlations
                    significant_mask = p_value_matrix < alpha

                    #Set diagonal of correlation_matrix_train to value 0
                    np.fill_diagonal(correlation_matrix_test.values, 0.)


                    df=pd.DataFrame(np.array(train_df['link_name'].tolist()), columns=link_dict.keys())

                    # Calculate the correlation matrix and the p-value matrix
                    correlation_matrix_train = df.corr()
                    p_value_matrix = pd.DataFrame(np.zeros(correlation_matrix_train.shape), columns=correlation_matrix_train.columns, index=correlation_matrix_train.index)

                    for i in range(len(correlation_matrix_train.columns)):
                        for j in range(len(correlation_matrix_train.columns)):
                            if i != j:
                                _, p_value_matrix.iat[i, j] = pearsonr(df.iloc[:, i], df.iloc[:, j])

                    # Set a significance level
                    alpha = 0.05

                    # Create a mask for significant correlations
                    significant_mask = p_value_matrix < alpha

                    #Set diagonal of correlation_matrix_train to value 0
                    np.fill_diagonal(correlation_matrix_train.values, 0.)

                    # Sample one-hot encoded data (replace this with your actual data)
                    one_hot_labels = test_y_pred

                    # Convert the one-hot array to a DataFrame for easier manipulation
                    df = pd.DataFrame(one_hot_labels, columns=link_dict.keys())

                    # Calculate the correlation matrix and the p-value matrix
                    correlation_matrix_pred = df.corr()
                    p_value_matrix = pd.DataFrame(np.zeros(correlation_matrix_pred.shape), columns=correlation_matrix_pred.columns, index=correlation_matrix_pred.index)

                    for i in range(len(correlation_matrix_pred.columns)):
                        for j in range(len(correlation_matrix_pred.columns)):
                            if i != j:
                                _, p_value_matrix.iat[i, j] = pearsonr(df.iloc[:, i], df.iloc[:, j])

                    # Set a significance level
                    alpha = 0.05

                    # Create a mask for significant correlations
                    significant_mask = p_value_matrix < alpha

                    #Set diagonal of correlation_matrix_train to value 0
                    np.fill_diagonal(correlation_matrix_pred.values, 0.)


                    #Fill nan with zeros in correlation_matrix_pred
                    correlation_matrix_pred=correlation_matrix_pred.fillna(0)
                    correlation_matrix_test=correlation_matrix_test.fillna(0)

                    distance = np.linalg.norm(correlation_matrix_test - correlation_matrix_pred, 'fro')

                    row_results.append(distance)
                    row_results.append(configuration['threshold'])

                    # if bool_bayesian==False and calibrated==False and harmonic_score==True:
                    #     #Save correlation matrices with dataset name + iteration name as pickle
                    #     correlation_matrix_test.to_pickle(f"{dataset}_LH_{seed}_correlation_matrix_test_{seed}_KNN_{configuration['kNN']}_threshold_{configuration['threshold']}.pkl")

                    ####CSD metric calculation#####

                    test_correlation_matrix, significant_mask = significant_correlation_matrix(test_y_true, link_dict, alpha=0.05)
                    train_correlation_matrix, significant_mask = significant_correlation_matrix(np.array(train_df['link_name'].tolist()), link_dict, alpha=0.05)
                    pred_correlation_matrix, significant_mask = significant_correlation_matrix(test_y_pred, link_dict, alpha=0.05)
                    correlation_matrix_pred=pred_correlation_matrix.fillna(0)
                    correlation_matrix_test=test_correlation_matrix.fillna(0)
                    #Calculate CSD normalized worst
                    distance = np.linalg.norm(correlation_matrix_test-correlation_matrix_pred)/np.linalg.norm(1+np.abs(correlation_matrix_test))
                    row_results.append(distance)               
                    #Calculate CSD normalized ideal
                    distance = np.linalg.norm(correlation_matrix_test-correlation_matrix_pred)/np.sqrt(2*correlation_matrix_test.shape[0]*(correlation_matrix_test.shape[0]-1))
                    row_results.append(distance)  

                    emissions: float = tracker2.stop()  # returns emissions in kg CO2eq
                    # Get energy consumption in kWh
                    kwh_consumed_training_and_testing = tracker2._total_energy.kWh  # total energy (kWh)
                    total_KWh = kwh_consumed_training_and_testing + kwh_consumed_dataset_loading
                    row_results.append(total_KWh)
                    row_results.append(seed)
                    row_results.append(embedding_type)
                    current_label = row_keys[-1] if len(row_keys) > 0 else f'run_{seed}'
                    #Se non esiste la cartella la creala
                    if not os.path.exists(f'{dataset}_results_balanced_gradual'):
                        os.makedirs(f'{dataset}_results_balanced_gradual', exist_ok=True)
                    pd.to_pickle(test_y_pred, f'{dataset}_results_balanced_gradual/{dataset}_predictions_{seed}_{embedding_type}_{current_label}.pkl')

                    list_results.append(row_results)
                    indices.append(current_label)

        # end of seed run: restore stdout/stderr and close logfile
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            log_file.close()
        except Exception:
            pass


columns=['micro','Macro']
for el in configuration['other_top_perf']:
    columns+=[f'm@{el}',f'M@{el}']
columns+=['m@1000','M@1000','P@R','KNN','Distance','Threshold','CSD Worst','CSD Ideal','KWh','Seed','EmbeddingType']
#df_res=pd.DataFrame((np.array(list_results)*100).round(2),index=indices,columns=columns)
df_res = pd.DataFrame(np.array(list_results), index=indices, columns=columns)

#Save df_res with dataset name + iteration name as pickle 
df_res.to_pickle(f'gs_results/{dataset}__results_balanced_gradual_performance.pkl')

