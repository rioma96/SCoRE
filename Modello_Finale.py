import tensorflow as tf
import pickle as pickle
import numpy as np
import pandas as pd
import glob
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
import sys
#tracker.start()
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# to include the Repo code without installing in the environment
import sys
sys.path.append('../')

from CBKGE.NN_creation_and_dependencies import *
from CBKGE.NN_preproc import *
from CBKGE.utilities_validation import *



configuration = {'input_shape': 1536,#2304, # 1536,
                 'architecture':'MLP',
                 'distance':'euclidean',#'cosine',#'euclidean',
                 'activation':'swish', #'swish',#tf.math.sin,
                 'learning_rate':0.0005,
                 'temperature':0.01,
                 'base_temperature':1,
                 'optimizer':tf.keras.optimizers.serialize(tf.keras.optimizers.AdamW(), use_legacy_format=True),#tf.keras.optimizers.legacy.Adam(),
                 'output_dimensions':15,
                 'depth':5,#5
                 'pert': 0.01,
                 'epochs':50,
                 'batch_size':256,
                 'similarity_to_class':'kNN_chunks',#'w_average'#'kNN_chunks',  #'centroid','w_average','kNN_chunks'
                 'val_batch':200000,
                 'kNN':15,
                 'n_jobs':-1,
                 'top_perf':1000,
                 
                 'eval_method':'multi', #NEW 'multi', 'unroll', 'isin', model/performance evaluation method
                 'threshold':0.5,  #NEW threshold to be used in 'multi' to go from relation probabilities to relation occurrences
                 
                 'loss':'similarity', ##NEW 'jaccard' 'similarity'      
                 'jaccard_threshold':0.2, ##NEW threshold for 'jaccard' loss
                 
                 'balance_triples':False,
                 'quantile_cutoff_balancing':0.95,
                 'balance_labels':False, ## NEW! No balance, triple balance, triple+labels balance
                 
                 'bag_threshold':0.,   ##NEW (when using 'multi') percentage cutoff for bag label appearence
                 'get_thresholds':True,
                 'return_headtail':False,
                 'other_top_perf':[100,200,300],
                 'calibrated':False,
                 'harmonic_score':True
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
    'nyt10m': {'c': 0.6, 'k': 50},
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


## create link dictionary and training and test datasets 
configuration['class_probabilities'],link_dict, training_dataset, train_df = from_file_list_to_tfdataset(file_list_train, 
                                                          balance_triples=configuration['balance_triples'], 
                                                          quantile_cutoff=configuration['quantile_cutoff_balancing'],
                                                          balance_labels=configuration['balance_labels'],
                                                          cls=False,
                                                          return_df=True,
                                                          get_thresholds=True)
_, test_dataset = from_file_list_to_tfdataset(file_list_test, 
                                            relation_dict=link_dict,
                                            return_df=False,
                                            return_headtail=configuration['return_headtail'],
                                            cls=False)
if validation:
    _, validation_dataset = from_file_list_to_tfdataset(file_list_val, 
                                                        relation_dict=link_dict, 
                                                        balance_triples=False,  
                                                        quantile_cutoff=configuration['quantile_cutoff_balancing'],
                                                        balance_labels=False,
                                                        cls=False)

buffer_size=len(training_dataset)
configuration['val_batch'] = len(training_dataset)
training_dataset=training_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
test_dataset=test_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
if validation:
    validation_dataset=validation_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)


#Select a number of runs to perform the evaluation on the same dataset
for run in range(3):


    tf.keras.backend.clear_session()
    model = CreateModel(training_configuration = configuration)
    model.summary()




    if validation:
        callback= tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0)


        history = model.fit(training_dataset.batch(configuration['batch_size']),
                                                    epochs = configuration['epochs'], 
                                                    validation_data=validation_dataset.batch(configuration['batch_size']), 
                                                    callbacks=[callback])

    else:
        callback= tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-4,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0)


        history = model.fit(training_dataset.batch(configuration['batch_size']),
                            epochs = configuration['epochs'], 
                            callbacks=[callback]) 

    #Uncomment if you want save the weights of the model after training
    #model.save_weights(f'{dataset}_model_{run}.h5')   


    list_results=[]
    row_keys=[]

    for bool_bayesian in [False]:
        for calibrated in [False]:
            for harmonic_score in [True]:
                row_results=[]
                configuration['get_thresholds']=bool_bayesian
                configuration['calibrated']=calibrated
                configuration['harmonic_score']=harmonic_score

                test_y_true, test_y_pred, test_y_score, results_perf, best_indices, test_y_pred_at_R = evaluation_and_performance(test_configuration=configuration,
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

                if bool_bayesian==False and calibrated==False and harmonic_score==True:
                    #Save correlation matrices with dataset name + iteration name as pickle
                    correlation_matrix_test.to_pickle(f"{dataset}_LH_{run}_correlation_matrix_test_{run}_KNN_{configuration['kNN']}_threshold_{configuration['threshold']}.pkl")

                list_results.append(row_results)


    indices=row_keys
    columns=['micro','Macro']
    for el in configuration['other_top_perf']:
        columns+=[f'm@{el}',f'M@{el}']
    columns+=['m@1000','M@1000','P@R','KNN','Distance','Threshold']
    df_res=pd.DataFrame((np.array(list_results)*100).round(2),index=indices,columns=columns)


    #Save df_res with dataset name + iteration name as pickle 
    df_res.to_pickle(f'{dataset}_results_performance_{run}.pkl')


