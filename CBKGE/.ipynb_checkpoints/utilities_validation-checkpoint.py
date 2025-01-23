########################
##### FROM SIMILARITY TO CLASSIFICATION FUNCTIONS
########################

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from CBKGE.NN_creation_and_dependencies import CreateModelSkeleton, CreateModel
from CBKGE.NN_preproc import df_to_tf_dataset


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def evaluation_and_performance(test_configuration:dict,
                               training_dataset:tf.data.Dataset,
                               test_dataset:tf.data.Dataset,
                               model:tf.keras.models):
    
    #from sklearn.metrics import confusion_matrix 
    eval_method = test_configuration['eval_method']

    if eval_method!='multi':
        test_y_true, test_y_pred, top_indices, test_y_score = similarity_to_class(test_configuration=test_configuration,
                                                                                training_dataset=training_dataset,
                                                                                test_dataset=test_dataset,
                                                                                model=model)
    else:
        test_y_true, test_y_pred, top_indices, test_y_score, test_headtails = similarity_to_class(
                                                                            test_configuration=test_configuration,
                                                                            training_dataset=training_dataset,
                                                                            test_dataset=test_dataset,
                                                                            model=model
                                                                            )
        
        
        
    
    if eval_method!='multi':
        fn_micro=f1_micro_yt
        fn_macro=f1_macro_yt
    else:
        fn_micro=f1_micro_yt_ml
        fn_macro=f1_macro_yt_ml
        

    results_perf={}
    #cm=confusion_matrix(test_y_true, test_y_pred)
    results_perf['total']={'microf1':fn_micro(test_y_true, test_y_pred),
                'macrof1':fn_macro(test_y_true, test_y_pred)}


    #top_cm=confusion_matrix(test_y_true[top_indices], test_y_pred[top_indices])
    results_perf['@'+str(test_configuration['top_perf'])]={
                'microf1':fn_micro(test_y_true[top_indices], test_y_pred[top_indices]),
                'macrof1':fn_macro(test_y_true[top_indices], test_y_pred[top_indices])}
    
    if eval_method!='multi':
        return  test_y_true, test_y_pred, test_y_score, results_perf
    else: 
        return  test_y_true, test_y_pred, test_y_score, test_headtails, results_perf





def similarity_to_class(test_configuration:dict,
                        training_dataset:tf.data.Dataset,
                        test_dataset:tf.data.Dataset,
                        model:tf.keras.models):
    
    '''
    This function discriminates between different available methods to transform a 
    similarity output into a classification output. 
    Avaliable procedures are:
        - centroid: compute class centroids based on training set labels and classify 
                    test points accordingly
        - w_average: compute weighted expectation label of test set points using the 
                     same distance formulation used in training
        - kNN_chunks: compute weighted expectation label of test set points using the 
                      same distance formulation used in training and considering only 
                      the kNN from each trianing set chunk
    
    '''

    procedure = test_configuration['similarity_to_class']
    eval_method = test_configuration['eval_method']
        
    if procedure=='centroid':
        f_class=centroid_prediction
        
    elif procedure=='w_average':      
        f_class = average_prediction

    elif procedure=='kNN_chunks' and eval_method!='multi':
        f_class = chunk_kNN_prediction
        
    elif procedure=='kNN_chunks' and eval_method=='multi':
        f_class = chunk_kNN_ml_prediction
        
    if (procedure!='kNN_chunks') and (eval_method=='multi'):
        raise Exception("Sorry, 'multi' evaluation method only implemented for kNN_chunks")
    
       
    if eval_method!='multi':
        test_y_true, test_y_pred, top_indices, test_y_score = f_class(training_dataset=training_dataset,
                                                                    test_dataset=test_dataset,
                                                                    model=model,
                                                                    test_configuration=test_configuration)
        return test_y_true, test_y_pred, top_indices, test_y_score
    
    else:
        test_y_true, test_y_pred, top_indices, test_y_score, headtails = f_class(training_dataset=training_dataset,
                                                                                test_dataset=test_dataset,
                                                                                model=model,
                                                                                test_configuration=test_configuration)
        return test_y_true, test_y_pred, top_indices, test_y_score, headtails


    


## Unroll evaluation method: unroll multilabel records in the test set
def unroll_eval(y_true, y_pred, y_score):
    unroll_indices , unroll_ytrue = np.split(np.argwhere(y_true),2,1)
    unroll_y_pred = y_pred[unroll_indices]
    unroll_y_score = y_score[unroll_indices]
 
    return unroll_ytrue, unroll_y_pred, unroll_y_score


##Is-in evaluation method: override y_true so to consider as a positive prediction whatever label among the true ones
def isin_eval(y_true, y_pred, y_score):
    ## function to be used in apply to set y_true value
    def function_select_y(row):
        import random
        if row['y_pred'] in row['y_true']:
            return row['y_pred']
        else:
            return random.choice(row['y_true'])

    df_isin_true_values=pd.DataFrame()
    # create a df with 'row' = record index and 'y_true' = multilabel lists [[l11,l12],[l2],[l31,l32,l33],...]
    df_isin_true_values['y_true']=pd.DataFrame(np.argwhere(y_true),columns=['row','col']).groupby(['row'])['col'].apply(list).reset_index(drop=True)
    df_isin_true_values['y_pred']=y_pred
    isin_ytrue=df_isin_true_values.apply(function_select_y,axis=1).values

    return isin_ytrue, y_pred, y_score


def centroid_prediction(training_dataset:tf.data.Dataset,
                        test_dataset:tf.data.Dataset,
                        model:tf.keras.models,
                        test_configuration:dict
                        ):
    '''
    This function computes the class centroids starting from the multi-label training set.
    Subsequently, it computes the distance between validation points and the centroids and 
    selects as predicted label, the class associated with the centroid of minimum distance. 
    
    Inputs:
    - training_dataset: elements shape ([training data,output dimensions], [training data,label_one_ho_encoding_size])
    - test_dataset: elements shape ([test data,output dimensions], [test data, 1 or label_one_ho_encoding_size])
    - model: neural network trained with supervised contrastive learning
    - distance: selected distance measure (it should match the one used in training the contrastive network)
    - val_batch: batch size to be used in the validation stage
    
    Note that the validation set should appear in a ground truth fashion (single label). Using a
    one-hot encoding of the validation set labels can be handled by the function.
    
    '''
    
    val_batch = test_configuration['val_batch']
    n_jobs = test_configuration['n_jobs']
    top_perf = test_configuration['top_perf']
    model_weights = model.get_weights()
    eval_method = test_configuration['eval_method']
    
    ##compute centroids
    count_rel=0  #will contain the counting of class occurrences appearing in the training set
    centroids=0  #will contain the embedding of the centriod of each class

    #compute sum of training batch embeddings per label vector
    results_training = Parallel(n_jobs=n_jobs)(
        delayed(process_training_batch_centroids)(batch_x, batch_y, model_weights, test_configuration) for batch_x, batch_y in training_dataset.batch(val_batch)
    )

    for sum_batch_coord, sum_batch_y in results_training:
        centroids += sum_batch_coord
        count_rel += sum_batch_y

    centroids = centroids / (count_rel + 1e-20)
    
    # compute distance of validation set from centroids and predict labels
    
    val_y_pred=[]  #will contain the predicted labels for each validation set element
    val_y_true=[]  #will contain the true labels for each validation set element
    val_y_scores=[]

    results_testing = Parallel(n_jobs=n_jobs)(
        delayed(process_test_batch_centroids)(batch_x, batch_y, model_weights, test_configuration, centroids) for batch_x, batch_y in test_dataset.batch(val_batch)
    )

    for val_y_true_batch, val_y_pred_batch , val_y_score_batch in results_testing:
        val_y_pred.append(val_y_pred_batch)
        val_y_true.append(val_y_true_batch)
        val_y_scores.append(val_y_score_batch)

    val_y_pred = np.concatenate(val_y_pred)
    val_y_true = np.concatenate(val_y_true)
    val_y_scores = np.concatenate(val_y_scores)
    
    if eval_method=='unroll':
        fn=unroll_eval     
    elif eval_method=='isin':
        fn=isin_eval
        
    val_y_true, val_y_pred, val_y_scores=fn(val_y_true, val_y_pred, val_y_scores)
    val_y_true=val_y_true.flatten()
    val_y_pred=val_y_pred.flatten()
    val_y_scores=val_y_scores.flatten()

    top_indices = np.argsort(val_y_scores)[:top_perf]

    return val_y_true, val_y_pred, top_indices, val_y_scores


def process_training_batch_centroids(batch_x, batch_y, model_weights, configuration):
    model=CreateModelSkeleton(configuration)
    model.set_weights(model_weights)
    
    batch_y_pred = model.predict(batch_x, verbose=0)
    sum_batch_coord = np.matmul(batch_y_pred.T,batch_y.numpy())
    sum_batch_y = np.sum(batch_y.numpy(), 0, keepdims=True)
    return sum_batch_coord, sum_batch_y


def process_test_batch_centroids(batch_x, batch_y, model_weights, configuration, centroids):
    distance = configuration['distance']

    model=CreateModelSkeleton(configuration)
    model.set_weights(model_weights)
    
    batch_embedding = model.predict(batch_x, verbose=0)
    distances_embedding_centroids = distance_matrix_numpy(batch_embedding, z2=centroids.T, dist=distance)
    val_y_pred_batch = np.argmin(distances_embedding_centroids, 1)
    val_y_score_batch = np.min(distances_embedding_centroids, 1)

    assert len(batch_y.shape) == 2

    return batch_y, val_y_pred_batch, val_y_score_batch



def average_prediction(training_dataset:tf.data.Dataset,
                       test_dataset:tf.data.Dataset,
                       model:tf.keras.models,
                       test_configuration:dict):
    '''
    This function computes the average weighted label prediction from the multi-label training set.
    It computes the distance between validation and training points embeddings and 
    selects as predicted label, the one having highest probability. 
    
    Inputs:
    - training_dataset: elements shape ([training data,output dimensions], [training data,label_one_ho_encoding_size])
    - test_dataset: elements shape ([test data,output dimensions], [test data, 1 or label_one_ho_encoding_size])
    - model: neural network trained with supervised contrastive learning
    - distance: selected distance measure (it should match the one used in training the contrastive network)
    - val_batch: batch size to be used in the validation stage
    - temperature: hyperparameter to be used in computing label weights
    
    Note that the validation set should appear in a ground truth fashion (single label). Using a
    one-hot encoding of the validation set labels can be handled by the function.
    
    '''
    
    val_batch = test_configuration['val_batch']
    n_jobs = test_configuration['n_jobs']
    top_perf = test_configuration['top_perf']
    eval_method = test_configuration['eval_method']
    model_weights = model.get_weights()


    test_y_pred = []
    test_y_score = []
    test_y_true = []
    

    #load each chunck of validation data and compute embeddings
    for test_batch_x, test_batch_y in test_dataset.batch(val_batch):
        test_batch_embedding = model.predict(test_batch_x, verbose=0)

        test_batch_weights=0. ## will contain weighted one-hot econding labels of the val chunck
        test_batch_sum_weights=0. ## will contain total weight, to normaliza predictions.
        

        #load each chunck of validation data and compute embeddings
        results = Parallel(n_jobs=n_jobs)(
                    delayed(process_train_batch_average)(train_batch_x, 
                                                        train_batch_y,
                                                        test_batch_embedding,
                                                        model_weights,
                                                        test_configuration
                    ) for train_batch_x, train_batch_y in training_dataset.batch(val_batch)
                )

        for sum_weights, weights in results:
            test_batch_sum_weights += sum_weights
            test_batch_weights += weights

        # Compute val chunk predicted labels using argmax and add them to the validation set label array
        norm_w_batch =  (test_batch_weights / np.sum(test_batch_sum_weights, 1, keepdims=True))
        y_pred_batch = np.argmax(norm_w_batch, 1)
        test_y_score_batch = np.max(norm_w_batch, 1)
        test_y_pred.append(y_pred_batch)
        test_y_score.append(test_y_score_batch)
        test_y_true.append(test_batch_y)
        
    assert len(test_batch_y.shape) == 2
        
    test_y_true = np.concatenate(test_y_true)
    test_y_pred = np.concatenate(test_y_pred)
    test_y_score = np.concatenate(test_y_score)
    
    
    if eval_method=='unroll':
        fn=unroll_eval     
    elif eval_method=='isin':
        fn=isin_eval
       
    
    test_y_true, test_y_pred, test_y_score=fn(test_y_true, test_y_pred, test_y_score)
    
    test_y_true=test_y_true.flatten()
    test_y_pred=test_y_pred.flatten()
    test_y_score=test_y_score.flatten()

    top_indices = np.argsort(-test_y_score)[:top_perf]

    return test_y_true, test_y_pred, top_indices, test_y_score
    

# Define the parallelizable function for the inner loop
def process_train_batch_average(train_batch_x, 
                                train_batch_y,
                                test_batch_embedding, 
                                model_weights,
                                test_configuration):
    
    distance = test_configuration['distance']
    temperature = test_configuration['temperature']
    prediction_method = test_configuration['prediction_method']

    model=CreateModelSkeleton(test_configuration)
    model.set_weights(model_weights)

    train_batch_embedding = model.predict(train_batch_x, verbose=0)
    distances_embedding = distance_matrix_numpy(test_batch_embedding, z2=train_batch_embedding, dist=distance)

    #compute weights
    weights=np.exp(-distances_embedding/temperature)
    sum_weights=np.sum(weights,1,keepdims=True)
    #compute label weights
    weights=np.matmul(weights,train_batch_y.numpy()) #[val batch, n classes]

    return sum_weights, weights
    



def chunk_kNN_prediction(training_dataset:tf.data.Dataset,
                         test_dataset:tf.data.Dataset,
                         model:tf.keras.models,
                         test_configuration:dict):
    '''
    This function computes the average weighted label prediction from the multi-label training set.
    It computes the distance between validation and training points embeddings and 
    selects as predicted label, the one having highest probability. 
    
    Inputs:
    - training_dataset: elements shape ([training data,output dimensions], [training data,label_one_ho_encoding_size])
    - test_dataset: elements shape ([test data,output dimensions], [test data, 1 or label_one_ho_encoding_size])
    - model: neural network trained with supervised contrastive learning
    - distance: selected distance measure (it should match the one used in training the contrastive network)
    - val_batch: batch size to be used in the validation stage
    - temperature: hyperparameter to be used in computing label weights
    
    Note that the validation set should appear in a ground truth fashion (single label). Using a
    one-hot encoding of the validation set labels can be handled by the function.
    
    '''
    from sklearn.neighbors import NearestNeighbors
    val_batch = test_configuration['val_batch']
    n_jobs = test_configuration['n_jobs']
    distance = test_configuration['distance']
    top_perf = test_configuration['top_perf']
    eval_method = test_configuration['eval_method']
    model_weights = model.get_weights()
    threshold = test_configuration['threshold']
    
    
    if distance == 'cosine_triangle':
        from scipy.spatial.distance import cosine
        def cosine_triangle(x,y):
            return np.sqrt(np.maximum(2* cosine(x,y),  np.finfo(float).eps))
        
        distance=cosine_triangle
        algo='brute'
    
    elif distance == 'euclidean_squared':
        from scipy.spatial.distance import sqeuclidean
        distance=sqeuclidean
        algo='brute'
        
    elif distance == 'euclidean':
        algo='ball_tree'
        
    elif distance == 'cosine':
        algo='brute'
   
    test_y_pred=[]
    test_y_true=[]
    test_y_score=[]
        
    
    #load each chunck of validation data and compute embeddings
    for test_batch_x, test_batch_y in test_dataset.batch(val_batch):

        test_batch_embedding = model.predict(test_batch_x, verbose=0)
        test_batch_weights=0. ## will contain weighted one-hot econding labels of the val chunck
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_train_batch_kNN)(train_batch_x, 
                                             train_batch_y, 
                                             test_batch_embedding, 
                                             model_weights, 
                                             test_configuration,
                                             algo       
            ) for train_batch_x, train_batch_y in training_dataset.batch(val_batch)
        )

        for weights in results:
            test_batch_weights += weights

        # Compute val chunk predicted labels using argmax and add them to the validation set label array
        if eval_method == 'multi':
            norm_w_batch = (test_batch_weights/(test_batch_weights.sum(0,keepdims=True)))[0]
            y_pred_batch = (norm_w_batch>=threshold).astype(int)
            test_y_score_batch = np.where(norm_w_batch>=threshold,norm_w_batch,0).sum(1)/((y_pred_batch).sum(1)+1e-20)
        else:
            norm_w_batch =  (test_batch_weights/test_batch_weights.sum(1,keepdims=True))
            y_pred_batch = np.argmax(norm_w_batch, 1)
            test_y_score_batch = np.max(norm_w_batch, 1)
        test_y_pred.append(y_pred_batch)
        test_y_score.append(test_y_score_batch)
        test_y_true.append(test_batch_y)

    assert len(test_batch_y.shape) == 2
    
    test_y_true = np.concatenate(test_y_true)
    test_y_pred = np.concatenate(test_y_pred)
    test_y_score = np.concatenate(test_y_score)
    
    
    if eval_method=='unroll':
        fn=unroll_eval     
    elif eval_method=='isin':
        fn=isin_eval
    
    if eval_method!='multi':
        test_y_true, test_y_pred, test_y_score = fn(test_y_true, test_y_pred, test_y_score)
        test_y_true=test_y_true.flatten()
        test_y_pred=test_y_pred.flatten()
        test_y_score=test_y_score.flatten()

    top_indices = np.argsort(-test_y_score)[:top_perf]

    return test_y_true, test_y_pred, top_indices, test_y_score


# Define the parallelizable function for the inner loop
def process_train_batch_kNN(train_batch_x, 
                            train_batch_y,
                            test_batch_embedding, 
                            model_weights,
                            test_configuration,
                            algo
                            ):
    
    kNN = test_configuration['kNN']
    distance = test_configuration['distance']
    temperature = test_configuration['temperature']
    n_jobs = test_configuration['n_jobs']
    top_perf = test_configuration['top_perf']
    eval_method=test_configuration['eval_method']

    model=CreateModelSkeleton(test_configuration)
    model.set_weights(model_weights)
    train_batch_embedding = model.predict(train_batch_x, verbose=0)

    nbrs = NearestNeighbors(n_neighbors=kNN, algorithm=algo, metric=distance, n_jobs=n_jobs).fit(train_batch_embedding)
    distances_embedding, indices = nbrs.kneighbors(test_batch_embedding)

    weights = np.exp(-distances_embedding / temperature)
    weightsP = np.sum(np.expand_dims(weights, -1) * train_batch_y.numpy()[indices, :], 1)
    
    if eval_method=='multi':
        weightsN=np.sum(np.expand_dims(weights, -1) * (1-train_batch_y.numpy())[indices, :], 1)
        weightsPN=np.concatenate([np.expand_dims(weightsP, 0),np.expand_dims(weightsN, 0)],0)
        
        return weightsPN
    else:
        return weightsP
    
################
##################################
################
    
    
def chunk_kNN_ml_prediction(training_dataset:tf.data.Dataset,
                             test_dataset:tf.data.Dataset,
                             model:tf.keras.models,
                             test_configuration:dict):
    '''
    This function computes the average weighted label prediction from the multi-label training set.
    It computes the distance between validation and training points embeddings and 
    selects as predicted label, the one having highest probability. 
    
    Inputs:
    - training_dataset: elements shape ([training data,output dimensions], [training data,label_one_ho_encoding_size])
    - test_dataset: elements shape ([test data,output dimensions], [test data, 1 or label_one_ho_encoding_size])
    - model: neural network trained with supervised contrastive learning
    - distance: selected distance measure (it should match the one used in training the contrastive network)
    - val_batch: batch size to be used in the validation stage
    - temperature: hyperparameter to be used in computing label weights
    
    Note that the validation set should appear in a ground truth fashion (single label). Using a
    one-hot encoding of the validation set labels can be handled by the function.
    
    '''
    from sklearn.neighbors import NearestNeighbors
    val_batch = test_configuration['val_batch']
    n_jobs = test_configuration['n_jobs']
    distance = test_configuration['distance']
    top_perf = test_configuration['top_perf']
    eval_method = test_configuration['eval_method']
    model_weights = model.get_weights()
    
    
    if distance == 'cosine_triangle':
        from scipy.spatial.distance import cosine
        def cosine_triangle(x,y):
            return np.sqrt(np.maximum(2* cosine(x,y),  np.finfo(float).eps))
        
        distance=cosine_triangle
        algo='brute'
    
    elif distance == 'euclidean_squared':
        from scipy.spatial.distance import sqeuclidean
        distance=sqeuclidean
        algo='brute'
        
    elif distance == 'euclidean':
        algo='ball_tree'
        
    elif distance == 'cosine':
        algo='brute'
        
    if test_configuration['get_thresholds']:
        thresholds=test_configuration['thresholds']
    else: 
        threshold = test_configuration['threshold']
   
    test_y_pred=[]
    test_y_true=[]
    test_y_score=[]
    test_y_true_headtails=[]
    
    #load each chunck of validation data and compute embeddings
    for test_batch_x, test_batch_y, test_batch_hts in test_dataset.batch(val_batch):

        test_batch_embedding = model.predict(test_batch_x, verbose=0)
        test_batch_weights=0. ## will contain weighted one-hot econding labels of the val chunck
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_train_batch_kNN)(train_batch_x, 
                                             train_batch_y, 
                                             test_batch_embedding, 
                                             model_weights, 
                                             test_configuration,
                                             algo       
            ) for train_batch_x, train_batch_y in training_dataset.batch(val_batch)
        )

        for weights in results:
            test_batch_weights += weights

        # Compute val chunk predicted labels using argmax and add them to the validation set label array
        norm_w_batch = (test_batch_weights/(test_batch_weights.sum(0,keepdims=True)))[0]
        if test_configuration['get_thresholds']:
            y_pred_batch = np.where(norm_w_batch>thresholds,1,0).astype(int)
            test_y_score_batch = np.where(norm_w_batch>thresholds,norm_w_batch,0).sum(1)/((y_pred_batch).sum(1)+1e-20)   
        else:
            y_pred_batch = (norm_w_batch>=threshold).astype(int)
            test_y_score_batch = np.where(norm_w_batch>=threshold,norm_w_batch,0).sum(1)/((y_pred_batch).sum(1)+1e-20)
        
        test_y_pred.append(y_pred_batch)
        test_y_score.append(test_y_score_batch)
        test_y_true.append(test_batch_y)
        test_y_true_headtails.append(test_batch_hts)

    assert len(test_batch_y.shape) == 2
    
    test_y_true = np.concatenate(test_y_true)
    test_y_pred = np.concatenate(test_y_pred)
    test_y_score = np.concatenate(test_y_score)
    test_y_true_headtails = np.concatenate(test_y_true_headtails)

    top_indices = np.argsort(-test_y_score)[:top_perf]

    return test_y_true, test_y_pred, top_indices, test_y_score, test_y_true_headtails



    


#########################################################################
########### DISTANCE MATRIX BETWEEN BATCHES 
##########################################################################

def distance_matrix_numpy(z1,z2=None, dist="euclidean"):
    """
    Compute distance matrix in the embedding space
    z1 = embeddings of size [ batch_size1 , embedding_dims ]
    z2 = embeddings of size [ batch_size2 , embedding_dims ]
    dist = type of distance definition to be used.
    Possible chices are:
        - 'euclidean_squared'
        - 'cosine'
        - 'euclidean'
        - 'cosine_triangle'
    """
    if z2 is None:
        z2 = z1
        
    # Get the dot product between all embeddings
    # shape (batch_size1, batch_size2)
    dot_product=np.matmul(z1,z2.T)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    z1_square_norm = np.diag(np.matmul(z1, z1.T)) # shape (batch_size1,)
    z2_square_norm = np.diag(np.matmul(z2, z2.T)) # shape (batch_size2,)

    # Compute the pairwise distance matrix as:
    if dist == "euclidean_squared":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size1, batch_size2)
        distances = (
            np.expand_dims(z1_square_norm, 1)
            - 2.0 * dot_product
            + np.expand_dims(z2_square_norm, 0)
        )
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = np.maximum(distances, 0.0) 
    elif dist == "cosine_triangle":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size, batch_size)
        distances = np.sqrt(
            2.0
            + 1e-10
            - 2.0
            * dot_product
            / np.sqrt(
                np.expand_dims(z1_square_norm, 1) * np.expand_dims(z2_square_norm, 0) + 1e-10
            )
        )
    elif dist == "cosine":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size1, batch_size2)
        distances = (
            1.0
            + 1e-10
            - 1.0
            * dot_product
            / np.sqrt(
                np.expand_dims(z1_square_norm, 1) * np.expand_dims(z2_square_norm, 0) + 1e-10
            )
        )
    elif dist == "euclidean":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size1, batch_size2)
        distances = (
            np.expand_dims(z1_square_norm, 1)
            - 2.0 * dot_product
            + np.expand_dims(z2_square_norm, 0)
        )
        distances = np.maximum(distances, 0.0)
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = np.equal(distances, 0.0).astype('float')
        distances = distances + mask * 1e-20
        distances = np.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask) 

    return distances


##########################################################################
########### METRICS FOR PERFORMANCE ASSESSMENT AND VALIDATION
##########################################################################

#f1 micro
def f1_micro_yt(y_true,y_pred):
    tp=0
    fp=0
    fn=0
    for yi in np.unique(y_true):
        tp+=((y_true==yi)&(y_pred==yi)).sum()
        fp+=((y_true!=yi)&(y_pred==yi)).sum()
        fn+=((y_true==yi)&(y_pred!=yi)).sum()

    f1_micro=tp/(tp+0.5*(fn+fp))
    return f1_micro


def f1_macro_yt(y_true,y_pred):
    f1_macro=[]
    for yi in np.unique(y_true):
        tp=((y_true==yi)&(y_pred==yi)).sum()
        fp=((y_true!=yi)&(y_pred==yi)).sum()
        fn=((y_true==yi)&(y_pred!=yi)).sum()
        if (fn+fp)!=0:
            f1_macro.append(tp/(tp+0.5*(fn+fp)))
    f1_macro=np.nanmean(f1_macro)
    return f1_macro



#f1 micro
def precision_micro_yt(y_true,y_pred):
    tp=0
    fp=0
    for yi in np.unique(y_true):
        tp+=((y_true==yi)&(y_pred==yi)).sum()
        fp+=((y_true!=yi)&(y_pred==yi)).sum()

    prec_micro=tp/(tp+fp)
    return prec_micro


def precision_macro_yt(y_true,y_pred):
    prec_macro=[]
    for yi in np.unique(y_true):
        tp=((y_true==yi)&(y_pred==yi)).sum()
        fp=((y_true!=yi)&(y_pred==yi)).sum()
        if (tp+fp)!=0:
            prec_macro.append(tp/(tp+fp))
    prec_macro=np.nanmean(prec_macro)
    return prec_macro


#f1 micro
def recall_micro_yt(y_true,y_pred):
    tp=0
    fn=0
    for yi in np.unique(y_true):
        tp+=((y_true==yi)&(y_pred==yi)).sum()
        fn+=((y_true==yi)&(y_pred!=yi)).sum()

    rec_micro=tp/(tp+fn)
    return rec_micro


def recall_macro_yt(y_true,y_pred):
    rec_macro=[]
    for yi in np.unique(y_true):
        tp=((y_true==yi)&(y_pred==yi)).sum()
        fn=((y_true==yi)&(y_pred!=yi)).sum()
        if (tp+fn)!=0:
            rec_macro.append(tp/(tp+fn))
    rec_macro=np.nanmean(rec_macro)
    return rec_macro


#f1 micro multilabel
def f1_micro_yt_ml(y_true,y_pred):
    tp=0
    fp=0
    fn=0
    for i in range(y_true.shape[0]):
        tp+=((y_true[i,:]==1)&(y_pred[i,:]==1)).sum()
        fp+=((y_true[i,:]==0)&(y_pred[i,:]==1)).sum()
        fn+=((y_true[i,:]==1)&(y_pred[i,:]==0)).sum()
    f1_micro=tp/(tp+0.5*(fn+fp))
    return f1_micro

#f1 macro multilabel
def f1_macro_yt_ml(y_true,y_pred):
    f1_macro=[]
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i])!=0:
            tp=((y_true[:,i])&(y_pred[:,i])).sum()
            fp=((1-y_true[:,i])&(y_pred[:,i])).sum()
            fn=((y_true[:,i])&(1-y_pred[:,i])).sum()
            if (fn+fp)!=0:
                f1_macro.append(tp/(tp+0.5*(fn+fp)))  
    f1_macro=np.nanmean(f1_macro)
    return f1_macro



def compute_bag_f1s(test_y_true,test_y_pred,test_headtails,threshold=0.5):
    bag_df=pd.DataFrame(test_headtails.astype('str'),columns=['syn_id_head','syn_id_tail'])
    bag_df['y_pred']=test_y_pred.tolist()
    bag_df['y_pred']=bag_df['y_pred'].transform(lambda x:np.array(x))
    bag_df['y_true']=test_y_true.tolist()
    bag_df['y_true']=bag_df['y_true'].transform(lambda x:np.array(x))
    
    count_bag_df=bag_df[[
        'syn_id_head','syn_id_tail']].groupby(['syn_id_head','syn_id_tail']).value_counts().reset_index(name='count')
    bag_true_rels=bag_df.groupby(['syn_id_head','syn_id_tail'])['y_true'].apply('sum').reset_index()
    bag_pred_rels=bag_df.groupby(['syn_id_head','syn_id_tail'])['y_pred'].apply('sum').reset_index()


    from functools import reduce
    bag_df = reduce(lambda  left,right: pd.merge(left,right,on=['syn_id_head','syn_id_tail'],
                                                how='outer'), [bag_true_rels, bag_pred_rels, count_bag_df])
    
    print(f'number of bags (shape) {bag_df.shape}')
    bag_df['score_true']=bag_df['y_true']/bag_df['count']
    bag_df['score_pred']=bag_df['y_pred']/bag_df['count']
    bag_df=bag_df[['syn_id_head','syn_id_tail','score_true','score_pred']]
    
    ybag_true=np.array([el.tolist() for el in bag_df['score_true'].tolist()])
    ybag_true=np.where(ybag_true>threshold,1.,0.).astype(int)
    ybag_pred=np.array([el.tolist() for el in bag_df['score_pred'].tolist()])
    ybag_pred=np.where(ybag_pred>threshold,1.,0.).astype(int)
    
    results_bag={'microf1' : f1_micro_yt_ml(ybag_true,ybag_pred),
                 'macrof1'  : f1_macro_yt_ml(ybag_true,ybag_pred)}
    
    return bag_df, results_bag




###########
### TRAIN SINGLE GRID POINT  
###########
def train_one_grid_point(gs_point:list, output_path:str, cols:list, 
                         df_training,  
                         df_validation=False):
    
    buffer_size=len(df_training)
    training_dataset=df_to_tf_dataset(df_training)
    training_dataset=training_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    

    if not hasattr(df_validation,'size'):
        validation=False
    else:
        validation=True
        validation_dataset=df_to_tf_dataset(df_validation)
        validation_dataset=validation_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)


    ###############
    #### TRAINING 
    ###############
    configuration = {'input_shape': 2304, #ATTENZIONE E' ARCODATO
                     'architecture':'MLP',
                     'base_temperature':1.0,
                     'optimizer':tf.keras.optimizers.legacy.Adam(),
                     'epochs':100,
                     'val_batch':100000,
                     'n_jobs':-1,
                     'top_perf':1000,
                     'eval_method':'isin', #'unroll', 'isin'
                    }


    for key,entry in zip(cols,gs_point):    
        if key in ['learning_rate', 'temperature','pert']:
            entry=float(entry)
        elif key in ['output_dimensions','depth', 'batch_size','kNN']:
            try:
                entry=int(entry)
            except ValueError:
                entry=bool(entry)
        elif key=='activation' and entry=='tf.math.sin':
            entry=eval(entry)

        configuration[key]=entry
        
    ## CREATE MODEL
    tf.keras.backend.clear_session()
    model = CreateModel(training_configuration = configuration)

    ## TRAIN MODEL
    if validation:
        callback= tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0)


        history = model.fit(training_dataset.batch(configuration['batch_size']),
                                                    epochs = configuration['epochs'], 
                                                    validation_data=validation_dataset.batch(configuration['batch_size']), 
                                                    callbacks=[callback],
                                                    verbose=False)
        
        ## EVALUATE RESULTS
        test_y_true, test_y_pred, test_y_score, results_perf = evaluation_and_performance(test_configuration=configuration,
                                                                                          training_dataset=training_dataset,
                                                                                          test_dataset=validation_dataset,
                                                                                          model=model
                                                                                          )

    else:
        callback= tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-3,
        patience=5,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0)


        history = model.fit(training_dataset.batch(configuration['batch_size']),
                            epochs = configuration['epochs'], 
                            callbacks=[callback],
                            verbose=False)  
        
        ## EVALUATE RESULTS
        test_y_true, test_y_pred, test_y_score, results_perf = evaluation_and_performance(test_configuration=configuration,
                                                                                          training_dataset=training_dataset,
                                                                                          test_dataset=training_dataset,
                                                                                          model=model
                                                                                          )


    



    line_to_save=[]
    line_to_save += gs_point
    line_to_save += [str(round(el,3)) for el in  list(results_perf['total'].values())]
    line_to_save += [str(round(el,3)) for el in  list(results_perf['@'+str(configuration['top_perf'])].values())]
    return line_to_save



###############
##### CLUSTERING PLOT ROUTINE
###############


import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from sklearn.manifold import TSNE

def scatter(x, labels, subtitle=None):
    # We choose a color palette with seaborn.
    unique_labels=list(np.unique(labels))
    
    palette = np.array(sns.color_palette("hls", len(unique_labels)))
    
    # We create a scatter plot.
    f = plt.figure(figsize=(3, 3))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    
    
    for lab in unique_labels:
        # Position of each label.
        xtext, ytext = np.median(x[labels == lab, :], axis=0)
        txt = ax.text(xtext, ytext, str(lab), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
        
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.show()
    
    
  


