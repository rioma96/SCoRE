import pandas as pd
import glob
from pathlib import Path
import pickle
import json
import gc
import re
import multiprocessing as mp
from itertools import chain
import itertools
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import argparse
import time 
import concurrent.futures
from multiprocessing import Lock
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import ParameterGrid
from codecarbon import track_emissions
import scipy
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors



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
    output_dim=test_configuration['output_dim']
    attention_features=test_configuration['attention_features']
    dense_neurons=test_configuration['dense_neurons']
    activation=test_configuration['activation']
    LLM_encoder, _ , _ = init_BERT()

    model=SentenceAttention(LLM_encoder=LLM_encoder, output_dim=output_dim, attention_feautures=attention_features, dense_neurons=dense_neurons, activation=activation)
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
        #test_y_true_headtails.append(test_batch_hts)

    assert len(test_batch_y.shape) == 2
    
    test_y_true = np.concatenate(test_y_true)
    test_y_pred = np.concatenate(test_y_pred)
    test_y_score = np.concatenate(test_y_score)
    #test_y_true_headtails = np.concatenate(test_y_true_headtails)

    top_indices = np.argsort(-test_y_score)[:top_perf]

    return test_y_true, test_y_pred, top_indices, test_y_score#, test_y_true_headtails



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
        
    test_y_true, test_y_pred, top_indices, test_y_score = chunk_kNN_ml_prediction(training_dataset=training_dataset,
                                                                                test_dataset=test_dataset,
                                                                                model=model,
                                                                                test_configuration=test_configuration)
    return test_y_true, test_y_pred, top_indices, test_y_score

def evaluation_and_performance(test_configuration:dict,
                               training_dataset:tf.data.Dataset,
                               test_dataset:tf.data.Dataset,
                               model:tf.keras.models):
    
    #from sklearn.metrics import confusion_matrix 
    eval_method = test_configuration['eval_method']

    test_y_true, test_y_pred, top_indices, test_y_score = similarity_to_class(
                                                                            test_configuration=test_configuration,
                                                                            training_dataset=training_dataset,
                                                                            test_dataset=test_dataset,
                                                                            model=model
                                                                            )
        
    print(test_y_true.shape,test_y_pred.shape)
        
        
        
    

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
        return  test_y_true, test_y_pred, test_y_score, results_perf



@tf.function
def remove_diag(tensor):
    # set tensor diagonal elements to zero
    tensor = tensor - tf.linalg.tensor_diag(tf.linalg.diag_part(tensor))
    return tensor



@tf.function
def distance_matrix(z, dist="euclidean"):
    """
    Compute distance matrix in the embedding space
    z = embeddings of size [ batch , embedding_dims ]
    dist = type of distance definition to be used.
    Possible chices are:
        - 'euclidean_squared'
        - 'cosine'
        - 'euclidean'
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(z, tf.transpose(z))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.linalg.diag_part(dot_product)
        
    # Compute the pairwise distance matrix as:
    if dist == "euclidean_squared":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 0)
        )
        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)
    elif dist == "cosine_triangle":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size, batch_size)
        distances = tf.math.sqrt(
            2.0
            + 1e-10
            - 2.0
            * dot_product
            / tf.math.sqrt(
                tf.expand_dims(square_norm, 1) * tf.expand_dims(square_norm, 0) + 1e-10
            )
        )
    elif dist == "cosine":
        # sqrt( 2 + eps - 2 (a . b)/|a||b| )
        # shape (batch_size, batch_size)
        distances = (
            1.0
            + 1e-10
            - 1.0
            * dot_product
            / tf.math.sqrt(
                tf.expand_dims(square_norm, 1) * tf.expand_dims(square_norm, 0) + 1e-10
            )
        )
    elif dist == "euclidean":
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 0)
        )
        distances = tf.maximum(distances, 0.0)
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.cast(tf.equal(distances, 0.0), dtype=tf.float32)
        distances = distances + mask * 1e-20
        distances = tf.sqrt(distances)
        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


class Supervised_ML_Contrastive(tf.keras.losses.Loss):
    """
    Supervised multi-label contrastive loss to use in distant-to-fully supervised training
    Args:
        temperature : to increase or decrease magnitude of attraction/repulsion
        base_temperature : to rescale the loss value
        distance : type of metric to be used in evaluating distances

        y_true_ml: true multi-labels of the batch of shape [bsz,n_classes]
        z: hidden vector of shape [bsz, n_features].
    """

    def __init__(self, 
                 temperature:float = 0.5, 
                 base_temperature:float = 1.0, 
                 distance: str = "euclidean"):

        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.distance = distance

    @tf.function
    def call(self, y_true_ml, z):

        batch_size = tf.cast(tf.shape(y_true_ml)[0],tf.float32)
        y_true_ml = tf.expand_dims(y_true_ml, 1)

        ## create weighted beta mask: consider y&yT and normalize each row
        ## [bsz,1,10] [1,bsz,10] --> [bsz, bsz, 10] -> [bsz,bsz]
        beta_mask = tf.reduce_sum(
            tf.cast(y_true_ml * tf.transpose(y_true_ml, [1, 0, 2]), tf.float32), -1
        )
        beta_mask = remove_diag(beta_mask)
        beta_mask /= tf.maximum(tf.reduce_sum(beta_mask, -1, keepdims=True), 1)
        
        ## create boolean beta mask
        bool_mask = tf.cast(beta_mask > 1e-7, tf.float32)

        ### Distance matrix and logits
        distances = tf.cast(distance_matrix(z, self.distance), tf.float32)
        distances = remove_diag(distances)

        logits = tf.divide(distances, self.temperature)

        # compute mean of log-likelihood over positive
        exp_logits = tf.exp(-logits)
        exp_logits = remove_diag(exp_logits)

        loss = beta_mask * (
            logits
            + tf.math.log(1e-10 + tf.reduce_sum(exp_logits, axis=1, keepdims=True))
        )

        loss = (
            (self.temperature / self.base_temperature)
            * tf.reduce_sum(loss)
            / batch_size
        )
        
        return loss

def init_BERT(only_real_token=False, extract_CLS_only=False):
    
    preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    
    # Step 1: tokenize batches of text inputs.
    text_inputs = [tf.keras.layers.Input(shape=(), dtype=tf.string)] # 2 input layers for 2 text inputs
    tokenize = hub.KerasLayer(preprocessor.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_inputs]
    
    # Step 3: pack input sequences for the Transformer encoder.
    seq_length = 512  # Your choice here.
    bert_pack_inputs = hub.KerasLayer(
        preprocessor.bert_pack_inputs,
        arguments=dict(seq_length=seq_length))  # Optional argument.
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
        trainable=False)
    outputs = encoder(encoder_inputs)
    # Get the output of the [CLS] token, which represents the sentence embedding.
    pooled_output = outputs["pooled_output"]      # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    
    # Obtain the token mask indicating which tokens belong to the actual input sentences.
    input_mask = encoder_inputs["input_mask"]
    
    # Cast input mask to float32 to match the data type of sequence output.
    input_mask = tf.cast(input_mask, dtype=tf.float32)
    
    # Apply the input mask to filter out padding tokens.
    filtered_sequence_output = sequence_output * tf.expand_dims(input_mask, axis=-1)
    
    if only_real_token:
        # Keep only tokens from the original input sentence (excluding [CLS] and [SEP])
        start_token_index = 1  # Start after [CLS]
        
        # Compute the end token index based on the input_mask
        end_token_index = tf.reduce_sum(input_mask, axis=-1) - 1
        
        # Squeeze the end_token_index tensor to remove extra dimensions
        end_token_index = tf.squeeze(end_token_index, axis=-1)
        
        # Convert end_token_index to a scalar tensor
        end_token_index = tf.cast(end_token_index, dtype=tf.int32)
        
        # Create a range of indices from start_token_index to end_token_index
        indices = tf.range(start_token_index, end_token_index)
        filtered_sequence_output = filtered_sequence_output[:, start_token_index:end_token_index]
    
    if extract_CLS_only:
        embedding_model = tf.keras.Model(text_inputs, pooled_output)  # Extract [CLS] token embedding if you put pooled_output.
    else:
        embedding_model = tf.keras.Model(text_inputs, filtered_sequence_output) # Extract tokens masked embedding if you put filtered_sequence_output.
    
    #sentences = tf.constant([sentence])
    return embedding_model, preprocessor,tokenize


def load_wiki20m(path, file_name,columns,NA=False):
    
    raw_data = []
    clean_text=[]
    processed_data = []
    # Opening JSON file
    with open(path+file_name,encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                raw_data.append(eval(line))
    
    for sent_num in range(len(raw_data)):

        doc = raw_data[sent_num]

        relation=[doc['relation']] #In that way i don't consider relation name when the relation is labeled NA.
                                #If you want consider them you have to change relation with 'r' and then create a map 'r' to NA
        #If NA is set to false we don't consider triples with NA relation
        if not NA and "NA" in relation:
            continue
        
        text= ' '.join(doc['token'])
        begin_sentence=0
        end_sentence=len(text)
        if text in clean_text:
            clean_text_position = clean_text.index(text)
        else:
            clean_text.append(text)
            clean_text_position = clean_text.index(text)
    
        head_tok_start = doc['h']['pos'][0]
        head_tok_end = doc['h']['pos'][1]
        head_name = doc['h']['name']
        head_id = doc['h']['id']
        tail_tok_start = doc['t']['pos'][0]
        tail_tok_end = doc['t']['pos'][1]
        tail_name = doc['t']['name']
        tail_id = doc['t']['id']

    
    #   ------------------------------------------HEAD POSITION--------------------------------------------
        head_text_start = 0
        #Calcolo la posizione di partenza nel testo della head
        for i in range(head_tok_start):
            ent_end = head_text_start
            if (head_text_start >= 0) and (head_text_start <= len(text)):
                ent_end = head_text_start + len(doc['token'][i])
                head_text_start = ent_end + 1
    
        #Salvo la posiz di partenza della head
        head_start_bound =head_text_start
        head_end_bound = 0
        #Calcolo la posizione testuale di termine della head
        for i in range(head_tok_start,head_tok_end):
            ent_end = head_text_start + len(doc['token'][i])
            head_text_start = ent_end + 1
            head_end_bound = ent_end
    
    #  -------------------------------------------TAIL POSITION----------------------------------------------
    
        tail_text_start = 0
        #Calcolo la posizione di partenza nel testo della head
        for i in range(tail_tok_start):
            ent_end = tail_text_start
            if (tail_text_start >= 0) and (tail_text_start <= len(text)):
                ent_end = tail_text_start + len(doc['token'][i])
                tail_text_start = ent_end + 1
    
        #Salvo la posiz di partenza della head
        tail_start_bound =tail_text_start
        tail_end_bound = 0
        #Calcolo la posizione testuale di termine della head
        for i in range(tail_tok_start,tail_tok_end):
            ent_end = tail_text_start + len(doc['token'][i])
            tail_text_start = ent_end + 1
            tail_end_bound = ent_end

        processed_data.append([head_id,tail_id,relation,clean_text_position,begin_sentence,end_sentence,head_start_bound,head_end_bound,tail_start_bound,tail_end_bound])

    processed_data=pd.DataFrame(processed_data,columns= columns)
    return processed_data,clean_text

def load_nyt10d(path,file_name,columns,NA=False):
    raw_data = []
    clean_text=[]
    processed_data = []
    # Opening JSON file
    with open(path+file_name,encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                raw_data.append(eval(line))
    
    for sent_num in range(len(raw_data)):

        doc = raw_data[sent_num]
        text= doc['text']
        relation=[doc['relation']]

        if not NA and "NA" in relation:
            continue
        begin_sentence=0
        end_sentence=len(text)
        if text in clean_text:
            clean_text_position = clean_text.index(text)
        else:
            clean_text.append(text)
            clean_text_position = clean_text.index(text)
    
        head_start_bound = doc['h']['pos'][0]
        head_end_bound = doc['h']['pos'][1]
        head_name = doc['h']['name']
        head_id = doc['h']['id']
        tail_start_bound = doc['t']['pos'][0]
        tail_end_bound = doc['t']['pos'][1]
        tail_name = doc['t']['name']
        tail_id = doc['t']['id']


        processed_data.append([head_id,tail_id,relation,clean_text_position,begin_sentence,end_sentence,head_start_bound,head_end_bound,tail_start_bound,tail_end_bound])

    processed_data=pd.DataFrame(processed_data,columns= columns)
    return processed_data,clean_text

def load_nyt10m(path,file_name,columns,NA=False):
    return load_nyt10d(path,file_name,columns,NA=False)

def load_gds(path,file_name,columns,NA=False):
    raw_data = []
    clean_text=[]
    processed_data = []
    # Opening JSON file
    with open(path+file_name,encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                raw_data.append(json.loads(line))
                
    #Pulisco i dati e mantengo solo le frasi con 1 match per head e tail, se di più c'è ambiguità perchè non sono presenti riferimenti allìentità specifica in GDS
    data_cleaned = []
    for sent in raw_data:
        frase = ' '.join(sent['sent'])
        sub = [(m.start(), m.end()) for m in re.finditer(re.escape(sent['sub']), frase)]
        ob = [(m.start(), m.end()) for m in re.finditer(re.escape(sent['obj']), frase)]
        if len(sub) ==1 and len(ob) == 1:
            data_cleaned.append(sent)
    
    for sent_num in range(len(raw_data)):

        doc = raw_data[sent_num]
        relation = [doc['rel']]

        if not NA and "NA" in relation:
            continue
        text= ' '.join(doc['sent'])
        begin_sentence=0
        end_sentence=len(text)
        if text in clean_text:
            clean_text_position = clean_text.index(text)
        else:
            clean_text.append(text)
            clean_text_position = clean_text.index(text)

        head_id = doc['obj']
        head_start_bound = [m.start() for m in re.finditer(re.escape(doc['obj']), text)]
        head_end_bound = [m.end() for m in re.finditer(re.escape(doc['obj']), text)]
        tail_id = doc['sub']
        tail_start_bound = [m.start() for m in re.finditer(re.escape(doc['sub']), text)]
        tail_end_bound = [m.end() for m in re.finditer(re.escape(doc['sub']), text)]

        
        processed_data.append([head_id,tail_id,relation,clean_text_position,begin_sentence,end_sentence,head_start_bound[0],head_end_bound[0],tail_start_bound[0],tail_end_bound[0]])

    processed_data=pd.DataFrame(processed_data,columns= columns)
    return processed_data,clean_text

#Global variable to store all clean text of T-Rex (this is to resolve indices problem in parallel workflow)

def elaborate_Trex(path,columns,rebel=False,NA=False):
    raw_data = []
    processed_data = []
    erroneous_counter=0
    positive_counter=0

    if rebel:
        with open(path,encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()
                if len(line) > 0:
                    raw_data.append(json.loads(line))
    else:
        # Opening JSON file
        f = open(path,encoding='utf-8')
        # returns JSON object as
        # a dictionary
        raw_data = json.load(f)
        #Chiudo il file, dovrebbe liberare la memoria
        f.close()

    #if rebel:
        #raw_data=raw_data[0]['0']
    
    for sent_num in range(len(raw_data)):
        
        if rebel:
            sent_num=str(sent_num)
            
        doc = raw_data[sent_num]
        text= doc["text"]
        begin_sentence=0
        end_sentence=len(text)
        # if text in clean_text:
        #     clean_text_position = clean_text.index(text)
        # else:
        #     clean_text.append(text)
        #     clean_text_position = clean_text.index(text)
            
        #Check if document has annotated triples
        if len(doc["triples"]) > 0:
            for triple in doc["triples"]:
                #This try-catch is for triples where at least one entity has text boundaries set to None
                try:
                    #extract start and end sentence bound from json
                    begin_sentence=doc["sentences_boundaries"][triple['sentence_id']][0]
                    end_sentence=doc["sentences_boundaries"][triple['sentence_id']][1]
                    #Keep only Q123 removing previous entity path
                    head_id = triple["object"]["uri"].split("/")[-1]
                    tail_id = triple["subject"]["uri"].split("/")[-1]
                    relation = [triple["predicate"]["uri"].split("/")[-1]]
                    head_start_bound = triple["object"]['boundaries'][0]
                    head_end_bound = triple["object"]['boundaries'][1]
                    tail_start_bound = triple["subject"]['boundaries'][0]
                    tail_end_bound = triple["subject"]['boundaries'][1]

                    if not NA and "NA" in relation:
                        continue
    
                    processed_data.append([head_id,tail_id,relation,text,begin_sentence,end_sentence,head_start_bound,head_end_bound,tail_start_bound,tail_end_bound])
                    positive_counter+=1
                except:
                    erroneous_counter+=1
    processed_data=pd.DataFrame(processed_data,columns= columns)
    # print(erroneous_counter/(erroneous_counter+positive_counter))
    return processed_data

def load_Trex(path,file_name,columns,NA=False,rebel=False):
    path=path+file_name
    #T-Rex has multiple files, we want to parallelize the dataset preprocessing (400x speedup)
    file_list = sorted(glob.glob(path + "*"))
    pool = mp.Pool()
    results = pool.starmap(elaborate_Trex, [(i,columns,rebel,NA) for i in file_list])
    pool.close()
    pool.join()
    df_result=pd.concat(results)

    
    #------------------------Clean Text list creation and replace text with index of clean_text in DF-------------------------
    clean_text=pd.DataFrame(df_result['doc_pos'].unique(),columns=["doc_pos"])
    clean_text.reset_index(drop=True, inplace=True)
    clean_text.reset_index( inplace=True)
    clean_text.rename(columns = {'index':'Position'}, inplace = True)
    text_pos_dict = pd.Series(clean_text.Position.values,index=clean_text.doc_pos.values).to_dict()
    df_result['doc_pos']=df_result['doc_pos'].map(text_pos_dict)
    clean_text=clean_text['doc_pos'].to_list()
    #--------------------------------------------------------------------------------------------------------------------------

    #Parallel data processing causes duplicate row, this code remove duplicate
    df_result=df_result.loc[~df_result.astype(str).duplicated()]

    return df_result, clean_text

def load_DisRex(path, file_name,columns,NA=False):
    return load_wiki20m(path, file_name,columns,NA)

def load_Rebel(path,file_name,columns,NA=False):
    # return load_Trex(path,file_name,columns,True,NA)
    processed_data = []
    erroneous_counter=0
    positive_counter=0
    raw_data=[]
    clean_text=[]
    file = sorted(glob.glob(path+file_name+"*"))
    # Opening JSON file
    with open(file[0],encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                raw_data.append(json.loads(line))
                
    for sent_num in range(len(raw_data)):
        
        # sent_num=str(sent_num)
            
        doc = raw_data[sent_num]
        text= doc["text"]
        begin_sentence=0
        end_sentence=len(text)
        if text in clean_text:
            clean_text_position = clean_text.index(text)
        else:
            clean_text.append(text)
            clean_text_position = clean_text.index(text)
            
        #Check if document has annotated triples
        if len(doc["triples"]) > 0:
            for triple in doc["triples"]:
                #This try-catch is for triples where at least one entity has text boundaries set to None
                try:
                    #extract start and end sentence bound from json
                    begin_sentence=0
                    end_sentence=len(text)
                    #Keep only Q123 removing previous entity path
                    head_id = triple["object"]["uri"].split("/")[-1]
                    tail_id = triple["subject"]["uri"].split("/")[-1]
                    relation = [triple["predicate"]["uri"].split("/")[-1]]
                    head_start_bound = triple["object"]['boundaries'][0]
                    head_end_bound = triple["object"]['boundaries'][1]
                    tail_start_bound = triple["subject"]['boundaries'][0]
                    tail_end_bound = triple["subject"]['boundaries'][1]
                    # Rebel shades -> distant_supervision\n",
                    distant_y = triple.get("distant_predicate_uris",None)

                    if not NA and "NA" in relation:
                        continue

                    processed_data.append([head_id,tail_id,relation,distant_y,clean_text_position,begin_sentence,end_sentence,head_start_bound,head_end_bound,tail_start_bound,tail_end_bound])
                    positive_counter+=1
                except:
                    erroneous_counter+=1
    if not 'distant_y' in columns:
        columns.insert(3,'distant_y')
    processed_data=pd.DataFrame(processed_data,columns= columns)
    # print(erroneous_counter/(erroneous_counter+positive_counter))
    return processed_data, clean_text

#Callable to generate data dinamically
def load_data(dataset_name,path,NA=False,distant_supervision_available=False):
    
    columns=['syn_id_head', 'syn_id_tail','link_name','doc_pos', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end']
    position_columns = ['doc_pos', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end']
    
    if dataset_name=="NYT10D":
        load_dataset_function=load_nyt10d
    elif dataset_name=="Wiki20m":
        load_dataset_function=load_wiki20m
    elif dataset_name=="NYT10m":
        load_dataset_function=load_nyt10m
    elif dataset_name=="GDS":
        load_dataset_function=load_gds
    elif dataset_name=="TRex":
        load_dataset_function=load_Trex
    elif dataset_name=="DisRex":
        load_dataset_function=load_DisRex
    elif dataset_name=="Rebel":
        load_dataset_function=load_Rebel
    elif dataset_name=="Rebel_Shades":
        load_dataset_function=load_Rebel
    elif dataset_name=="Test": #For testing purpose
        load_dataset_function=load_nyt10m
        
    #Load and preprocess dataset
    print("Loading and starting preprocessing....")
    result, ct = load_dataset_function(path,"",columns,NA)
    print("File loaded and preprocessed")
    
    result["doc_pos"]=result["doc_pos"].apply(lambda x: ct[x])

    result.rename(columns={"doc_pos":"sentence"}, inplace=True)

    return result


def add_entities_markers(sentence,h_s, h_e, t_s, t_e):
    sentence_with_markers=sentence[:h_s]+"<e1>"+sentence[h_s:h_e]+"</e1>"+sentence[h_e:t_s]+"<e2>"+sentence[t_s:t_e]+"</e2>"+sentence[t_e:]
    new_h_s=h_s+0 #adjust head start adding 4 char of <e1>
    new_h_e=h_e+9 #adjust head end addung 4 char of <e1>
    new_t_s=t_s+9 #adjusting tail start adding 9 char of <e1> </e1> <e2>
    new_t_e=t_e+18 #adjusting tail end adding 18 char of <e1> </e1> <e2>
    return sentence_with_markers,new_h_s,new_h_e,new_t_s,new_t_e

def embedd_row(row):
    row['sentence'], row['head_start'], row['head_end'], row['tail_start'], row['tail_end'] = add_entities_markers(row['sentence'],row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
    return row

def set_minimal_lists(el):
    return el if isinstance(el, list) else [el]

def from_file_list_to_tfdataset(dataset_name, 
                                path,
                                NA=False,
                                distant_supervision_available=False,
                                relation_dict=None,
                                controllino=False,
                                return_headtail=False,
                                return_df=False):
    

    # Se facciamo embedding realtime
    file_list=glob.glob(path+"*.pkl")
    df = pd.concat([pd.read_pickle(file) for file in file_list], ignore_index=True)
    df=df.apply(lambda row: embedd_row(row), axis=1)
    df = df.groupby(['sentence', 'sent_start', 'sent_end','head_start', 'head_end', 'tail_start', 'tail_end','syn_id_head', 'syn_id_tail']).agg(list)
    new_link_name_column = [list(set(itertools.chain(*list_of_iterables))) for list_of_iterables in df.loc[:,"link_name"]]
    df = df.map(lambda x: x[0])
    df["link_name"] = new_link_name_column
    df.reset_index(inplace=True,drop=False)
    df = df[['sentence', 'link_name', 'syn_id_head', 'syn_id_tail']]
    
    
    #controllino
    
    print("Dataset loaded")
    
    if relation_dict is None:
        codes, relation_dict = pd.factorize(df['link_name'].explode())
        relation_dict = dict([(el, i) for i, el in enumerate(relation_dict)])
        df['link_name']=df['link_name'].map(lambda x: [relation_dict[i] for i in x])
    else:
        # Check if all values in df['link_name'] are in relation_dict
        if all(value in relation_dict for value in df['link_name'].explode().unique()):
            # If all values exist in relation_dict, proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])
        else:
            # Remove rows where at least one value is not in relation_dict
            df = df[df['link_name'].apply(lambda x: all(i in relation_dict for i in x))]
            # Proceed with the mapping
            df['link_name'] = df['link_name'].map(lambda x: [relation_dict[i] for i in x])


    list_of_classes=np.array(list(relation_dict.values()))
    print("Relations mapped")
    one_hot = MultiLabelBinarizer(classes=list_of_classes)
    df['link_name'] = df['link_name'].apply(set_minimal_lists)
    df['link_name'] = one_hot.fit_transform(df['link_name'].to_list()).tolist()
    
    
    print("one_hot.fit_transform done")
    print("Starting tf.data.Dataset.from_tensor_slices")
   
    if return_headtail:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['sentence'].tolist()),
            #np.array(df['K'].tolist()),
            np.array(df['link_name'].tolist()),
            np.array(df[['syn_id_head', 'syn_id_tail']].to_numpy())
        ))
        
    else:
        training_set = tf.data.Dataset.from_tensor_slices((
            np.array(df['sentence'].tolist()),
            #np.array(df['K'].tolist()),
            np.array(df['link_name'].tolist())
        ))

    print("tfdataset created")
    
    to_return = [relation_dict, training_set]  
    if return_df:
        to_return = to_return+[df] 
        
    return to_return


class SentenceAttention(tf.keras.Model):
    """
    token-level attention for passage-level relation extraction.
    """

    def __init__(self, LLM_encoder, attention_feautures, dense_neurons=3, output_dim=20, activation='relu'):
        """
        Args:
            LLM_encoder: encoder for whole passage (bag of sentences)
            attention_feautures: number of attention feautures
        """
        super(SentenceAttention, self).__init__()
        self.LLM_encoder = LLM_encoder
        self.embed_dim = self.LLM_encoder.layers[-1].output_shape[-1]
        self.attention_feautures = attention_feautures
        self.fc_output = tf.keras.layers.Dense(dense_neurons, activation=activation)
        self.relation_embeddings = tf.Variable(tf.keras.initializers.GlorotNormal()((self.attention_feautures, self.embed_dim)),
                                               trainable=True)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.flatten = tf.keras.layers.Flatten()
        self.output_dim = tf.keras.layers.Dense(output_dim, activation=activation)
        self.lambda_layer=tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))

    @tf.function
    def call(self, batch_sentence, train=True):
        """
        Args:
            token: [batch,L], index of tokens   [batch,L]
        Return:
            logits, (B, N)
            Prova Contrastive: R=20, Dense=3
        """
        #batch_size = tf.shape(rep)[0]
        batch_size = tf.shape(batch_sentence)[0]
        #Comment if you want to head tail cls attention and not sentence
        rep = self.LLM_encoder(batch_sentence)  # [batch,L,emb_dim]
        
        
        att_mat = tf.tile(tf.expand_dims(self.relation_embeddings, axis=0), [batch_size, 1, 1])  #  [batch,R,emb_dim]
        att_scores = tf.transpose(tf.matmul(rep, tf.transpose(att_mat, perm=[0, 2, 1])), perm=[0, 2, 1])  #    [batch,R,L]
        att_scores = self.softmax(att_scores) #      [batch,R,L]
        rel_logits = tf.matmul(att_scores, rep) #    [batch,R,L]x[batch,L,emb_dim] -> [batch,R,emb_dim]
        rel_scores = self.fc_output(rel_logits) # [batch,R,emb_dim] -> [batch,R,dense_neurons] -> [batch,R,dense_neurons]
        rel_scores = self.flatten(rel_scores) # [batch,R,dense_neurons] -> [batch,R*dense_neurons]
        rel_scores = self.output_dim(rel_scores)  # [batch,R*dense_neurons] -> [batch,output_dim]
        rel_scores = self.lambda_layer(rel_scores)

        

        return rel_scores


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
    
    
    
def predict_results(dataset:tf.data.Dataset,
                         model:tf.keras.models,
                         threshold:float):
    

    test_y_pred=[]
    test_y_true=[]
  
        
    
    #load each chunck of validation data and compute embeddings
    for test_batch_x, test_batch_y in dataset.batch(100):

        test_batch_embedding = model.predict(test_batch_x, verbose=0)
        y_pred_batch=(test_batch_embedding>threshold).astype('float')[:,:,0]
        test_y_pred.append(y_pred_batch)
        test_y_true.append(test_batch_y)


    test_y_true = np.concatenate(test_y_true)
    test_y_pred = np.concatenate(test_y_pred).astype(int)
    shape = test_y_pred.shape[1]
    #test_y_true = tf.one_hot(test_y_true,shape).numpy().astype(int)

    #pd.DataFrame([test_y_true, test_y_pred]).to_pickle("./ttresults.pkl")

    mf1=f1_micro_yt_ml(test_y_true,test_y_pred)
    Mf1=f1_macro_yt_ml(test_y_true,test_y_pred)

    return mf1, Mf1, test_y_true,test_y_pred


@track_emissions
def main():
    
    buffer_size=100000
    dataset_name="Wiki20m"
    print(dataset_name)
    link_dict, training_dataset, _=from_file_list_to_tfdataset(dataset_name,"/unimore_home/lmariotti/CBKGE/DatasetsProcessed/Wiki20m/Train/",return_df=True)
    training_dataset=training_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    link_dict, val_dataset, _=from_file_list_to_tfdataset(dataset_name,"/unimore_home/lmariotti/CBKGE/DatasetsProcessed/Wiki20m/Val/",return_df=True,relation_dict=link_dict)
    link_dict, test_dataset, _=from_file_list_to_tfdataset(dataset_name,"/unimore_home/lmariotti/CBKGE/DatasetsProcessed/Wiki20m/Test/",return_df=True,relation_dict=link_dict)


    # Initialize BERT
    LLM_encoder, preprocessor, tokenize = init_BERT()

    # Initialize the model
    model = SentenceAttention(LLM_encoder, output_dim=10, attention_feautures=20, dense_neurons=3, activation='relu')

    configuration = {
        'kNN': 2,
        'distance': 'euclidean',
        'temperature': 0.1,
        'n_jobs': -1,
        'top_perf': 1000 ,
        'eval_method': 'multi',
        'output_dim': 10 ,
        'attention_features':25,
        'dense_neurons': 3,
        'activation': 'swish' ,
        'val_batch':1000,
        'get_thresholds':False,
        'threshold':0.5
    }

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = Supervised_ML_Contrastive(
            temperature=0.1,
            base_temperature=1,
            distance='euclidean',
        )
    batch_size=128
    epochs=1
    model.compile(loss=loss, optimizer=optimizer)

    # Train the model
    #history = model.fit(training_dataset.batch(batch_size),
    #                    epochs=epochs,
    #                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)],
    #                    validation_data=val_dataset.batch(batch_size))

    test_y_true, test_y_pred, test_y_score, results_perf = evaluation_and_performance(test_configuration=configuration,
                                                                                training_dataset=training_dataset,
                                                                                test_dataset=test_dataset,
                                                                                model=model
                                                                                )

    # Evaluate the model
    #micro, macro, _, _ = predict_results(test_dataset, model, 0.5)
    #print("Micro F1: ",micro, "Macro F1: ",macro)
    print('Performance on the whole validation set:\n')
    for key in results_perf['total']:
        if key!='list_f1':
            perf=results_perf['total'][key]
            print(f'{key}: {perf}')
        
    top_string='@'+str(configuration['top_perf'])
    print(f'\n\Performance {top_string} on validation set:\n')
    for key in results_perf[f'@'+str(configuration['top_perf'])]:
        if key!='list_f1':
            perf=results_perf[f'@'+str(configuration['top_perf'])][key]
            print(f'{key}: {perf}')


    
if __name__ == "__main__":
    main()
