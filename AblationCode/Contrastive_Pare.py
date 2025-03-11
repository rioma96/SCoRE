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

sys.path.append('../')
from CBKGE.NN_creation_and_dependencies import *
from CBKGE.utilities_validation import *




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

def add_masked_entities_markers(sentence,h_s, h_e, t_s, t_e):
    sentence_with_markers=sentence[:h_s]+"[MASK]"+sentence[h_e:t_s]+"[MASK]"+sentence[t_e:]
    return sentence_with_markers

def embedd_row(row):
    row['sentence'], row['head_start'], row['head_end'], row['tail_start'], row['tail_end'] = add_entities_markers(row['sentence'],row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
    return row

def mask_row(row):
    row['sentence'] = add_masked_entities_markers(row['sentence'],row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
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
    #df=df.apply(lambda row: embedd_row(row), axis=1)
    df=df.apply(lambda row: mask_row(row), axis=1) #Masking entities
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_paths',
            help='list of train/val file path str([train path ,val path, test path])')
    parser.add_argument('--output_path', default='./PredictionResults.csv',
            help='output file path and name')
    parser.add_argument('--dataset_name', default='Wiki20m',
            help='dataset name')
    args = parser.parse_args()

    ## INPUTS
    dataset_paths=args.dataset_paths
    dataset_paths = dataset_paths.replace("\r","")
    dataset_paths = dataset_paths.replace("\n","")
    dataset_paths=dataset_paths.split(",")

    output_path=args.output_path
    dataset_name=args.dataset_name
    out_file=output_path.split('.csv')[0]+str(int(time.time()))+'.csv'
    path_train, path_val, path_test = dataset_paths
    print(output_path)
    
    buffer_size=100000

    link_dict, training_dataset, _=from_file_list_to_tfdataset(dataset_name,path_train,return_df=True)
    training_dataset=training_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    link_dict, val_dataset, _=from_file_list_to_tfdataset(dataset_name,path_val,return_df=True,relation_dict=link_dict)
    link_dict, test_dataset, test_df=from_file_list_to_tfdataset(dataset_name,path_test,return_df=True,relation_dict=link_dict)
    
    # Initialize BERT
    LLM_encoder, preprocessor, tokenize = init_BERT()


    configuration = {
        'kNN': 10,
        'distance': 'cosine',
        'temperature': 0.2,
        'base_temperature': 1.,
        'n_jobs': -1,
        'top_perf': 1000 ,
        'eval_method': 'multi',
        'output_dim': 15 ,
        'attention_features':37, #37 disrex, 58 nyt10d, wiki20m 81
        'dense_neurons': 15,
        'activation': 'swish' ,
        'val_batch':30000,
        'get_thresholds':False,
        'threshold':0.5,
        'architecture':'PARE_CL', #PARE_CL (original), PARE_CL_SIMPLE, PARE_CL_MLP
        'similarity_to_class':'kNN_chunks',
        'epochs': 50, #60 (original)
        'batch_size':128, #128 (original)
        'gpu_enabled':True
    }

    print(output_path+' con Attention Feautures: '+str(configuration['attention_features'])+' SentenceAttention Original')

    tf.keras.backend.clear_session()

    # Initialize the model
    model = SentenceAttention(LLM_encoder,   #Alternative model: SentenceAttention
                output_dim=configuration['output_dim'],
                attention_features=configuration['attention_features'],
                dense_neurons=configuration['dense_neurons'],
                activation=configuration['activation'])
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = Supervised_ML_Contrastive(
            temperature=configuration['temperature'],
            base_temperature=configuration['base_temperature'],
            distance=configuration['distance']
        )

    model.compile(loss=loss, optimizer=optimizer)

    if os.path.exists('model_'+dataset_name+'_contrastive_Pare_weights.h5'):

        history= model.predict(training_dataset.take(128).batch(configuration['batch_size']))
        model.load_weights('model_'+output_path+'_contrastive_Pare_weights.h5')

    else:

        # Train the model
        history = model.fit(training_dataset.batch(configuration['batch_size']),
                        epochs=configuration['epochs'],
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)],
                        validation_data=val_dataset.batch(configuration['batch_size']))
        
        # Save the model
        #model.save('model_'+dataset_name+'_contrastive_Pare', save_format="tf")
        # Save weights
        model.save_weights('model_'+dataset_name+'_contrastive_Pare_weights.h5')

    
    # Predict the results
    test_y_true, test_y_pred, test_y_score, results_perf = evaluation_and_performance(test_configuration=configuration,
                                                                                training_dataset=training_dataset,
                                                                                test_dataset=test_dataset,
                                                                                model=model
                                                                                )

    # Evaluate the model
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

    

