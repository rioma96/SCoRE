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
import time 
import concurrent.futures
from multiprocessing import Lock, Pool
from codecarbon import track_emissions
import argparse
import logging
from codecarbon import track_emissions


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
    # Opening TXT file
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



def check_if_head_tail_are_included_in_sentence(sentence, head_start, head_end, tail_start, tail_end):
    head_included = False
    tail_included = False
    if (head_start >= 0 and head_start<len(sentence)) and (head_end>0 and head_end < len(sentence)):
        head_included = True
    if (tail_start >= 0 and tail_start<len(sentence)) and (tail_end>0 and tail_end < len(sentence)):
        tail_included = True
    return head_included, tail_included



def find_entity_tensor_slice(frase, preprocessor,entity_start_pos, entity_end_pos):
    tokenized_to_entity_end = preprocessor.tokenize(tf.constant([frase[:entity_end_pos]])).to_list()[0]
    entity_tokenized=preprocessor.tokenize(tf.constant([frase[entity_start_pos:entity_end_pos]])).to_list()[0]
    all_token_slice=[]#Token dell'entità
    for tok in entity_tokenized:
        for i in tok:
            all_token_slice.append(i)
    entity_lenght=len(all_token_slice)
    
    all_token_sentence=[]
    for element in tokenized_to_entity_end:
        for i in element:
            all_token_sentence.append(i)

    # +1 because first token is CLS token and idex must be adjusted
    bound_end=len(all_token_sentence)+1
    bound_start=(bound_end-entity_lenght)
    
    return bound_start,bound_end


def check_if_head_tail_are_included_in_Tokens(sentence, preprocessor, h_s, h_e, t_s, t_e, bert_token_window=512):
    bert_token_window=bert_token_window-2 #Remove CLS and SEP token from token window
    max_index = max(h_s, h_e, t_s, t_e)
    sentence_to_index = sentence[:max_index]
    sub_sentence = tf.constant([sentence_to_index])
    tokenized_sub_text = preprocessor.tokenize(sub_sentence)
    tokens_of_sub_text=[]

    for row in tokenized_sub_text:
        # Iterate over elements in each row
        for element in row:
            for i in element.numpy():
                tokens_of_sub_text.append(i)

    if len(tokens_of_sub_text)>bert_token_window:
        return False,tokens_of_sub_text
    else:
        return True,tokens_of_sub_text

    
def add_entities_markers(sentence,h_s, h_e, t_s, t_e):
    sentence_with_markers=sentence[:h_s]+"[E1] "+sentence[h_s:h_e]+" [/E1]"+sentence[h_e:t_s]+"[E2] "+sentence[t_s:t_e]+" [/E2]"+sentence[t_e:]
    new_h_s=h_s+0 #adjust head start adding 4 char of <e1>
    new_h_e=h_e+11 #adjust head end addung 4 char of <e1>
    new_t_s=t_s+11 #adjusting tail start adding 9 char of <e1> </e1> <e2>
    new_t_e=t_e+22 #adjusting tail end adding 18 char of <e1> </e1> <e2>
    return sentence_with_markers,new_h_s,new_h_e,new_t_s,new_t_e

####Function to generate sentence embedding for each row of a dataframe####
####Returns a list containing a tensor for CLS, tensors for head tokens and tail tokens#####
def embedd_row(row,model,preprocessor,markers=False):
    if markers:
        row['sentence'], row['head_start'], row['head_end'], row['tail_start'], row['tail_end'] = add_entities_markers(row['sentence'],row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
    emt=model(tf.constant([row['sentence']]))
    included,_=check_if_head_tail_are_included_in_Tokens(row['sentence'], preprocessor, row['head_start'], row['head_end'],
                                                         row['tail_start'], row['tail_end'], bert_token_window=512)
    head_included, tail_included = check_if_head_tail_are_included_in_sentence(row['sentence'], row['head_start'], row['head_end'], row['tail_start'], row['tail_end'])
    if not included or not head_included or not tail_included:
        print("Entity not included in sentence: ", row['sentence'])
        print("Head included: ", head_included)
        print("Tail included: ", tail_included)
        print("Included in tokens: ", included)
        print("Head start: ", row['head_start'])
        print("Head end: ", row['head_end'])
        print("Tail start: ", row['tail_start'])
        print("Tail end: ", row['tail_end'])
        return np.nan


    tensors_cls_head_tail=[]
    tensors_head_start_index, tensors_head_end_index=find_entity_tensor_slice(row['sentence'],preprocessor, row['head_start'], row['head_end'])
    tensors_tail_start_index, tensors_tail_end_index=find_entity_tensor_slice(row['sentence'],preprocessor, row['tail_start'], row['tail_end'])
    tensors_cls_head_tail.append(emt[0][0]) #CLS
    tensors_cls_head_tail.append(emt[0][tensors_head_start_index:tensors_head_end_index]) #head
    tensors_cls_head_tail.append(emt[0][tensors_tail_start_index:tensors_tail_end_index]) #tail

    return tensors_cls_head_tail
    

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
COLUMNS = ['syn_id_head', 'syn_id_tail', 'link_name', 'doc_pos', 'sent_start', 'sent_end',
           'head_start', 'head_end', 'tail_start', 'tail_end']
POSITION_COLUMNS = ['doc_pos', 'sent_start', 'sent_end', 'head_start', 'head_end', 'tail_start', 'tail_end']
CHUNK_SIZE = 30000

# Utility function to generate chunks
def chunk_dataframe(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield df[start:start + chunk_size]

# Move process_chunk outside of preprocess_and_embed
def process_chunk(chunk):
    # Initialize BERT once
    logger.info("Initializing BERT model...")
    bert_model, preprocessor, _ = init_BERT()
    logger.info("BERT model initialized.")
    chunk['embeddings'] = chunk.apply(lambda row: embedd_row(row, bert_model, preprocessor), axis=1)
    return chunk[~chunk['embeddings'].isna()]

def preprocess_and_embed(dataset_name, NA=False):
    # Dataset specific configurations
    dataset_configs = {
        "nyt10d": ("RawDatasets/NYT10D/",
                    "Datasets/NYT10D/",
                    ["nyt10_test.txt", "nyt10_train.txt"], load_nyt10d),
        "wiki20m": ("RawDatasets/Wiki20m/",
                    "Datasets/Wiki20m/",
                    ["wiki20m_train.txt", "wiki20m_val.txt", "wiki20m_test.txt"], load_wiki20m),
        "nyt10m": ("RawDatasets/NYT10m/",
                    "Datasets/NYT10m/",
                    ["nyt10m_test.txt", "nyt10m_train.txt", "nyt10m_val.txt"], load_nyt10m),
        "disrex": ("RawDatasets/DisRex/DiS-ReX/english/",
                   "Datasets/DisRex/",
                   ["disrex_val.txt", "disrex_test.txt", "disrex_train.txt"], load_DisRex),
        "wiki20distant": ("RawDatasets/Wiki20Distant/",
                          "Datasets/Wiki20Distant/",
                          ["wiki20d_train.txt", "wiki20d_val.txt"], load_wiki20m),

    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} configuration not found.")

    path, path_out, file_names, load_dataset_function = dataset_configs[dataset_name]


    # Ensure output directory exists
    os.makedirs(path_out, exist_ok=True)

    for file in file_names:
        output_path = os.path.join(path_out, f"{dataset_name}{file}_chunk_0.pkl")
        if os.path.isfile(output_path):
            logger.info(f"Embedding for {file} already exists.")
            continue

        # Load and preprocess dataset
        logger.info(f"Loading and preprocessing {file}...")
        result, ct = load_dataset_function(path, file, COLUMNS, NA)

        result["doc_pos"] = result["doc_pos"].apply(lambda x: ct[x])
        result.rename(columns={"doc_pos": "sentence"}, inplace=True)

        # Process dataset in chunks
        logger.info(f"Processing {file} in chunks...")
        chunks = chunk_dataframe(result, CHUNK_SIZE)

        with Pool(processes=3) as pool:
            # Pass bert_model and preprocessor to the pool workers
            results = pool.starmap(process_chunk, [(chunk,) for chunk in chunks])
            pool.close()
            pool.join()

        # Save each processed chunk
        for i, chunk in enumerate(results):
            chunk_output_path = os.path.join(path_out, f"{dataset_name}{file}_chunk_{i}.pkl")
            if not os.path.isfile(chunk_output_path):
                logger.info(f"Saving chunk {i} for {file}...")
                chunk.to_pickle(chunk_output_path)
                logger.info(f"Chunk {i} saved to {chunk_output_path}")

            # Clean up memory
            del chunk
            gc.collect()

    logger.info(f"Processing completed for dataset {dataset_name}.")

# Main function
@track_emissions()
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process.')
    args = parser.parse_args()

    dataset = args.dataset
    use_NA = False

    preprocess_and_embed(dataset, use_NA)

    end_time = time.time()
    logger.info(f"Execution time: {end_time - start_time:.2f} seconds for dataset: {dataset}")

if __name__ == "__main__":
    main()