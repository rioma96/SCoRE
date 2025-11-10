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
import time
import argparse
import logging
from codecarbon import EmissionsTracker

# Trasformers e TF (tokenizer fast + TFBertModel)
from transformers import BertTokenizerFast, TFBertModel
import tensorflow as tf
import h5py
from tqdm import tqdm


def load_wiki20m(path, file_name,columns,NA=False):
    
    raw_data = []
    clean_text=[]
    clean_text_map = {}
    processed_data = []
    # Opening JSON file
    with open(path+file_name,encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) > 0:
                raw_data.append(json.loads(line))
    
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
        if text in clean_text_map:
            clean_text_position = clean_text_map[text]
        else:
            clean_text_position = len(clean_text)
            clean_text.append(text)
            clean_text_map[text] = clean_text_position
    
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
                raw_data.append(json.loads(line))
    
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
    clean_text_map = {}
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
        if text in clean_text_map:
            clean_text_position = clean_text_map[text]
        else:
            clean_text_position = len(clean_text)
            clean_text.append(text)
            clean_text_map[text] = clean_text_position

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
                except Exception as e:
                    erroneous_counter+=1
                    logger.debug("Skipped triple due to error: %s", e)
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
    clean_text_map = {}
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
        if text in clean_text_map:
            clean_text_position = clean_text_map[text]
        else:
            clean_text_position = len(clean_text)
            clean_text.append(text)
            clean_text_map[text] = clean_text_position
            
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
                except Exception as e:
                    erroneous_counter+=1
                    logger.debug("Rebel: skipped triple due to error: %s", e)
    if not 'distant_y' in columns:
        columns.insert(3,'distant_y')
    processed_data=pd.DataFrame(processed_data,columns= columns)
    # print(erroneous_counter/(erroneous_counter+positive_counter))
    return processed_data, clean_text


# Per mantenere consistenza con il tuo codice originale:
COLUMNS = ['syn_id_head', 'syn_id_tail', 'link_name', 'doc_pos', 'sent_start', 'sent_end',
           'head_start', 'head_end', 'tail_start', 'tail_end']
POSITION_COLUMNS = ['doc_pos', 'sent_start', 'sent_end', 'head_start', 'head_end', 'tail_start', 'tail_end']
CHUNK_SIZE = 30000  # lo mantengo come nel tuo codice originale; puoi ridurlo in base alla RAM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Setup modello e tokenizer (caricati una volta)
# ============================================================
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 512  # finestra BERT, pari a quanto usavi
DEFAULT_BATCH_SIZE = 64

def init_transformers(model_name=MODEL_NAME):
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = TFBertModel.from_pretrained(model_name)
    # Impostazioni opzionali per migliori performance (XBENCH/TF)
    # tf.config.optimizer.set_jit(True)  # XLA JIT (opzionale, testare)
    return tokenizer, model

# ============================================================
# Funzione di utilità per chunking
# ============================================================
def chunk_dataframe(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield df[start:start + chunk_size]

# ============================================================
# Embedding batch (tokenizer fast + offset mapping)
# ============================================================
def embed_batch(sentences, head_spans, tail_spans, tokenizer, model,
                batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_LENGTH):
    """
    Restituisce una lista con lo stesso ordine delle 'sentences'.
    Per le righe non processabili (es: entità troncata) restituisce None.
    Altrimenti restituisce dict {'cls': np.array, 'head': np.array, 'tail': np.array}.
    """
    results = []
    model_trainable_flag = getattr(model, 'training', False)
    for i in range(0, len(sentences), batch_size):
        batch_sents = sentences[i:i+batch_size]
        batch_heads = head_spans[i:i+batch_size]
        batch_tails = tail_spans[i:i+batch_size]

        # Tokenizzazione con offset mapping (fast tokenizer)
        enc = tokenizer(
            batch_sents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="tf"
        )

        # Forward pass (disabilitiamo gradiente)
        # training=False passed to model call; avoid deprecated set_learning_phase
        outputs = model(enc["input_ids"], attention_mask=enc["attention_mask"], training=False)
        last_hidden = outputs.last_hidden_state  # Tensor shape (batch, seq_len, hidden)
        # Convertire offset mapping e last_hidden in numpy array (batch, seq_len, ...)
        try:
            last_hidden_np = last_hidden.numpy()
        except Exception:
            # In some TF setups, tensor may already be numpy-like; fallback
            last_hidden_np = np.array(last_hidden)
        offset_mapping = enc["offset_mapping"].numpy()

        for b_idx in range(len(batch_sents)):
            sent = batch_sents[b_idx]
            h_start, h_end = batch_heads[b_idx]
            t_start, t_end = batch_tails[b_idx]

            # Check bounds carattere rispetto alla lunghezza della frase
            if not (0 <= h_start < h_end <= len(sent)) or not (0 <= t_start < t_end <= len(sent)):
                # invalid char bounds -> skip
                logger.debug("Bounds invalid or out of sentence range; skipping.")
                results.append(None)
                continue

            offsets = offset_mapping[b_idx]  # array di (start,end) per token

            # Trova token i cui offset overlap con l'entità (condizione overlap migliore)
            head_token_idxs = [j for j, (s, e) in enumerate(offsets) if e > 0 and (s < h_end and e > h_start)]
            tail_token_idxs = [j for j, (s, e) in enumerate(offsets) if e > 0 and (s < t_end and e > t_start)]

            # Se è vuoto vuol dire che la tokenizzazione ha troncato l'entità o l'entità non è riconosciuta
            if len(head_token_idxs) == 0 or len(tail_token_idxs) == 0:
                logger.debug(f"Entity tokens not found (maybe truncated). head_idxs:{head_token_idxs} tail_idxs:{tail_token_idxs}")
                results.append(None)
                continue

            # Calcola embeddings (mean pooling sui token dell'entità)
            try:
                cls_emb = last_hidden_np[b_idx][0]
                head_emb = np.mean(last_hidden_np[b_idx][head_token_idxs, :], axis=0)
                tail_emb = np.mean(last_hidden_np[b_idx][tail_token_idxs, :], axis=0)
                results.append({"cls": cls_emb, "head": head_emb, "tail": tail_emb})
            except Exception as e:
                logger.exception("Error while extracting embeddings for a batch element: %s", e)
                results.append(None)

        # cleanup per batch
        del enc, outputs, last_hidden, offset_mapping
        gc.collect()

    return results

# ============================================================
# Funzione che processa un chunk (DataFrame) e salva HDF5 + meta CSV
# ============================================================
def process_and_save_chunk(df_chunk, tokenizer, model, dataset_name, file_name, out_dir,
                           chunk_id, batch_size=DEFAULT_BATCH_SIZE, max_length=MAX_LENGTH):
    """
    df_chunk: DataFrame che contiene almeno le colonne:
        'sentence','head_start','head_end','tail_start','tail_end'
    Salva:
      - out_dir/{dataset_name}_{file_safe}_chunk_{chunk_id}.h5  (datasets: cls, head, tail)
      - out_dir/{dataset_name}_{file_safe}_chunk_{chunk_id}_meta.csv  (metadati delle righe valide)
    """
    os.makedirs(out_dir, exist_ok=True)
    sentences = df_chunk["sentence"].tolist()
    head_spans = list(zip(df_chunk["head_start"], df_chunk["head_end"]))
    tail_spans = list(zip(df_chunk["tail_start"], df_chunk["tail_end"]))

    embeddings = embed_batch(sentences, head_spans, tail_spans, tokenizer, model,
                             batch_size=batch_size, max_length=max_length)

    valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
    if len(valid_indices) == 0:
        logger.info(f"No valid embeddings in chunk {chunk_id} for file {file_name}. Nothing saved.")
        return

    # Costruisci array numpy da salvare (solo righe valide)
    cls_arr = np.stack([embeddings[i]["cls"] for i in valid_indices])
    head_arr = np.stack([embeddings[i]["head"] for i in valid_indices])
    tail_arr = np.stack([embeddings[i]["tail"] for i in valid_indices])

    file_safe = file_name.replace(os.sep, "_").replace(".", "_")
    lower_file = file_name.lower()
    if 'test' in lower_file:
        split = 'test'
    elif 'train' in lower_file:
        split = 'train'
    elif 'val' in lower_file or 'validation' in lower_file:
        split = 'val'
    else:
        split = 'other'

    target_dir = os.path.join(out_dir, dataset_name, split)
    os.makedirs(target_dir, exist_ok=True)
    pkl_path = os.path.join(target_dir, f"{dataset_name}{file_safe}_chunk_{chunk_id}.pkl")
    meta_path = os.path.join(target_dir, f"{dataset_name}{file_safe}_chunk_{chunk_id}_meta.csv")

    # Salva come pickle (dict di numpy arrays)
    to_save = {"cls": cls_arr, "head": head_arr, "tail": tail_arr}
    try:
        with open(pkl_path, "wb") as pf:
            pickle.dump(to_save, pf, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved embeddings to {pkl_path} ({cls_arr.shape[0]} examples).")
    except Exception as e:
        logger.exception("Failed to save pickle file %s: %s", pkl_path, e)

    # Salva metadati (righe del dataframe corrispondenti ai valid_indices)
    df_meta = df_chunk.iloc[valid_indices].copy()
    # Conserva anche l'indice originale per tracciabilità
    df_meta.reset_index(inplace=True)
    df_meta.rename(columns={"index": "orig_index"}, inplace=True)
    df_meta.to_csv(meta_path, index=False, encoding='utf-8')
    logger.info(f"Saved metadata to {meta_path}.")

    # liberiamo memoria
    del cls_arr, head_arr, tail_arr, embeddings, df_meta, to_save
    gc.collect()

# ============================================================
# Funzione principale: integra i loader originali e l'embedding ottimizzato
# ============================================================
def preprocess_and_embed(dataset_name, NA=False, chunk_size=CHUNK_SIZE, batch_size=DEFAULT_BATCH_SIZE, out_root="Datasets_Emb"):
    """
    dataset_name: come prima (es 'nyt10d', 'wiki20m', ...)
    NA: se includere NA relations
    Salva per ogni file del dataset i chunk in out_root/
    """
    # Config dataset (come nel tuo codice)
    dataset_configs = {
        "nyt10d": ("RawDatasets/NYT10D/", "Datasets/NYT10D/", ["nyt10_test.txt", "nyt10_train.txt"], load_nyt10d),
        "wiki20m": ("RawDatasets/Wiki20m/", "Datasets/Wiki20m/", ["wiki20m_train.txt", "wiki20m_val.txt", "wiki20m_test.txt"], load_wiki20m),
        "nyt10m": ("RawDatasets/NYT10m/", "Datasets/NYT10m/", ["nyt10m_test.txt", "nyt10m_train.txt", "nyt10m_val.txt"], load_nyt10m),
        "disrex": ("RawDatasets/DiS-ReX/english/", "Datasets/DisRex/", ["disrex_val.txt", "disrex_test.txt", "disrex_train.txt"], load_DisRex),
        "wiki20distant": ("RawDatasets/Wiki20Distant/", "Datasets/Wiki20Distant/", ["wiki20d_train.txt", "wiki20d_val.txt"], load_wiki20m),
        # aggiungi altri se li hai
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} configuration not found.")

    raw_path, path_out, file_names, load_dataset_function = dataset_configs[dataset_name]
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(out_root, exist_ok=True)

    # Inizializza tokenizer + modello UNA SOLA VOLTA
    tokenizer, model = init_transformers(MODEL_NAME)

    for file in file_names:
        # Decide split folder from filename (train/val/test/other)
        fname_lower = file.lower()
        if 'test' in fname_lower:
            split = 'test'
        elif 'train' in fname_lower:
            split = 'train'
        elif 'val' in fname_lower or 'validation' in fname_lower:
            split = 'val'
        else:
            split = 'other'

        # Costruisco il percorso del file meta di controllo sotto out_root/dataset_name/split
        sample_meta_path = os.path.join(out_root, dataset_name, split, f"{dataset_name}{file}_chunk_0_meta.csv")
        if os.path.isfile(sample_meta_path):
            logger.info(f"Embedding for {file} already exists (meta file found at {sample_meta_path}). Skipping.")
            continue

        logger.info(f"Loading and preprocessing {file}...")
        # Carica dataset (le tue funzioni ritornano processed_data, clean_text in genere)
        result, ct = load_dataset_function(raw_path, file, COLUMNS, NA)

        # Sostituisci doc_pos con il testo effettivo dalla lista clean_text
        # Se la funzione restituisce clean_text come list/serie (come nel tuo codice)
        try:
            result["doc_pos"] = result["doc_pos"].apply(lambda x: ct[x])
        except Exception as e:
            # Se la funzione già restituisce 'sentence' o ha struttura diversa
            logger.debug("Non ho potuto mappare doc_pos con ct: %s", e)
            # Se 'doc_pos' non esiste, prova a usare colonna 'sentence' già presente
            if 'doc_pos' not in result.columns and 'sentence' in result.columns:
                result['doc_pos'] = result['sentence']
            else:
                raise

        # Rinomina in 'sentence' per compatibilità
        result.rename(columns={"doc_pos": "sentence"}, inplace=True)

        # Assicuriamoci che le colonne head_start/head_end siano int e valide
        for col in ["head_start", "head_end", "tail_start", "tail_end"]:
            if col not in result.columns:
                raise ValueError(f"Column {col} missing in processed data for {file}.")
            result[col] = result[col].astype(int)

        logger.info(f"Processing {file} in chunks (chunk_size={chunk_size})...")

        # Itera i chunk
        chunk_iter = chunk_dataframe(result, chunk_size)
        for i, chunk in enumerate(chunk_iter):
            logger.info(f"Processing chunk {i} for file {file} ({len(chunk)} rows)...")

            # Convert to contiguous DataFrame and drop duplicates (come facevi tu)
            chunk = chunk.reset_index(drop=True)
            chunk = chunk.loc[~chunk.astype(str).duplicated()].reset_index(drop=True)

            # Process & save chunk (HDF5 + meta CSV)
            try:
                process_and_save_chunk(chunk, tokenizer, model,
                                       dataset_name, file, out_root,
                                       chunk_id=i, batch_size=batch_size, max_length=MAX_LENGTH)
            except Exception as e:
                logger.exception("Errore durante process_and_save_chunk: %s", e)

            # cleanup
            del chunk
            gc.collect()

    logger.info(f"Processing completed for dataset {dataset_name}.")

# ============================================================
# Main: CLI e wrapper
# ============================================================
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset to process.')
    parser.add_argument('--chunk_size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--out_dir', type=str, default="Datasets_Emb")
    args = parser.parse_args()

    dataset = args.dataset
    use_NA = False

    # Use explicit EmissionsTracker to measure and save consumption
    tracker = EmissionsTracker(project_name=f"GenDataOptimized-{dataset}")
    tracker.start()
    try:
        preprocess_and_embed(dataset, NA=use_NA, chunk_size=args.chunk_size, batch_size=args.batch_size, out_root=args.out_dir)
    finally:
        emissions = tracker.stop()  # returns emissions in kg CO2eq
        # Get energy consumed in kWh if available from tracker. Use tracker._last_emissions_data if present.
        kwh = None
        try:
            kwh = tracker._total_energy.kWh
            # energy_consumed is in kWh according to codecarbon internals
    
        except Exception:
            kwh = None

        # Save kWh to file out_dir/{dataset}consumi
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            out_file = os.path.join(args.out_dir, f"{dataset}_consumi.txt")
            with open(out_file, 'w', encoding='utf-8') as f:
                if kwh is not None:
                    f.write(str(kwh))
                else:
                    # fallback: write emissions (kg CO2eq) if kWh not available
                    f.write(str(emissions))
        except Exception as e:
            logger.exception("Failed to write emissions file: %s", e)

        end_time = time.time()
        logger.info(f"Execution time: {end_time - start_time:.2f} seconds for dataset: {dataset}")

if __name__ == "__main__":
    main()