import random
import numpy as np
import tensorflow as tf
from CBKGE.utilities_validation import evaluation_and_performance
import json
from active_learning_utils_ebu import *
import os



# Seleziona |step_size| indici casuali nel dataframe senza considerare gli "already_selected"
def random_sampling(full_df,selected_indices,step_size,seed=42):
    random.seed(seed)
    all_indices = set(range(len(full_df)))
    already_selected = set(selected_indices)
    remaining = list(all_indices - already_selected)
    if len(remaining) < step_size:
        new_indices = remaining
    else:
        new_indices = random.sample(remaining, step_size)
    return new_indices

'''
Per ogni classe (intesa come singola colonna link_name):
Trova gli indici non ancora selezionati (remaining_indices).
Da questi, prende max num sample casuali.
Evita duplicati (newly_selected tiene traccia).
Quando finiti i giri per classe:
Se non ha ancora step_size, completa con sampling random.
# Nota°: essendo multilable prendere un sample di una classe potrebbe portare a prendere contemporanemante 
# anche di un altra classe per minimizzare l'effetto collaterale si parte a prendere dalle classi meno rappresentate
'''
def fixed_num_and_random_remaining_sampling(full_df, selected_indices, step_size, num=5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    print(f"\n[DEBUG] STEP SIZE: {step_size}, MAX PER CLASS: {num}")
    print(f"[DEBUG] Selected_indices iniziali: {len(selected_indices)}")

    all_labels = np.array(full_df['link_name'].tolist())  # shape: (N, num_classes)
    num_classes = all_labels.shape[1]
    remaining_indices = list(set(range(len(full_df))) - set(selected_indices))  #come in random_sampling

    class_to_indices = {}

    # Trova gli indici per ogni classe (che ha 1 in quella colonna)
    for class_id in range(num_classes):
        class_indices = [i for i in remaining_indices if all_labels[i][class_id] == 1]      #lista di indici (sample) che contengono quella Label (classe)
        class_to_indices[class_id] = class_indices

    # Ordina le classi dalla meno rappresentata alla più rappresentata
    sorted_classes = sorted(class_to_indices.items(), key=lambda x: len(x[1]))

    print(f"[DEBUG] Classi trovate: {len(sorted_classes)} (ordinate per cardinalità)")

    newly_selected = []

    for class_id, indices in sorted_classes:
        # Escludi già selezionati (da questo step o precedenti)
        # set(indices) <-- quelli della classe attuale a cui tolgo quelli selezionati in questo passo di active dalle altre classi
        available = list(set(indices)  - set(newly_selected))

        if not available:
            print(f"[DEBUG] Classe {class_id}: nessun indice disponibile, skippo.")
            continue

        n_to_sample = min(len(available), num)
        sampled = random.sample(available, n_to_sample)

        print(f"[DEBUG] Classe {class_id}: {len(available)} disponibili → ne prendo {n_to_sample}")

        newly_selected.extend(sampled)

        if len(newly_selected) >= step_size:        #con "num" piccoli non si verifica mai
            print("[DEBUG] Raggiunto step_size durante fase classi.")
            newly_selected = newly_selected[:step_size]
            break

    # Se ancora non abbiamo raggiunto step_size completo con random su tutto il range di indici
    # Perchè alcune classi vuote
    remaining_pool = list(set(remaining_indices) - set(newly_selected))

    n_missing = step_size - len(newly_selected)
    if n_missing > 0:
        fill_random = random.sample(remaining_pool, min(n_missing, len(remaining_pool)))
        print(f"[DEBUG] Aggiungo {len(fill_random)} elementi random per raggiungere step_size")
        newly_selected.extend(fill_random)

    return newly_selected

''' Piccola demo:
import numpy as np

link_name = [
    [1, 0, 0],  # classe 0
    [1, 1, 0],  # classi 0 e 1
    [0, 0, 1],  # classe 2
    [0, 1, 0],  # classe 1
]

all_labels = np.array(link_name)  # shape: (4, 3)
print(all_labels)
num_classes = all_labels.shape[1]
print(f"Numero classi: {num_classes}")

class_to_indices = {}    #Dizionario in cui per ogni classe_id abbiamo set di 
                        # sample (indice) a cui appartiene (ha 1 per quella label)
indici=len(all_labels)
#Len(all_labels) dovrebbero essere gli indici rimanenti
for class_id in range(num_classes):
    class_indices = [i for i in range(indici) if all_labels[i][class_id] == 1]      #lista di indici (sample) che contengono quella Label (classe)
    class_to_indices[class_id] = class_indices
    
print(class_to_indices)

sorted_classes = sorted(class_to_indices.items(), key=lambda x: len(x[1]))
print(sorted_classes)
'''



''' Funzione per "Sampling Bilanciato"
Selezionare in modo equo da tutte le classi,  (step_size//num_classes)
Redistribuire equamente il budget (step_size) tra le classi che ancora hanno esempi disponibili nel caso alcune classi finiscano,
Fermarsi esattamente quando raggiungi step_size elementi.
Nota: rimane il problema che non è perfettamente bilanciato per il motivo descritto precedentemente.
'''
def balanced_sampling_until_step_size(full_df, selected_indices, step_size, seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

    all_labels = np.array(full_df['link_name'].tolist())  # shape: (N, num_classes)
    num_classes = all_labels.shape[1]
    remaining_indices = list(set(range(len(full_df))) - set(selected_indices))

   #analogo a funzione "fixed.."
    class_to_available = {}
    for class_id in range(num_classes):
        class_indices = [i for i in remaining_indices if all_labels[i][class_id] == 1]      
        class_to_available[class_id] = class_indices

    newly_selected = set()
    active_classes = set(class_to_available.keys())

    print(f"[DEBUG] Tentativo di bilanciamento per {step_size} elementi su {num_classes} classi")

    #Continua fino a che non ho aggiunto |step_size| sample e ho almeno una classe disponibile  (controllo per sicurezza)
    while len(newly_selected) < step_size and active_classes:
        budget_per_class = max((step_size - len(newly_selected)) // len(active_classes), 1)             #In questo modo sono sicuro di arrivare a prendere step_size el.
        print(f"[DEBUG] Iterazione: {len(newly_selected)}/{step_size} selezionati — budget per classe: {budget_per_class}")

        to_remove = set()               #class_id da rimuovere perchè vuote
        
        #prende per ogni classe la sua percentuale o i rimanenti, se non ci sono più sample la rimuove tra le attive
        for class_id in active_classes:
            available = list(set(class_to_available[class_id]) - newly_selected)

            if not available:
                to_remove.add(class_id)
                continue

            n_to_sample = min(len(available), budget_per_class)
            sampled = random.sample(available, n_to_sample)
            newly_selected.update(sampled)

            if len(newly_selected) >= step_size:                    #Se ci si dividesse pochi sample tipo 10 per 20 classi allora "budget.." sarebbe 1 a testa quindi usciamo prima
                break

        active_classes -= to_remove  # Rimuovi le classi esaurite

        if not active_classes:
            print("[DEBUG] Tutte le classi esaurite. Sampling completato in anticipo.")
            break

    selected_final = list(newly_selected)

    if len(selected_final) > step_size: #Non dovrebbe mai entrare qui
        print(f"WARNING: SELEZIONATI {len(selected_final) - step_size} sample in più")
        selected_final = selected_final[:step_size]  # Taglia esattamente a step_size

    print(f"[DEBUG] Totale selezionato: {len(selected_final)} elementi")
    
    return selected_final




#---------------------------------------------------------------
#               Entropy based Sampling
#---------------------------------------------------------------

# Computa l'entropy per un sample 
# Gli sarà passata la matrice intera:  (n_sample, n_laben)  con valore p_1 (proba. che sia 1)
# ES: 3 sample con 2 clasi
# s1:  [ 0.5 0.6  ]
# s2: ....
# La funzione trasforma ogni p_1 all'entropia associata e dopo fa la somma per riga
# ris: totale entropia per sample
def compute_entropy_of_samples(p_test):
    Hk = -(p_test * np.log2(p_test + 1e-10) + (1 - p_test) * np.log2(1 - p_test + 1e-10))
    sample_entropy = np.sum(Hk, axis=1)
    return sample_entropy

# La funzione calcolo l'entropia per ogni sample del pool (train rimanente) e prende i top_step_size
# incerti (>entropy)
def entropy_based_sampling(full_df, selected_indices, step_size, model, configuration, seed=42):

    random.seed(seed)
    np.random.seed(seed)

    remaining_indices = list(set(range(len(full_df))) - set(selected_indices))
    remaining_df = full_df.iloc[remaining_indices].reset_index(drop=True)

    # Costruiamo dataset TensorFlow da remaining_df (il pool)
    inputs = np.array(remaining_df['Concatenated'].tolist())
    labels = np.array(remaining_df['link_name'].tolist())
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    # Costruisco training dataset dagli already_selected (per la funzione "evaluation_and_performace")
    selected_df = full_df.iloc[selected_indices].reset_index(drop=True)
    inputs_train = np.array(selected_df['Concatenated'].tolist())
    labels_train = np.array(selected_df['link_name'].tolist())
    training_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))


    # Valuto il modello su questo pool per ottenere il vettore delle probabilità

    _, _, _, _, _, _, norm_w_batch, _, _ , _= evaluation_and_performance(
        test_configuration=configuration,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        model=model
    )

    print(f"[DEBUG] Shape di norm_w_batch {norm_w_batch[0].shape}")  #ti aspetti grande quanto il remaining training_df

    #norm_w_batch è il vettore delle probabilità, trasformiamo in array numpy così che "compute_entropy.." abbia il comportamento aspettato
    norm_w_batch = np.array(norm_w_batch)
    entropy_scores = compute_entropy_of_samples(norm_w_batch[0])

    # argsort ordina in modo crescente, [-step_size:] prende gli ultimi (+incerti) e [::-1] per invertire l'ordine
    top_k_idx_in_pool = np.argsort(entropy_scores)[-step_size:][::-1]

    # Devo ritornare gli indici top_k ma questi sono relativi a "remaning_indices", io li voglio  assoluti rispetto a full_idf
    selected_absolute_indices = [remaining_indices[i] for i in top_k_idx_in_pool]

    #------------------------------------------------------
    #               salvo info sull'entropia 
    #------------------------------------------------------
    debug = True      
    if debug:    
        # ---------------------
        # DEBUG: 
        # - 1° file   ( test funzionamento)
        # - 2° file  (analisi dei valori dell'entropia)
        # ---------------------
        debug_path = "debug_entropy.txt"
        with open(debug_path, "a") as f:
            f.write(f"\n\nGrandezza del train attuale: {len(selected_indices)}\n")
            f.write("==== Prime 10 predizioni (predict_proba) ====\n")
            for i in range(min(10, len(norm_w_batch[0]))):
                f.write(f"Sample {i}: {norm_w_batch[0][i].tolist()}\n")

            f.write("\n==== Prime 10 entropie ====\n")
            for i in range(min(10, len(entropy_scores))):
                f.write(f"Sample {i}: {entropy_scores[i]:.6f}\n")

            f.write("\n==== Top 10 entropie (ordinate) ====\n")
            top_10_entropy_values = entropy_scores[top_k_idx_in_pool[:10]]
            for i, score in enumerate(top_10_entropy_values):
                f.write(f"Rank {i+1}: {score:.6f}\n")

        print(f"[DEBUG] File di debug scritto in: {os.path.abspath(debug_path)}")

        # -------------------------
        # File 2: ad ogni iterazione salvo
        # --> train size , pool size
        # --> entropia min, max, avg      (entropia sul pool)
        # --> entropia avg per classe     (entropie sul pool)
        # --> analisi su num label per classe (media sul pool)

    analysis_dir = "entropy_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    entropy_analysis_path = os.path.join(analysis_dir, f"entropy_analysis_{seed}.txt")
    label_count_analysis_path = os.path.join(analysis_dir, f"class_label_entropy_analysis_{seed}.txt")


    all_labels = np.array(full_df['link_name'].tolist())  # shape: (N_total_samples, num_classes)
    num_classes = all_labels.shape[1]

    # Mappa: da indice assoluto a relativo (usato su entropy_scores)
    index_map = {abs_idx: i for i, abs_idx in enumerate(remaining_indices)}

    # Calcolo media entropia per classe, per ogni classe ottiene la lista di indici della classe assoluti (full_df)
    # ne calcola l'entropia (tramite index_map da cui ottiene indice  relativo) ==> fa la media e lo salva { class_id : media }
    entropy_per_class = {}
    for class_id in range(num_classes):
        class_indices = [i for i in remaining_indices if all_labels[i][class_id] == 1]
        if class_indices:
            entropies = [entropy_scores[index_map[i]] for i in class_indices]
            entropy_per_class[class_id] = round(float(np.mean(entropies)), 6)
        else:
            entropy_per_class[class_id] = None  # oppure 0.0

    analysis_data = {
        "train_size": len(selected_indices),
        "pool_size": len(remaining_indices),
        "entropy": {
            "min": round(float(np.min(entropy_scores)), 6),
            "mean": round(float(np.mean(entropy_scores)), 6),
            "max": round(float(np.max(entropy_scores)), 6),
        },
        "class_entropy_means": entropy_per_class
    }

    # === Salvataggio file entropia (come prima)
    with open(entropy_analysis_path, "a") as f:
        f.write(json.dumps(analysis_data) + "\n")

    # === NUOVO: Calcolo numero medio di label per i sample della classe
    class_avg_labels_per_sample = []

    for class_id in range(num_classes):
        class_indices = [i for i in remaining_indices if all_labels[i][class_id] == 1]
        if class_indices:
            label_counts = [np.sum(all_labels[i]) for i in class_indices]  # quante label ha ogni sample
            avg_labels = round(float(np.mean(label_counts)), 2)             # media con tronc a 2 valori
        else:
            avg_labels = -1.0  
        class_avg_labels_per_sample.append(avg_labels)

    # === Salvataggio riga in file .txt
    with open(label_count_analysis_path, "a") as f:
        line = " ".join(str(val) for val in class_avg_labels_per_sample)
        f.write(line + "\n")

    print(f"[DEBUG] Salvato anche file media label per classe in: {os.path.abspath(label_count_analysis_path)}")


    return selected_absolute_indices

''' DEMO:
full_df = 10 sample totali
already_selected = [1, 5, 8]
remaining_indices = [0, 2, 3, 4, 6, 7, 9]  # => 7 elementi nel pool

entropy_scores = np.array([0.1, 0.5, 0.2, 0.9, 0.3, 0.8, 0.6])
step_size = 3
np.argsort(entropy_scores) => [0, 2, 4, 1, 6, 5, 3]  (ordine da rovesciare)
[-3:][::-1] => [3, 5, 6] => entropie: [0.9, 0.8, 0.6]
top_k_idx_in_pool = [3, 5, 6]  (tramite argsort)
selected_absolute_indices = [remaining_indices[i] for i in top_k_idx_in_pool]
                        = [remaining_indices[3], remaining_indices[5], remaining_indices[6]]
                        = [4, 7, 9]     # Indici nel full_df!

'''



'''
 La funzione è una estensione di "balanced_sampling_..." in cui sostituisce
 "   sampled = random.sample(available, n_to_sample)   "  con la logica di selezione
 per entropia.
'''
def balanced_entropy_sampling(full_df, selected_indices, step_size, model, configuration, seed=42):

    # Questa parte iniziale analoga a "entropy2"
    # Creo il pool (indici rimanenti da selezionare)
    all_labels = np.array(full_df['link_name'].tolist())  # Multi-label (binaria per classe)
    num_classes = all_labels.shape[1]

    remaining_indices = list(set(range(len(full_df))) - set(selected_indices))
    remaining_df = full_df.iloc[remaining_indices].reset_index(drop=True)

    # Costruiamo dataset TensorFlow da remaining_df (il pool)
    inputs = np.array(remaining_df['Concatenated'].tolist())
    labels = np.array(remaining_df['link_name'].tolist())
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    # Costruisco training dataset dagli already_selected (per la funzione "evaluation_and_performace")
    selected_df = full_df.iloc[selected_indices].reset_index(drop=True)
    inputs_train = np.array(selected_df['Concatenated'].tolist())
    labels_train = np.array(selected_df['link_name'].tolist())
    training_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))

    try:    
        #ottengo probabilità
        # Qui chiama "evaluation.. e usa norm_w_batch"
        _, _, _, _, _, _, norm_w_batch, _, _ , _= evaluation_and_performance(
        test_configuration=configuration,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        model=model
    )
        proba_batch = norm_w_batch[0]
        print(f"Norm_w_bach shape: {norm_w_batch[0].shape}")
    except Exception as e:
        print(f"[ERRORE] durante model.predict: {e}")
        raise

    entropy_scores = compute_entropy_of_samples(proba_batch)

    # Creo mappa da indice assoluto (global) → indice relativo (tra i remaining su cui 
    # calcoliamo l'entropia)  [guarda demo]
    index_map = {abs_idx: i for i, abs_idx in enumerate(remaining_indices)}

    # Associa a ogni classe i sample disponibili (indici assoluti)  (come nella precedente "balanced..")
    # per ogni classe id [0,1..] associa la lista di indici che hanno quella classe (all_label[i]=sample_i,all_labels[i][j] --> sample i classe j)
    # Questo è corretto perchè all_labels è la matrice (sample,classi) creata sopra
    class_to_available = {
        class_id: [i for i in remaining_indices if all_labels[i][class_id] == 1] for class_id in range(num_classes)
    }

    newly_selected = set()          #dobbiamo tenerne traccia
    active_classes = set(class_to_available.keys())     #uguale

    print(f"[DEBUG] Tentativo di bilanciamento per {step_size} elementi su {num_classes} classi")

    while len(newly_selected) < step_size and active_classes:
        budget_per_class = max((step_size - len(newly_selected)) // len(active_classes), 1)         #budget di questa it per ogni classe
        print(f"[DEBUG] Iterazione: {len(newly_selected)}/{step_size} selezionati — budget per classe: {budget_per_class}")
        to_remove = set()

        for class_id in active_classes:
            available = list(set(class_to_available[class_id]) - newly_selected)

            if not available:       
                to_remove.add(class_id)
                continue

            # Ordina per entropia decrescente usando la index_map  [qui estensione rispetto a "balanced_until.."]
            # Vedi demo
            top_entropy = sorted(
                available,
                key=lambda i: entropy_scores[index_map[i]],     #index map associa l'indice globale al relativo dei rimainig (perchè la matrice entropy è solo dei remaining)
                reverse=True
            )

            selected = top_entropy[:min(len(top_entropy), budget_per_class)]        #seleziono le prime k = min(..)
            newly_selected.update(selected)

            if len(newly_selected) >= step_size:
                break

        active_classes -= to_remove

        if not active_classes:
            print("[DEBUG] Tutte le classi esaurite. Sampling completato in anticipo.")
            break

    selected_final = list(newly_selected)[:step_size]
    print(f"[DEBUG] Totale selezionato: {len(selected_final)} elementi")
    return selected_final

''' DEMO    
Immaginamo un ts con 10 sample, quindi gli indici totali full_df: [0 1 2 3 4 .. 9]

remaining_indices = list(set(range(10)) - set(selected_indices))
# Ad esempio: [1, 3, 4, 6, 7, 8, 9]

index_map = {abs_idx: i for i, abs_idx in enumerate(remaining_indices)}
# Risultato:     
# {
#   1: 0,
#   3: 1,
#   4: 2,
#   6: 3,
#   7: 4,
#   8: 5,
#   9: 6  ==> ovvero l'indice 9 globalmente ma posizione 6 in "remaining_indices"
# }    

Quindi ora:
top_entropy = sorted(
                available,
                key=lambda i: entropy_scores[index_map[i]],
                reverse=True
            )
--> "available" lista di indici available per la classe (tolto gli eventuali newly_selected)
    [Sono un sottoinsieme dei remaining indices]:
    Ad esempio   [1,4]
    La loro entropia è quindi:  Dove i = 1 e i = 4 , 
        entropy_scores[index_map[i]],
    index_map[i] ne trova l'indice locale ai RIMANENTI!! in linea con la matrice "entropy_scores" che è calcolato sul training rimanente
    Ovvero l'entropia dell'indice "1" si trova in realtà in entropy_scores[0] come dice la index_map
'''



# EBU --> riprende esattamente il codice dell'entropia ma:
#   - seleziona i candidati:  top step_size*mul entropici
#   - tra questi seleziona secondo EBU (necessita binarizzazione del pool --> fatta tramite model.predict)
def entropy_with_EBU(full_df, selected_indices, step_size, model, configuration, seed=42, mul=5, argmin = True):

    remaining_indices = list(set(range(len(full_df))) - set(selected_indices))
    remaining_df = full_df.iloc[remaining_indices].reset_index(drop=True)

    # Costruisco test dataset (il pool)
    inputs = np.array(remaining_df['Concatenated'].tolist())
    labels = np.array(remaining_df['link_name'].tolist())
    test_dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    # Costruisco training dataset
    selected_df = full_df.iloc[selected_indices].reset_index(drop=True)
    inputs_train = np.array(selected_df['Concatenated'].tolist())
    labels_train = np.array(selected_df['link_name'].tolist())
    training_dataset = tf.data.Dataset.from_tensor_slices((inputs_train, labels_train))

    # Valutazione del modello sul pool
    _, y_pred, _, _, _, _, norm_w_batch, _, _, _ = evaluation_and_performance(
        test_configuration=configuration,
        training_dataset=training_dataset,
        test_dataset=test_dataset,
        model=model
    )

    print(f"[DEBUG] Shape di norm_w_batch {norm_w_batch[0].shape}")  # ti aspetti grande quanto il remaining_df

    # Calcolo entropia per sample
    norm_w_batch = np.array(norm_w_batch)
    entropy_scores = compute_entropy_of_samples(norm_w_batch[0])

    # Top-k candidati più entropici 
    k = min(step_size * mul, len(entropy_scores))
    top_k_idx_in_pool = np.argsort(entropy_scores)[-k:][::-1]

    #Selezione finale tramite ebu, che vuole il pool binario 
    X_pool_prob = model.predict(inputs, verbose=0)
    test_dataset_bin = (X_pool_prob > 0).astype(int)  
    print(f"DEBUG: shape pool_bin {test_dataset_bin.shape}")
    print(f"DEBUG: shape test_y_pred {y_pred.shape}")
    
    '''
    print("\n[DEBUG] Prime 10 righe di test_dataset_bin (binarizzate da predict > 0):")
    for i in range(min(10, len(test_dataset_bin))):
        print(f"Sample {i}: {test_dataset_bin[i].tolist()}")

    print("\n[DEBUG] Prime 10 righe di y_pred:")
    for i in range(min(10, len(y_pred))):
        print(f"Sample {i}: {y_pred[i].tolist()}")

    print("\n[DEBUG] Prime 10 righe di norm w batch:")
    for i in range(min(10, len(norm_w_batch[0]))):
        print(f"Sample {i}: {norm_w_batch[0][i].tolist()}") '''

    final_indices = select_by_ebu_multilabel(
                                    test_dataset_bin,
                                    top_k_idx_in_pool,
                                    step_size,
                                    y_pred,
                                    argmin
                                    )

    # Conversione da indici locali (pool) a globali
    selected_absolute_indices = [remaining_indices[i] for i in final_indices]
    return selected_absolute_indices





# A partire dal full_df crea il tensore usando solo il sottoinsieme del train che ci interessa
#   (indici selezionati in precedenza + selezionati ora)
#  argmin parametro dell'ebu
def select_active_subset(full_df, selected_indices, step_size, 
                        sampling_function=random_sampling, seed=42, 
                        model=None, configuration=None,argmin=True):        
    
    #Alla prima iterazione sarà random sampling, dopo quella passata come parametro
    if selected_indices == []:
        new_indices = random_sampling(full_df, selected_indices, step_size, seed)
    elif sampling_function.__name__ in [ "entropy_based_sampling" ,  "balanced_entropy_sampling"]:
        assert model is not None and configuration is not None, "Model e configuration sono obbligatori per entropy sampling"
        new_indices = sampling_function(full_df, selected_indices, step_size, model, configuration, seed)

    elif sampling_function.__name__ in [ "entropy_with_EBU"]:
        assert model is not None and configuration is not None, "Model e configuration sono obbligatori per entropy sampling"
        new_indices = sampling_function(full_df, selected_indices, step_size, model, configuration, seed,argmin)

    else:
        new_indices = sampling_function(full_df, selected_indices, step_size, seed)


    already_selected = set(selected_indices)
    final_indices = sorted(list(already_selected.union(new_indices)))
    df_subset = full_df.iloc[final_indices].reset_index(drop=True)          #Questo è "train_df" filtrato


    # Costruisco il tf.data.Dataset da df_subset     [controllare se manca qualcosa]
    inputs = np.array(df_subset['Concatenated'].tolist())
    labels = np.array(df_subset['link_name'].tolist())
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))


    return dataset, final_indices, df_subset


    

