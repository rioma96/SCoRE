import random
import numpy as np
import tensorflow as tf

'''Warning di ram:
2025-07-06 14:13:18.775703: W tensorflow/core/kernels/data/prefetch_autotuner.cc:52] 
Prefetch autotuner tried to allocate 288921600 bytes after encountering the first element
 of size 288921600 bytes.This already causes the autotune ram budget to be exceeded.
 To stay within the ram budget, either increase the ram budget or reduce element size
'''
 
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




# A partire dal full_df crea il tensore usando solo il sottoinsieme del train che ci interessa
#   (indici selezionati in precedenza + selezionati ora)
def select_active_subset(full_df, selected_indices, step_size, sampling_function=random_sampling,seed=42):


    #funzione di sampling presa da parametro [riusabilità]
    new_indices = sampling_function(full_df,selected_indices,step_size,seed=seed)


    already_selected = set(selected_indices)
    final_indices = sorted(list(already_selected.union(new_indices)))
    df_subset = full_df.iloc[final_indices].reset_index(drop=True)          #Questo è "train_df" filtrato


    # Costruisco il tf.data.Dataset da df_subset     [controllare se manca qualcosa]
    inputs = np.array(df_subset['Concatenated'].tolist())
    labels = np.array(df_subset['link_name'].tolist())
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))


    return dataset, final_indices, df_subset
