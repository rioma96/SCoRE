import pandas as pd
import json
import matplotlib.pyplot as plt


'''
print("ACTIVE WITH full train")

df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_1_with_entropy.pkl')
print(df)
print("\n\n\n")



print("ACTIVE WITH random sampling")
for i in range(1,5):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_random.pkl')
    print(df)
print("\n\n\n")

print("ACTIVE WITH entropy")
for i in range(1,17):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy.pkl')
    print(df)
print("\n\n\n")



print("ACTIVE WITH fixed_sampling")
for i in range(1,35):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_fixed.pkl')
    print(df)
print("\n\n\n")

print("ACTIVE WITH balanced_sampling")
for i in range(1,35):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_balanced.pkl')
    print(df)
print("\n\n\n")


print("PERFORMANCE con training completo")
print("Prima run:")
df = pd.read_pickle('nyt10m_results_performance_0.pkl')
print(df)
print("Seconda run:")
df = pd.read_pickle('nyt10m_results_performance_1.pkl')
print(df)
print("Terza run:")
df = pd.read_pickle('nyt10m_results_performance_2.pkl')
print(df)
'''



#==============================  PLOTTING ==================================
import pandas as pd
import matplotlib.pyplot as plt

metrics = ['micro', 'Macro', 'm@100', 'M@100', 'm@200', 'M@200', 'm@300', 'M@300', 'm@1000', 'M@1000', 'P@R']

# Carica performance su tutto il training set
full_df = pd.read_pickle('nyt10m_results_performance_0.pkl')
full_row = full_df.iloc[0]  # Estrae l'unica riga


#             PLOT PER  FIXED
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 35):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_fixed.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

     # Aggiungi punto rosso per la performance full training
    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

# Rimuovi eventuale subplot vuoto (nel caso 11 metriche su 12 slot)
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Fixed_sampling (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_fixed.png", bbox_inches='tight')



#             PLOT PER  BALANCED
#
results = {metric: [] for metric in metrics}
iterations = []

for i in range(1, 35):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_balanced.pkl')
        row = df.iloc[0]  
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Balanced_sampling (40° per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_balanced.png", bbox_inches='tight')



#             RANDOM
#
results = {metric: [] for metric in metrics}
iterations = []

for i in range(1, 19):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_random.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Random_sampling (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_random.png", bbox_inches='tight')


#             PLOT PER  Entropy
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 17):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

     # Aggiungi punto rosso per la performance full training
    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

# Rimuovi eventuale subplot vuoto (nel caso 11 metriche su 12 slot)
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Entropy sampling (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_entropy.png", bbox_inches='tight')

#
#             PLOT PER  Entropy balanced
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 17):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_balanced.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

     # Aggiungi punto rosso per la performance full training
    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

# Rimuovi eventuale subplot vuoto (nel caso 11 metriche su 12 slot)
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Balanced_sampling (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_entropy_balanced.png", bbox_inches='tight')




#
#             PLOT PER  Entropy_EBU_ARGMIn
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 14):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_EBU_argmin.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

     # Aggiungi punto rosso per la performance full training
    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

# Rimuovi eventuale subplot vuoto (nel caso 11 metriche su 12 slot)
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Entropy-Ebu-Argmin (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_entropy_EBU_argmin.png", bbox_inches='tight')


#
#             PLOT PER  Entropy_EBU_ARGMAX
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 14):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_EBU_argmax.pkl')
        row = df.iloc[0]  # Prende l'unica riga del DataFrame
        for metric in metrics:
            results[metric].append(row[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i}: {e}")

# Plotting
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    axes[idx].plot(iterations, results[metric], marker='o')

     # Aggiungi punto rosso per la performance full training
    axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')  

    axes[idx].set_title(metric)
    axes[idx].set_xlabel('Iterazione')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(True)

# Rimuovi eventuale subplot vuoto (nel caso 11 metriche su 12 slot)
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Entropy-Ebu-Argmax (40° it per tot training)', fontsize=16, y=1.02)

plt.savefig("active_learning_metrics_entropy_EBU_argmax.png", bbox_inches='tight')








#
#       PLOT nello stesso grafico     [#commenta per visualizzare solo dei sottoinsiemi]
# 
metrics = ['micro', 'Macro']

# Inizializza dizionari per le metriche
fixed_results = {metric: [] for metric in metrics}
balanced_results = {metric: [] for metric in metrics}
entropy_results = {metric: [] for metric in metrics}
random_results = {metric: [] for metric in metrics}
entropy_balanced_results = {metric: [] for metric in metrics}  
entropy_EBU_argmin_results = {metric: [] for metric in metrics}  
entropy_EBU_argmax_results = {metric: [] for metric in metrics}  



iterations = []
entropy_iterations = []
random_iterations = []
entropy_balanced_iterations = []  
entropy_EBU_argmin_iterations = []
entropy_EBU_argmax_iterations = []

# Mettere a true si hanno tutti i risultati fino a 40 iterazioni
benchmark_completo = False
d_completo = {
    "tot_it":40,
    "i_entropy":40,
    "i_random":40,
    "i_entropy_balanced":40,
    "i_entropry_EBU_argmin":40,
    "i_entropy_EBU_argmax":40
}
d_NON_completo = {
    "tot_it":35,
    "i_entropy":17,
    "i_random":28,
    "i_entropy_balanced":17,
    "i_entropry_EBU_argmin":14,
    "i_entropy_EBU_argmax":14,
}
if benchmark_completo:
    d = d_completo
else:
    d = d_NON_completo


# Caricamento dati
for i in range(1, d["tot_it"]):

    try:
        df_fixed = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_fixed.pkl')
        df_balanced = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_balanced.pkl')
        row_fixed = df_fixed.iloc[0]
        row_balanced = df_balanced.iloc[0]

        for metric in metrics:
            fixed_results[metric].append(row_fixed[metric])
            balanced_results[metric].append(row_balanced[metric])
        iterations.append(i)
    except Exception as e:
        print(f"Errore alla iterazione {i} (fixed/balanced): {e}")

    # Entropy 
    if i <= d["i_entropy"]:
        try:
            df_entropy = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy.pkl')
            row_entropy = df_entropy.iloc[0]
            for metric in metrics:
                entropy_results[metric].append(row_entropy[metric])
            entropy_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (entropy): {e}")

    # Random (fino a 28)
    if i <= d["i_random"]:
        try:
            df_random = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_random.pkl')
            row_random = df_random.iloc[0]
            for metric in metrics:
                random_results[metric].append(row_random[metric])
            random_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (random): {e}")

    # Entropy Balanced 
    if i <= d["i_entropy_balanced"]:
        try:
            df_entropy_bal = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_balanced.pkl')
            row_entropy_bal = df_entropy_bal.iloc[0]
            for metric in metrics:
                entropy_balanced_results[metric].append(row_entropy_bal[metric])
            entropy_balanced_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (entropy_balanced): {e}")
    
    # Entropy Ebu argmin
    if i <= d["i_entropry_EBU_argmin"]:
        try:
            df_entropy_bal = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_EBU_argmin.pkl')
            row_entropy_bal = df_entropy_bal.iloc[0]
            for metric in metrics:
                entropy_EBU_argmin_results[metric].append(row_entropy_bal[metric])
            entropy_EBU_argmin_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (entropy_balanced): {e}")
    
    # Entropy Ebu argmax
    if i <= d["i_entropy_EBU_argmax"]:
        try:
            df_entropy_bal = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy_EBU_argmax.pkl')
            row_entropy_bal = df_entropy_bal.iloc[0]
            for metric in metrics:
                entropy_EBU_argmax_results[metric].append(row_entropy_bal[metric])
            entropy_EBU_argmax_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (entropy_balanced): {e}")
    
# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    # Fixed
    #ax.plot(iterations, fixed_results[metric], marker='o', color='blue', label='Fixed Sampling')

    # Balanced
    ax.plot(iterations, balanced_results[metric], marker='s', color='green', label='Balanced Sampling')

    # Entropy
    ax.plot(entropy_iterations, entropy_results[metric], marker='^', color='orange', label='Entropy Sampling')

    # Random
    #ax.plot(random_iterations, random_results[metric], marker='D', color='purple', label='Random Sampling')

    # Entropy Balanced
    ax.plot(entropy_balanced_iterations, entropy_balanced_results[metric], marker='v', color='brown', label='Entropy Balanced Sampling')

    # Entropy Ebu argmin
    ax.plot(entropy_EBU_argmin_iterations, entropy_EBU_argmin_results[metric], marker='x', color='cyan', label='Entropy + EBU (argmin)')

    # Entropy per Ebu argmax
    ax.plot(entropy_EBU_argmax_iterations, entropy_EBU_argmax_results[metric], marker='P', color='magenta', label='Entropy + EBU (argmax)')


    # Full Training point 
    ax.scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')

    ax.set_title(metric)
    ax.set_xlabel('Iterazione')
    ax.set_ylabel(metric)
    ax.grid(True)

    if idx == 0:
        ax.legend()

# Rimuove subplot vuoti
if len(metrics) < len(axes):
    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle('Confronto: Fixed, Balanced, Entropy, Random, Entropy Balanced (con Full Training)', fontsize=16, y=1.02)
plt.savefig("active_learning_confronto_tutte_le_strategie.png", bbox_inches='tight')










#---------------------------------------------------------------------
#   Plot delle stats sull'entropia
#   Grafico 1: plot di min,max,avg entropy per train_size (iterazione)
#   Grafico 2: plot della avg entropy per classe per train_size
#  Nota sul grafico 2: ad un certo punti alcune classi possono terminare i sample nel pool
#                      andando a non risultare nel grafico
#---------------------------------------------------------------------
# carico da file 
file_path = "entropy_analysis.txt"
records = []

with open(file_path, "r") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[WARNING] Riga saltata per errore JSON: {e}")
            continue

# trasformo in df per plotting semplice ===
general_stats = []
class_entropy = {}

for record in records:
    train_size = record["train_size"]
    pool_size = record["pool_size"]
    entropy = record["entropy"]
    class_means = record["class_entropy_means"]

    #dataframe che avrà solo le stats aggregate
    general_stats.append({
        "train_size": train_size,
        "pool_size": pool_size,
        "entropy_min": entropy["min"],
        "entropy_mean": entropy["mean"],
        "entropy_max": entropy["max"]
    })

    #Dizionario in cui per ogni classe (se non presente creo la chiave) e associo iterativamente l'entropia
    #   { class_id : [ (train_size_i , entropy_i ) , ... ]}
    for cls, ent in class_means.items():
        cls = int(cls)
        if cls not in class_entropy:
            class_entropy[cls] = []
        class_entropy[cls].append((train_size, ent))

#print("Dizionare delle entropie per classi:\n")
#print(class_entropy)
df_general = pd.DataFrame(general_stats)

# Plot 1:come definito sopra ===
plt.figure(figsize=(10, 6))
plt.plot(df_general["train_size"], df_general["entropy_min"], label="Min Entropy")
plt.plot(df_general["train_size"], df_general["entropy_mean"], label="Mean Entropy")
plt.plot(df_general["train_size"], df_general["entropy_max"], label="Max Entropy")
plt.xlabel("Train Size")
plt.ylabel("Entropy")
plt.title("Entropia (min, mean, max) vs Train Size")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_entropy_overall.png")

# Plot2 entropia media per classe 

plt.figure(figsize=(28, 18))

#  colormap  per avere colori distinti
num_classes = len(class_entropy)
cmap = plt.colormaps.get_cmap('tab20').resampled(num_classes)   # per dare un colore diverso ognuno

#   { class_id : [ (train_size_i , entropy_i ) , ... ]}
for idx, (cls_id, values) in enumerate(sorted(class_entropy.items())):
    values = sorted(values, key=lambda x: x[0]) # Ordina per train_size (teoricamente non necessario)
    x = [v[0] for v in values]                  #le x sono i train_size
    y = [v[1] for v in values]                  # le y sono le entropie
    color = cmap(idx)  # Colore unico per ogni classe
    plt.plot(x, y, label=f"Class {cls_id}", color=color)

plt.xlabel("Train Size")
plt.ylabel("Entropy (Class-wise Mean)")
plt.title("Entropia media per classe vs Train Size")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_entropy_per_class.png")
