import pandas as pd

'''
print("ACTIVE WITH full train")

df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_1_with_entropy.pkl')
print(df)
print("\n\n\n")
'''

'''
print("ACTIVE WITH random sampling")
for i in range(1,5):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_random.pkl')
    print(df)
print("\n\n\n")
'''
print("ACTIVE WITH entropy")
for i in range(1,25):
    print(f"Iterazione:{i}")
    df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy2.pkl')
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


#             PLOT PER  Entropy2
#
# Inizializza dizionario per salvare i valori per ogni metrica
results = {metric: [] for metric in metrics}
iterations = []

# Carica i pickle e raccogli le metriche nel dizionario
for i in range(1, 30):
    try:
        df = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy2.pkl')
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

plt.savefig("active_learning_metrics_entropy2.png", bbox_inches='tight')









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

iterations = []
entropy_iterations = []
random_iterations = []
entropy_balanced_iterations = []  

# Caricamento dati
for i in range(1, 35):

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

    # Entropy (fino a 29)
    if i <= 29:
        try:
            df_entropy = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy2.pkl')
            row_entropy = df_entropy.iloc[0]
            for metric in metrics:
                entropy_results[metric].append(row_entropy[metric])
            entropy_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (entropy): {e}")

    # Random (fino a 28)
    if i <= 28:
        try:
            df_random = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_random.pkl')
            row_random = df_random.iloc[0]
            for metric in metrics:
                random_results[metric].append(row_random[metric])
            random_iterations.append(i)
        except Exception as e:
            print(f"Errore alla iterazione {i} (random): {e}")

    # Entropy Balanced (fino a 27)
    if i <= 27:
        try:
            df_entropy_bal = pd.read_pickle(f'active_performance/nyt10m_active_results_iteration_{i}_with_entropy2_balanced.pkl')
            row_entropy_bal = df_entropy_bal.iloc[0]
            for metric in metrics:
                entropy_balanced_results[metric].append(row_entropy_bal[metric])
            entropy_balanced_iterations.append(i)
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
    ax.plot(random_iterations, random_results[metric], marker='D', color='purple', label='Random Sampling')

    # Entropy Balanced
    ax.plot(entropy_balanced_iterations, entropy_balanced_results[metric], marker='v', color='brown', label='Entropy Balanced Sampling')

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
