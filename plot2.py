import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
#       Crea i subplot per ogni metodo SEPARATAMENTE
#


# === PARAMETRI ===
methods = ["random_sampling","balanced_random","entropy","balanced_entropy","entropy_EBU_argmin","entropy_EBU_argmax"]
seeds = [42, 71, 88]
metrics = ['micro', 'Macro', 'm@100', 'M@100', 'm@200', 'M@200', 'm@300', 'M@300', 'm@1000', 'M@1000', 'P@R']

# === Carica performance su tutto il training set ===
full_df = pd.read_pickle('nyt10m_results_performance_0.pkl')
full_row = full_df.iloc[0]

for method in methods:
    print(f"Processing method: {method}")

    results_dir = f'performance_{method}'
    all_runs = []

    for seed in seeds:
        path = os.path.join(results_dir, f'performance_seed_{seed}.pkl')
        with open(path, 'rb') as f:
            run_results = pickle.load(f)  # List of DataFrames per iterazione
            all_runs.append(run_results)

    # Verifica consistenza iterazioni
    n_iters = len(all_runs[0])
    assert all(len(run) == n_iters for run in all_runs), "Mismatch nel numero di iterazioni"

    # Calcola media per ogni iterazione
    mean_dfs = []
    for iter_idx in range(n_iters):
        dfs_at_iter = [run[iter_idx] for run in all_runs]

        #Crea uno "stack" di dataframe (ovvero i 3 df per i 3 seed) e ci crea i dati aggregati (medie) per colonne
        stacked = np.stack([df.values for df in dfs_at_iter], axis=0)
        mean_df = pd.DataFrame(np.mean(stacked, axis=0), columns=dfs_at_iter[0].columns, index=dfs_at_iter[0].index)
        mean_dfs.append(mean_df)

    # === COSTRUISCI DIZIONARIO PER PLOTTARE ===
    plot_results = {metric: [] for metric in metrics}
    iterations = list(range(1, n_iters + 1))

    #ciclo il dizionario che ha calcolato la media tra i 3 df in modo da salvarmi i risultati in modo opportuno per plottarli
    for df in mean_dfs:
        row = df.iloc[0]
        for metric in metrics:
            plot_results[metric].append(row[metric])

    # === PLOTTING ===
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        axes[idx].plot(iterations, plot_results[metric], marker='o', label=f"{method} (mean)")
        axes[idx].scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')
        axes[idx].set_title(metric)
        axes[idx].set_xlabel('Iterazione')
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True)
        axes[idx].legend()

    # Rimuove subplot vuoti
    if len(metrics) < len(axes):
        for j in range(len(metrics), len(axes)):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plt.suptitle(f'Metriche Active Learning - {method} (media su seed {seeds})', fontsize=16, y=1.02)
    plt.savefig(f'active_learning_metrics_{method}.png', bbox_inches='tight')
    



#
#   Creazione dei plot "aggregati"
#



# === PARAMETRI ===

seeds = [42, 71, 88]
metrics = ['micro', 'Macro']

# === Carica performance full training
full_df = pd.read_pickle('nyt10m_results_performance_0.pkl')
full_row = full_df.iloc[0]

# === Colori e marker per ogni metodo
style_dict = {
    "random_sampling": ('purple', 'D', 'Random Sampling'),
    "balanced_random": ('green', 's', 'Balanced Random Sampling'),
    "entropy": ('orange', '^', 'Entropy Sampling'),
    "balanced_entropy": ('brown', 'v', 'Entropy Balanced Sampling'),
    "entropy_EBU_argmin": ('cyan', 'x', 'Entropy + EBU (argmin)'),
    "entropy_EBU_argmax": ('magenta', 'P', 'Entropy + EBU (argmax)')
}

# === Dizionario per raccogliere i risultati medi
all_results = {method: {} for method in methods}
iterations = None  # verrà inizializzato con range corretto

#si costruisce i dati (sfrutta la strategia utilizzata nel plotting singolo)
for method in methods:          #variabile salvata all'inizio
    print(f"[INFO] Caricamento {method}")
    results_dir = f'performance_{method}'
    all_runs = []

    for seed in seeds:
        path = os.path.join(results_dir, f'performance_seed_{seed}.pkl')
        with open(path, 'rb') as f:
            run_results = pickle.load(f)
            all_runs.append(run_results)

    n_iters = len(all_runs[0])
    assert all(len(run) == n_iters for run in all_runs), "Mismatch nel numero di iterazioni"

    mean_dfs = []
    for iter_idx in range(n_iters):
        dfs_at_iter = [run[iter_idx] for run in all_runs]
        stacked = np.stack([df.values for df in dfs_at_iter], axis=0)
        mean_df = pd.DataFrame(np.mean(stacked, axis=0), columns=dfs_at_iter[0].columns, index=dfs_at_iter[0].index)
        mean_dfs.append(mean_df)

    # Costruisce dizionario per il metodo
    for metric in metrics:
        all_results[method][metric] = [df.iloc[0][metric] for df in mean_dfs]

    if iterations is None:
        iterations = list(range(1, n_iters + 1))

# === PLOT: 2 metriche, tutti i metodi
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    for method in methods:
        color, marker, label = style_dict[method]
        ax.plot(iterations, all_results[method][metric], marker=marker, color=color, label=label)

    # Punto finale: performance full training
    ax.scatter(iterations[-1], full_row[metric], color='red', s=80, label='Full Training')

    ax.set_title(metric)
    ax.set_xlabel('Iterazione')
    ax.set_ylabel(metric)
    ax.grid(True)

    if idx == 0:
        ax.legend()

# Sistemazione layout
plt.tight_layout()
plt.suptitle('Confronto Strategie - Metriche micro e Macro', fontsize=16, y=1.02)
plt.savefig("active_learning_confronto_strategie.png", bbox_inches='tight')





#
# PLOT di analisi delle entropie (facendo la media tra i vari seed) [estensione che "plot.py"]
#

from collections import defaultdict
import numpy as np
import json

'''   [non testata servono i 3 file]
# === Configurazione
seeds = [42, 71, 88]
file_template = "entropy_analysis_{}.txt"

# === Lettura e parsing di tutti i file
all_general_stats = []
all_class_entropy = []

for seed in seeds:
    file_path = file_template.format(seed)
    records = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Riga saltata nel file {file_path} per errore JSON: {e}")
                continue

    general_stats = []
    class_entropy = defaultdict(list)  # {class_id: [(train_size, entropy)]}

    for record in records:
        train_size = record["train_size"]
        entropy = record["entropy"]
        class_means = record["class_entropy_means"]

        general_stats.append({
            "train_size": train_size,
            "entropy_min": entropy["min"],
            "entropy_mean": entropy["mean"],
            "entropy_max": entropy["max"]
        })

        for cls, ent in class_means.items():
            cls = int(cls)
            class_entropy[cls].append((train_size, ent))

    all_general_stats.append(pd.DataFrame(general_stats))
    all_class_entropy.append(class_entropy)

# === PLOT 1: Entropia min, mean, max vs Train Size (con std)
df_concat = pd.concat(all_general_stats, keys=range(len(seeds)), names=["seed", "index"])
grouped = df_concat.groupby("train_size")

mean_df = grouped.mean()
std_df = grouped.std()

plt.figure(figsize=(10, 6))
for stat in ["entropy_min", "entropy_mean", "entropy_max"]:
    plt.plot(mean_df.index, mean_df[stat], label=f"{stat} (mean)")
    plt.fill_between(mean_df.index,
                     mean_df[stat] - std_df[stat],
                     mean_df[stat] + std_df[stat],
                     alpha=0.2)

plt.xlabel("Train Size")
plt.ylabel("Entropy")
plt.title("Entropy min/mean/max vs Train Size (mean ± std over seeds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_entropy_overall_mean_std.png")
plt.close()

# === PLOT 2: Entropia media per classe (aggregata sui seed)
# Combina le class entropies di tutti i seed
combined_class_entropy = defaultdict(lambda: defaultdict(list))
# => {class_id: {train_size: [ent1, ent2, ent3]}}

for class_dict in all_class_entropy:
    for cls_id, values in class_dict.items():
        for train_size, ent in values:
            combined_class_entropy[cls_id][train_size].append(ent)

# Plot
plt.figure(figsize=(28, 18))
cmap = plt.colormaps.get_cmap('tab20').resampled(len(combined_class_entropy))

for idx, (cls_id, train_entropies) in enumerate(sorted(combined_class_entropy.items())):
    x_vals = sorted(train_entropies.keys())
    y_mean = [np.mean(train_entropies[x]) for x in x_vals]
    y_std = [np.std(train_entropies[x]) for x in x_vals]
    color = cmap(idx)

    plt.plot(x_vals, y_mean, label=f"Class {cls_id}", color=color)
    plt.fill_between(x_vals, np.array(y_mean) - y_std, np.array(y_mean) + y_std, color=color, alpha=0.2)

plt.xlabel("Train Size")
plt.ylabel("Entropy (Class-wise Mean)")
plt.title("Entropia media per classe vs Train Size (mean ± std over seeds)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_entropy_per_class_mean_std.png")
plt.close()
'''