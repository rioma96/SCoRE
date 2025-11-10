import copy
import tensorflow as tf
from CBKGE.NN_creation_and_dependencies import *
from CBKGE.NN_preproc import *
from CBKGE.utilities_validation import *
import glob
import numpy as np
import itertools
import pandas as pd
import os

def greedy_search_on_architectures(base_configuration, dataset, dict_dataset, param_grid, results_file="risultati_greedy.csv"):
    architectures = ['COMAL_SHARED', 'COMAL_NO_SHARED']
    keys, values = zip(*param_grid.items())

    for arch in architectures:
        results_file_arch = f"risultati_greedy_{arch}_{dataset}.csv"
        # Carica risultati già presenti, se esistono
        if os.path.exists(results_file_arch):
            df_existing = pd.read_csv(results_file_arch)
            df_existing = df_existing.drop_duplicates()
            # Assicurati che tutte le colonne della grid siano presenti
            for k in keys:
                if k not in df_existing.columns:
                    df_existing[k] = np.nan
            existing_configs = df_existing[list(keys)].to_dict(orient='records')
        else:
            df_existing = pd.DataFrame()
            existing_configs = []

        # Carica i dati UNA SOLA VOLTA per architettura/dataset
        multiple_mlp = arch in ['COMAL_SHARED', 'COMAL_NO_SHARED']
        path_train = dict_dataset[dataset]['train']
        path_test = dict_dataset[dataset]['test']
        path_val = dict_dataset[dataset]['val']
        validation = path_val is not None

        file_list_train = glob.glob(path_train + "*.pkl")
        file_list_test = glob.glob(path_test + "*.pkl")
        if validation:
            file_list_val = glob.glob(path_val + "*.pkl")

        # Caricamento dati
        class_probabilities, link_dict, training_dataset, train_df = from_file_list_to_tfdataset(
            file_list_train,
            balance_triples=base_configuration['balance_triples'],
            quantile_cutoff=base_configuration['quantile_cutoff_balancing'],
            balance_labels=base_configuration['balance_labels'],
            cls=base_configuration['cls'],
            return_df=True,
            get_thresholds=True,
            multiple_mlp=multiple_mlp
        )
        _, test_dataset, test_df = from_file_list_to_tfdataset(
            file_list_test,
            relation_dict=link_dict,
            return_df=True,
            return_headtail=base_configuration['return_headtail'],
            cls=base_configuration['cls'],
            multiple_mlp=multiple_mlp
        )
        if validation:
            _, validation_dataset, val_df = from_file_list_to_tfdataset(
                file_list_val,
                relation_dict=link_dict,
                balance_triples=False,
                quantile_cutoff=base_configuration['quantile_cutoff_balancing'],
                balance_labels=False,
                cls=base_configuration['cls'],
                return_df=True,
                multiple_mlp=multiple_mlp
            )


        buffer_size = len(training_dataset)
        training_dataset = training_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        test_dataset = test_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
        if validation:
            validation_dataset = validation_dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        # Poi, dentro il ciclo sulle configurazioni:
        for param_values in itertools.product(*values):
            param_dict = dict(zip(keys, param_values))
            # Salta se già presente
            if param_dict in existing_configs:
                print(f"Configurazione già presente, salto: {param_dict}")
                continue

            print(f"\n--- Test configurazione: {param_dict} ---")
            config = copy.deepcopy(base_configuration)
            config.update(param_dict)
            config['architecture'] = arch
            config['class_probabilities'] = class_probabilities
            config['n_branches'] = len(link_dict)
            config['val_batch'] = len(training_dataset)


            # Crea il modello
            tf.keras.backend.clear_session()
            model = CreateModel(training_configuration=config)

            # Allenamento
            if validation:
                callback = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=1e-4,
                    patience=2,
                    verbose=0,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True,
                    start_from_epoch=0
                )
                model.fit(
                    training_dataset.batch(config['batch_size']),
                    epochs=config['epochs'],
                    validation_data=validation_dataset.batch(config['batch_size']),
                    callbacks=[callback]
                )
            else:
                callback = tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    min_delta=1e-4,
                    patience=10,
                    verbose=0,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True,
                    start_from_epoch=0
                )
                model.fit(
                    training_dataset.batch(config['batch_size']),
                    epochs=config['epochs'],
                    callbacks=[callback]
                )

            # Valutazione per branch
            all_branch_preds = []
            all_branch_labels = []
            for branch_idx in range(config['n_branches']):
                model_branch = tf.keras.Model(
                    inputs=model.input,
                    outputs=model.outputs[branch_idx]
                )
                def extract_features_and_labels_for_branch(transformed_dataset, branch_idx):
                    X, y = [], []
                    for features, label_dict in transformed_dataset:
                        X.append(features.numpy())
                        y.append(label_dict[f'branch_{branch_idx}_norm'].numpy())
                    return np.array(X), np.array(y)
                if validation:
                    test_X, test_y = extract_features_and_labels_for_branch(validation_dataset, branch_idx)
                else:
                    test_X, test_y = extract_features_and_labels_for_branch(training_dataset, branch_idx)
                train_X, train_y = extract_features_and_labels_for_branch(training_dataset, branch_idx)
                test_ds_branch = tf.data.Dataset.from_tensor_slices((test_X, test_y))
                train_ds_branch = tf.data.Dataset.from_tensor_slices((train_X, train_y))

                branch_config = copy.deepcopy(config)
                branch_config['branch_idx'] = branch_idx

                test_y_true, test_y_pred, *_ = evaluation_and_performance(
                    test_configuration=branch_config,
                    training_dataset=train_ds_branch,
                    test_dataset=test_ds_branch,
                    model=model_branch
                )[:2]
                all_branch_preds.append(test_y_pred)
                all_branch_labels.append(test_y_true)
                del test_ds_branch, train_ds_branch, test_y_true, test_y_pred
                import gc; gc.collect()

            all_branch_preds_concat = np.concatenate(all_branch_preds, axis=1)
            all_branch_labels_concat = np.concatenate(all_branch_labels, axis=1)

            f1_micro = f1_micro_yt_ml(all_branch_labels_concat, all_branch_preds_concat)
            f1_macro = f1_macro_yt_ml(all_branch_labels_concat, all_branch_preds_concat)

            row = {**param_dict, "architecture": arch, "f1_micro": f1_micro, "f1_macro": f1_macro}
            print(f"Arch: {arch}, F1-micro: {f1_micro}, F1-macro: {f1_macro}")

            # Scrivi SOLO la nuova riga
            df_row = pd.DataFrame([row])
            if os.path.exists(results_file_arch):
                df_row.to_csv(results_file_arch, mode='a', index=False, header=False)
            else:
                df_row.to_csv(results_file_arch, index=False, header=True)
            # Aggiorna la lista delle configurazioni già presenti
            existing_configs.append(param_dict)

        # Alla fine, deduplica il file (opzionale ma consigliato)
        if os.path.exists(results_file_arch):
            df_final = pd.read_csv(results_file_arch)
            df_final = df_final.drop_duplicates(subset=list(keys) + ["architecture"])
            df_final.to_csv(results_file_arch, index=False)

        print(f"\nRisultati completi salvati in {results_file_arch}")
    return None

# Esempio di utilizzo:
if __name__ == "__main__":
    configuration = {
        'input_shape': 1536,
        'architecture': 'COMAL_SHARED',
        'distance': 'euclidean',
        'activation': 'swish',
        'learning_rate': 0.0005,
        'temperature': 0.01,
        'base_temperature': 1,
        'optimizer': tf.keras.optimizers.serialize(tf.keras.optimizers.AdamW()),
        'output_dimensions': 15,
        'depth': 2,
        'depth_encoder': 5,
        'pert': 0.1,
        'epochs': 10,
        'batch_size': 256,
        'similarity_to_class': 'kNN_chunks',
        'val_batch': 200000,
        'kNN': 50,
        'n_jobs': -1,
        'top_perf': 1000,
        'classic_kNN': True,
        'eval_method': 'multi',
        'threshold': 0.5,
        'loss': 'similarity_multiMLP',
        'jaccard_threshold': 0.3,
        'balance_triples': False,
        'quantile_cutoff_balancing': 0.95,
        'balance_labels': False,
        'bag_threshold': 0.,
        'get_thresholds': False,
        'return_headtail': False,
        'other_top_perf': [100, 200, 300],
        'calibrated': False,
        'harmonic_score': True,
        'cls': False,
        'q': None,
        'add_NA_class': True,
        'branch_idx': None
    }
    dict_dataset = {
        'nyt10d': {'train': 'Final/NYT10D/Train/', 'test': 'Final/NYT10D/Test/', 'val': None},
        'nyt10m': {'train': 'Final/NYT10m/Train/', 'test': 'Final/NYT10m/Test/', 'val': 'Final/NYT10m/Val/'},
        'wiki20m': {'train': 'Final/Wiki20m/Train/', 'test': 'Final/Wiki20m/Test/', 'val': 'Final/Wiki20m/Val/'},
        'disrex': {'train': 'Final/DisRex/Train/', 'test': 'Final/DisRex/Test/', 'val': 'Final/DisRex/Val/'},
        'wiki20distant': {'train': 'Final/Wiki20mDistant/Train/', 'test': 'Final/Wiki20m/Test/', 'val': 'Final/Wiki20mDistant/Val/'},
        "nyt10m_roberta": {'train': 'Datasets/NYT10m_roberta/Train/', 'test': 'Datasets/NYT10m_roberta/Test/', 'val': 'Datasets/NYT10m_roberta/Val/'},
        "nyt10d_roberta": {'train': 'Datasets/NYT10D_roberta/Train/', 'test': 'Datasets/NYT10D_roberta/Test/', 'val': None},
        "nyt10m_w2v": {'train': 'Datasets/NYT10m_w2v/Train/', 'test': 'Datasets/NYT10m_w2v/Test/', 'val': 'Datasets/NYT10m_w2v/Val/'},
        "wiki20m_w2v": {'train': 'Datasets/Wiki20m_w2v/Train/', 'test': 'Datasets/Wiki20m_w2v/Test/', 'val': 'Datasets/Wiki20m_w2v/Val/'},
        "disrex_w2v": {'train': 'Datasets/DisRex_w2v/Train/', 'test': 'Datasets/DisRex_w2v/Test/', 'val': 'Datasets/DisRex_w2v/Val/'},
        "nyt10d_w2v": {'train': 'Datasets/NYT10D_w2v/Train/', 'test': 'Datasets/NYT10D_w2v/Test/', 'val': None},
        "wiki20distant_w2v": {'train': 'Datasets/Wiki20Distant_w2v/Train/', 'test': 'Datasets/Wiki20m_w2v/Test/', 'val': 'Datasets/Wiki20Distant_w2v/Val/'}
    }
    datasets = ['disrex',"nyt10m",'nyt10d','wiki20m']  # Sostituisci con il dataset desiderato

    # Definisci la griglia dei parametri da esplorare
    param_grid = {
        "depth_encoder": [2, 3, 5],
        "depth": [2, 3, 5],
        "kNN": [5, 10, 50],
        "threshold": [0.4, 0.5, 0.6, 0.7, 0.8]
    }

    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        greedy_search_on_architectures(configuration, dataset, dict_dataset, param_grid, results_file="risultati_greedy.csv")