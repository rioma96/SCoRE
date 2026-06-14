# SCoRE

Official repository of the paper **SCoRE: Simple Corpus-based Relation Extraction using Supervised Multi-Label Contrastive Learning**.

## Overview

SCoRE is a relation extraction pipeline that:
- builds sentence-level embeddings from raw corpora (`GenDataOptimized.py`),
- trains a supervised multi-label contrastive model (`Modello_Finale.py`),
- evaluates ranking and classification performance on common distant/partially supervised RE benchmarks.

Supported datasets in this repository:
- `nyt10d`
- `nyt10m`
- `wiki20m`
- `wiki20distant`
- `disrex`

## Repository Structure

- `GenDataOptimized.py`: preprocessing + embedding generation from raw datasets.
- `Modello_Finale.py`: model training, validation, and inference/evaluation.
- `CBKGE/`: neural network creation, preprocessing, and validation utilities.
- `AblationCode/`: scripts and code used for ablation experiments.
- `download_raw_datasets.sh`: downloads raw datasets from Google Drive.
- `download_processed_datasets.sh`: downloads preprocessed data ready for training.
- `generate_*.sh`: helper scripts to generate embeddings for each dataset.
- `train_and_infer_*.sh`: helper scripts to train/evaluate on each dataset.
- `main_example_ml_prediction.ipynb`: example notebook.

## Requirements

- Python 3.10+ recommended
- `pip`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

### Option A (recommended): Download preprocessed data

This is the fastest way to run experiments.

```bash
bash download_processed_datasets.sh
```

### Option B: Build embeddings from raw data

1. Download raw datasets:

```bash
bash download_raw_datasets.sh
```

2. Generate embeddings for the desired dataset:

```bash
bash generate_nyt10d.sh
bash generate_nyt10m.sh
bash generate_wiki20m.sh
bash generate_wiki20distant.sh
bash generate_disrex.sh
```

> `GenDataOptimized.py` writes processed chunks under `Datasets/...`.

## Train and Evaluate

Run one of the provided scripts:

```bash
bash train_and_infer_nyt10d.sh
bash train_and_infer_nyt10m.sh
bash train_and_infer_wiki20m.sh
bash train_and_infer_wiki20distant.sh
bash train_and_infer_disrex.sh
```

All these scripts call:

```bash
python3 Modello_Finale.py <dataset_name>
```

`Modello_Finale.py` currently expects input files under `Final/...` paths (configured in the dataset dictionary inside the script).

## Notes

- Training is repeated for multiple runs in `Modello_Finale.py`.
- Default hyperparameters and dataset-specific thresholds are defined directly in the script.
- For ablation workflows, see `AblationCode/`.

## License

This project is distributed under the terms of the `LICENSE` file in this repository.
