#!/bin/bash

# Nome del dataset
DATASET="nyt10m"

# Esegui lo script Python per generare il dataset specificato
python3 GenDataOptimized.py --dataset $DATASET