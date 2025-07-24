#!/bin/bash

METODO="balanced_entropy"
SEEDS="42 71 88"

python3 script_active.py --method $METODO --seeds $SEEDS
