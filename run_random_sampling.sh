#!/bin/bash

METODO="random_sampling"
SEEDS="42 71 88"

python3 script_active.py --method $METODO --seeds $SEEDS
