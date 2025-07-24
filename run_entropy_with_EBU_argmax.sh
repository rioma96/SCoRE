#!/bin/bash

METODO="entropy_with_EBU_argmax"
SEEDS="42 71 88"

python3 script_active.py --method $METODO  --seeds $SEEDS --argmin False
