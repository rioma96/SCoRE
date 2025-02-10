#!/bin/bash
export PYTHONUNBUFFERED=1
dataset="NYT10m"
echo $dataset
python GenDataOptimized.py --dataset $dataset 