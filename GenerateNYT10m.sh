#!/bin/bash
export PYTHONUNBUFFERED=1
dataset="nyt10m"
echo $dataset
python GenDataOptimized_opt.py --dataset $dataset 