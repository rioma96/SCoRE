#!/bin/bash 
export PYTHONUNBUFFERED=1

dataset_paths="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/Wiki20m/Train/,/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/Wiki20m/Val/"
output_path="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/GSCodesForAttentionSentence/GSResults/res_att_Wiki20m.csv"


python3 SENT_att_GS.py  --dataset_paths $dataset_paths --output_path $output_path


