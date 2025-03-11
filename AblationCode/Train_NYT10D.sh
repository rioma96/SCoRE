#!/bin/bash 
export PYTHONUNBUFFERED=1

dataset_paths="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/NYT10DWOmarkers/Train/,/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/NYT10DWOmarkers/Train/"
output_path="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/GSCodesForAttentionSentence/GSResults/res_att_NYT10d.csv"


python3 SENT_att_GS.py  --dataset_paths $dataset_paths --output_path $output_path
