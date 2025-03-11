l#!/bin/bash 
export PYTHONUNBUFFERED=1

dataset_paths="/leonardo_work/IscrC_CB-KGE/CBKGE1/CBKGE/DatasetsProcessed/NYT10m/Train/,/leonardo_work/IscrC_CB-KGE/CBKGE1/CBKGE/DatasetsProcessed/NYT10m/Val/,/leonardo_work/IscrC_CB-KGE/CBKGE1/CBKGE/DatasetsProcessed/NYT10m/Test/"
output_path="/leonardo_work/IscrC_CB-KGE/CBKGE1/CBKGE/GSCodesForAttentionSentence/PredictionResults/res_att_NYT10m.csv"
dataset_name="NYT10M"

python3 SENT_att_test.py  --dataset_paths $dataset_paths --output_path $output_path
#python3 Contrastive_Pare.py  --dataset_paths $dataset_paths --output_path $output_path --dataset_name $dataset_name
