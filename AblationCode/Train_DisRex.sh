#!/bin/bash 
export PYTHONUNBUFFERED=1

dataset_paths="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/DisRex/Train/,/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/DatasetsProcessed/DisRex/Val/"
output_path="/leonardo/home/userexternal/lmariot1/CBKGE1/CBKGE/GSCodesForAttentionSentence/GSResults/res_att_DisRex.csv"


python3 SENT_att_GS.py  --dataset_paths $dataset_paths --output_path $output_path
