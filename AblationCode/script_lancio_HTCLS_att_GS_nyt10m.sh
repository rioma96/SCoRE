#!/bin/bash

#SBATCH -A IscrC_CB-KGE
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:4        # 4 gpus per node out of 4
#SBATCH --mem=96000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=GSSLNYT10D

 

cd $HOME/CBKGE1/CBKGE/GSCodesForAttentionSentence

source $HOME/CBKGE1/bin/activate

./Train_NYT10M.sh

deactivate
