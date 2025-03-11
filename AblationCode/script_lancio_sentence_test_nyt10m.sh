#!/bin/bash

#SBATCH -A IscrC_CB-KGE
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=194000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=Con_Pare_NYT10m

 

cd /leonardo_work/IscrC_CB-KGE/CBKGE1/CBKGE/GSCodesForAttentionSentence

source /leonardo_work/IscrC_CB-KGE/CBKGE1/bin/activate

module load cuda

./Test_NYT10M.sh

deactivate
