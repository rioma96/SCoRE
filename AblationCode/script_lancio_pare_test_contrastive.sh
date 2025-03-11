#!/bin/bash 

#SBATCH -A  ISG             # account name 

#SBATCH -p  high            # partition name 

#SBATCH --time 00:30:00    # format: HH:MM:SS 

#SBATCH -N 1                     # 1 node 

#SBATCH --ntasks-per-node=1      # 52 tasks out of 52 

#SBATCH --cpus-per-task=12 

#SBATCH --mem=64g                    # memory per node out of 495 Gb cnode or 1000 fnode 

#SBATCH --job-name=Grid_search_Att_nyt10d
 

cd $HOME/CBKGE/GSCodesForAttentionSentence/

source $HOME/CBKGE/bin/activate

python3 SENT_att_CL_test.py

deactivate
