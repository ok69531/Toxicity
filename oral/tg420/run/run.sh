#!/bin/bash
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --partition=gpu
##
#SBATCH --job-name=dt
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##

hostname
date

module add CUDA/11.3.0
module add ANACONDA/2020.11


/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/dt.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/gbt.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/lda.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/lgb.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/logistic.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/mlp.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/plsda.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/qda.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/rf.py
/home1/ok69531/anaconda3/envs/torch/bin/python /home1/ok69531/Toxicity/oral/tg420/run/xgb.py