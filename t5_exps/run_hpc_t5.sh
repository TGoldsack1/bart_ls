#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account=dcs-res
#SBATCH --partition=dcs-gpu
#SBATCH --mem=128GB
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk


module load Anaconda3/5.3.0
module load CUDA/10.2.89-GCC-8.3.0

source activate huggingface

python ./run_t5.py