#!/bin/bash
#SBATCH --time=168:00:00
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --mem=128GB
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk
#SBATCH --mail-type=ALL

# module unuse /usr/local/modulefiles/live/eb/all
# module unuse /usr/local/modulefiles/live/noeb
# module use /usr/local/modulefiles/staging/eb-znver3/all/

# module load CUDA/11.4.1
# module load GCCcore/9.5.0
# module load GCC/9.5.0


#module load cuDNN/8.0.4.30-CUDA-11.1.1
module load CUDAcore/11.1.1
module load GCCcore/9.3.0
module load GCC/9.3.0

# module load Anaconda3/2021.11
module load Anaconda3/5.3.0 

source activate bart-ls

pip freeze

bash scripts/summarization/my_ft_summ_v100.sh
