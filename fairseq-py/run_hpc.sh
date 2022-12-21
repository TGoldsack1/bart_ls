#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --partition=gpu-a100-tmp
#SBATCH --mem=128GB
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=tgoldsack1@sheffield.ac.uk
#SBATCH --mail-type=ALL

module unuse /usr/local/modulefiles/live/eb/all
module unuse /usr/local/modulefiles/live/noeb
module use /usr/local/modulefiles/staging/eb-znver3/all/

# module load cuDNN/8.2.2.26-CUDA-11.4.1
module load CUDA/11.4.1
module load GCCcore/9.5.0
module load GCC/9.5.0

# module load expat/2.4.1-GCCcore-11.2.0
# module load GCC/11.2.0
module load Anaconda3/2021.11

# conda init
# conda activate bart-ls
source activate bart-ls
# conda install cudnn=8.2.3

# cpan install XML::Parser::PerlSAX
# cpan install XML::DOM

# export ROUGE_EVAL_HOME=/home/acp20tg/bart_ls/fairseq-py/Yale-LILY-SummEval-e26bef9/evaluation/summ_eval/ROUGE-1.5.5/data

# export ROUGE_HOME=/home/acp20tg/.conda/envs/bart-ls/fairseq-py/Yale-LILY-SummEval-9b58833/evaluation/summ_eval/ROUGE-1.5.5/
# pip install -U  git+https://github.com/bheinzerling/pyrouge.git
# export ROUGE_HOME = "/home/acp20tg/bart_ls/fairseq-py/Yale-LILY-SummEval-e26bef9/evaluation/summ_eval/ROUGE-1.5.5/"

bash scripts/summarization/my_ft_summ.sh
