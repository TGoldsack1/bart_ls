#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="PLOS"
# DATA_BIN="/home/acp20tg/bart_ls/resources/${DATASET}_fs-bin"
# DATA_BIN="/home/acp20tg/bart_ls/resources/${DATSET}_fs-graph_text-bin"
# DATA_BIN="/home/acp20tg/bart_ls/resources/${DATSET}_fs-controllable_all-bin"
# DATA_BIN="/home/acp20tg/bart_ls/resources/${DATSET}_fs-graph_text-bin"

DATA_BIN="/root/bart_ls/resources/${DATASET}_fs-bin"


# TRY ADDDING --save-dir checkpoints/...
# OUT_FILE="checkpoints/${DATASET}/controllable/all"
# OUT_FILE="checkpoints/${DATASET}/graph_text"
<<<<<<< HEAD
#OUT_FILE="/fastdata/acp20tg/bart_ls/checkpoints/${DATASET}/dual_encode"

OUT_FILE="/root/autodl-tmp/checkpoints/${DATASET}/dual_encode/50"
=======
OUT_FILE="/fastdata/acp20tg/bart_ls/checkpoints/${DATASET}/decoder_attend"
>>>>>>> 752d23686cc478e1dc18d38fa708d42cb0b05cb2

RESTORE_CHECK="${OUT_FILE}/checkpoint_last.pt" #  ../checkpoints/model_100k.pt
# RESTORE_CHECK="../checkpoints/model_100k.pt" 
# TOKENIZERS_PARALLELISM=false

<<<<<<< HEAD
# BATCH SIZE: # 2 for eLife, 4 for PLOS
=======
#   --dual_graph_encoder \

>>>>>>> 752d23686cc478e1dc18d38fa708d42cb0b05cb2

TOKENIZERS_PARALLELISM=true CUDA_LAUNCH_BLOCKING=1 python train.py $DATA_BIN \
  --task summarization \
  --max-epoch 20 \
  --arch "bart_large" \
  --use-xformers \
  --attention-name block_noglobal \
  --pooling-layers 4 \
  --fast-stat-sync \
  --criterion label_smoothed_cross_entropy \
  --truncate-source \
  --truncate-target \
  --source-lang src \
  --target-lang tgt \
  --required-seq-len-multiple 1024 \
  --max-source-positions 16384 \
  --max-target-positions 1024 \
  --update-freq 4 \
  --batch-size 4 \
  --optimizer "adam" \
  --adam-betas "(0.9, 0.98)" \
  --clip-norm 0.1 \
  --lr 1e-4 \
  --checkpoint-activations \
  --lr-scheduler "polynomial_decay" \
  --warmup-updates 500 \
  --eval-rouge \
  --eval-rouge-args '{"beam": 4, "max_len_b": 700, "lenpen": 2.0, "no_repeat_ngram_size": 3, "min_len": 20}' \
  --seed=3 \
  --memory-efficient-fp16 \
  --total-num-update 10000 \
  --num-workers 4 \
  --skip-invalid-size-inputs-valid-test \
  --combine-val \
  --no-epoch-checkpoints \
  --maximize-best-checkpoint-metric \
  --best-checkpoint-metric rouge_avg \
  --log-format json \
  --log-interval 10 \
  --custom-dict ../checkpoints/dict.txt \
  --restore-file ../checkpoints/model_100k.pt \
  --decoder_graph_attn \
  --save-dir $OUT_FILE
