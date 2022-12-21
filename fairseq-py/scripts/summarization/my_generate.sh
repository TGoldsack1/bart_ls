#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATSET="eLife"
MODEL="graph_text"

DATA_BIN="/home/acp20tg/bart_ls/resources/${DATSET}_fs-bin"
CHECKPOINT_PATH="/home/acp20tg/bart_ls/fairseq-py/checkpoints/${DATSET}/${MODEL}/checkpoint_best.pt"
SUMMARY_SAVE_DIR="/home/acp20tg/bart_ls/results/${DATSET}/${MODEL}/results.txt"

python scripts/summarization/long_generate.py \
            --model-dir ${CHECKPOINT_PATH} \
            --data-dir ${DATA_BIN} \
            --save-dir ${SUMMARY_SAVE_DIR} \
            --split test \
            --bsz 4
