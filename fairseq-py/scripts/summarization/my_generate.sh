#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

DATASET="PLOS"
MODEL="dual_encode"

DATA_BIN="/root/bart_ls/resources/${DATASET}_fs-bin"
CHECKPOINT_PATH="/root/autodl-tmp/checkpoints/${DATASET}/${MODEL}/50/checkpoint_best.pt"
SUMMARY_SAVE_DIR="/root/bart_ls/results/${DATASET}/${MODEL}/50/results.txt"


python scripts/summarization/long_generate.py \
            --model-dir ${CHECKPOINT_PATH} \
            --data-dir ${DATA_BIN} \
            --save-dir ${SUMMARY_SAVE_DIR} \
            --dual_graph_encoder \
            --split test \
            --bsz 4
