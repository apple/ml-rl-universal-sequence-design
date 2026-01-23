#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -x

python univ_seq_pkg/univ_seq/main.py \
  --debug \
  --N_set '[128]' \
  --genie_aided True \
  --approx_capacity_gap True \
  --K_min 16 \
  --auto_adjust False \
  --iterative_learning True \
  --lookahead_window 16 \
  --min_initial_training 50 \
  --settle_threshold 1 \
  --freeze_set 2 \
  --n_steps 64 \
  --eval_freq 512 \
  --use_async_vec_env 1 \
#  --load_model True \
#  --model_path "init_model/d6fbnstvgf_best_model.zip" \

exit

