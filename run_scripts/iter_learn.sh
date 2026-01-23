#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -x

# N2048
python univ_seq_pkg/univ_seq/main.py \
  --debug \
  --N_set '[2048]' \
  --genie_aided True \
  --approx_capacity_gap True \
  --K_min 16 \
  --K_min_embed 36 \
  --auto_adjust False \
  --iterative_learning True \
  --lookahead_window 16 \
  --min_initial_training 50 \
  --settle_threshold 1 \
  --freeze_set 2 \
  --n_steps 64 \
  --eval_freq 512 \
  --use_async_vec_env 1 \
  --learned_seq_n1024_path "init_model/3di7d8f5fv_learned_seq.mat" \
  --monitor_lower_n False \
  --embed_lower_n True \
  --promote_ln_node True \
#  --load_model True \
#  --model_path "init_model/g5e3y8w345_best_model.zip" \

exit
