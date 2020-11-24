#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
mkdir -p "$DGD_DIR_NET_TRAIN"

# train
python "src/model_train_eval.py" \
  --pipeline_config_path="$DGD_CONF_NET" \
  --num_train_steps=100 \
  --model_dir="$DGD_DIR_NET_TRAIN" \
  --checkpoint_dir="$DGD_DIR_NET_TRAIN" \
  --sample_1_of_n_eval_examples=1 \
  2>&1 | op/model/filter_log.sh
