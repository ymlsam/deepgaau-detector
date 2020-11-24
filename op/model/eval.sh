#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
mkdir -p "$DGD_NET_TRAIN"

# train
python "src/model_eval.py" \
  --config_path="$DGD_NET_CONF" \
  --model_dir="$DGD_NET_TRAIN" \
  --ckpt_dir="$DGD_NET_TRAIN" \
  2>&1 | op/model/filter_log.sh
