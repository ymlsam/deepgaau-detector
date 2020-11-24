#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
mkdir -p "$DGD_NET_EXPORT"

# export fine-tuned model
python "src/model_export.py" \
  --config_path="$DGD_NET_CONF" \
  --trained_ckpt_dir "$DGD_NET_TRAIN" \
  --output_directory "$DGD_NET_EXPORT" \
  2>&1 | op/model/filter_log.sh
