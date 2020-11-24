#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
mkdir -p "$DGD_DIR_NET_EXPORT"

# export fine-tuned model
python "src/model_export.py" \
  --pipeline_config_path="$DGD_CONF_NET" \
  --trained_checkpoint_dir "$DGD_DIR_NET_TRAIN" \
  --output_directory "$DGD_DIR_NET_EXPORT" \
  2>&1 | op/model/filter_log.sh
