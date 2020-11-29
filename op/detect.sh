#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
mkdir -p "$DGD_DATA/output"

# select checkpoint
DGD_NET_TRAIN_CKPT_NUM=$(find "$DGD_NET_TRAIN" -type f -name "ckpt-*.index" | cut -f 1 -d "." | rev | cut -f 1 -d "-" | rev | sort -n | tail -1)

if [ "$DGD_NET_TUNE" == "" ] || [ "$DGD_NET_TRAIN_CKPT_NUM" == "" ]; then
  DGD_NET_CKPT="$DGD_NET_EXPORT/checkpoint/ckpt-0"
else
  DGD_NET_CKPT="$DGD_NET_TRAIN/ckpt-$DGD_NET_TRAIN_CKPT_NUM"
fi

echo "checkpoint: $DGD_NET_CKPT"

# run main routine
python "src/detect.py" \
  --config_path="$DGD_NET_CONF" \
  --ckpt_path="$DGD_NET_CKPT" \
  --in_path="$DGD_DATA/test" \
  --out_dir="$DGD_DATA/output"
