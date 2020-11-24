#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# run main routine
python "src/detect.py" \
  --config_path="$DGD_NET_CONF" \
  --ckpt_path="$DGD_NET_EXPORT/checkpoint/ckpt-0" \
  --in_path="$DGD_DATA/test" \
  --out_dir="$DGD_DATA/detect"
