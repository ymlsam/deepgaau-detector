#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# list trainable variable names
python "src/model_var.py" \
  --config_path="$DGD_NET_CONF" \
  2>&1 | op/model/filter_log.sh
