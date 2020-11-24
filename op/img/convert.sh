#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# convert train/dev/test datasets into TFRecord
python src/helper/record_from_xmls.py -l "$DGD_CONF_LABEL" -i "$DGD_DIR_DATA"
