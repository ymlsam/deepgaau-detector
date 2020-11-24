#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# split source dataset into train/dev/test datasets
python src/helper/split_dataset.py -i "$DGD_DIR_DATA/raw" -o "$DGD_DIR_DATA"
