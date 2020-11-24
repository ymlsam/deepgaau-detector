#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# clean up train/dev/test directories
echo "cleaning up train/dev/test directories"
rm "$DGD_DATA/train/"*
rm "$DGD_DATA/dev/"*
rm "$DGD_DATA/test/"*

# split source dataset into train/dev/test datasets
echo "splitting into train/dev/test datasets"
python src/helper/split_dataset.py -i "$DGD_DATA/raw" -o "$DGD_DATA"
