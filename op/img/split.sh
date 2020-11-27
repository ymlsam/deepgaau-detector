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
python src/helper/split_dataset.py -i "$DGD_DATA/input" -o "$DGD_DATA"

# for illustration of eager few-shot learning, you may manually clean up the "train" folder after running above script, and copy "ducky.0" to "ducky.4" (both jpg & xml files) from "raw" folder to "train" folder
