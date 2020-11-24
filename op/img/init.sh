#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# create data directory
cd "$DGD" || exit
mkdir -p "$DGD_DIR_DATA/raw"

# import sample images
cd "$DGD_DIR_DATA/raw" || exit
cp "$DGD_DIR_TF/models/research/object_detection/test_images/ducky/train/"* .
cp "$DGD_DIR_TF/models/research/object_detection/test_images/ducky/test/"* .
