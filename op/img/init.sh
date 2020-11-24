#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# create data directory
cd "$DGD" || exit
mkdir -p "$DGD_DATA/raw"
mkdir -p "$DGD_DATA/train"
mkdir -p "$DGD_DATA/dev"
mkdir -p "$DGD_DATA/test"
mkdir -p "$DGD_DATA/detect"

# import sample images (ducky images are included already)
#cd "$DGD_DATA/raw" || exit
#cp "$DGD_TF/models/research/object_detection/test_images/ducky/train/"* .
#cp "$DGD_TF/models/research/object_detection/test_images/ducky/test/"* .
