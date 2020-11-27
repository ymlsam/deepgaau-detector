#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# create data directory
cd "$DGD" || exit
mkdir -p "$DGD_DATA/input"
mkdir -p "$DGD_DATA/train"
mkdir -p "$DGD_DATA/dev"
mkdir -p "$DGD_DATA/test"
mkdir -p "$DGD_DATA/output"

# import sample images (ducky images are included already)
#cd "$DGD_DATA/input" || exit
#cp "$DGD_TF/models/research/object_detection/test_images/ducky/train/"* .
#cp "$DGD_TF/models/research/object_detection/test_images/ducky/test/"* .
