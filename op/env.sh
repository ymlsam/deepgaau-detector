#!/bin/bash

# go to project root
cd "$DGD" || exit

# export env var

# library
export DGD_TF="lib/tensorflow"

# data
export DGD_DATA_NAME="ducky"
export DGD_DATA="data/$DGD_DATA_NAME"
export DGD_DATA_INPUT="$DGD_DATA/input"
export DGD_DATA_LABEL="$DGD_DATA/label.pbtxt"

# network
export DGD_NET="model/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
#export DGD_NET="model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
export DGD_NET_IMPORT="$DGD_NET/import"
export DGD_NET_EXPORT="$DGD_NET/$DGD_DATA_NAME/export"
export DGD_NET_TUNE=""
export DGD_NET_TRAIN="$DGD_NET/$DGD_DATA_NAME$DGD_NET_TUNE/training"
export DGD_NET_CONF="$DGD_NET/$DGD_DATA_NAME$DGD_NET_TUNE/pipeline.config"

# activate python virtual env
. op/venv/activate.sh
