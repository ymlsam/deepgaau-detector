#!/bin/bash

# go to project root
cd "$DGD" || exit

# export env var
export DGD_DIR_DATA="data/ducky"
export DGD_DIR_MODEL="model"
export DGD_DIR_TF="lib/tensorflow"

export DGD_NET="ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
#export DGD_NET="ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"

export DGD_DIR_NET="$DGD_DIR_MODEL/$DGD_NET"
export DGD_DIR_NET_IMPORT="$DGD_DIR_NET/import"
export DGD_DIR_NET_TRAIN="$DGD_DIR_NET/training"
export DGD_DIR_NET_EXPORT="$DGD_DIR_NET/export"

export DGD_CONF_LABEL="conf/data/ducky.pbtxt"
export DGD_CONF_NET="$DGD_DIR_NET/pipeline.config"

# activate python virtual env
. op/venv/activate.sh
