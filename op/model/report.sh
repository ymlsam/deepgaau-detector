#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# tensorboard
tensorboard --logdir="$DGD_DIR_NET_TRAIN"
