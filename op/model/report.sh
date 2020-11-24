#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# tensorboard
tensorboard --logdir="$DGD_NET_TRAIN"
