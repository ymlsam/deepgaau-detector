#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# annotate images via labelImg
labelImg "$DGD_DIR_DATA/raw"
