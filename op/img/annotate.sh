#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# annotate images via labelImg
labelImg "$DGD_DATA/input"

# click "Open Dir" to select image directory

# click "Change Save Dir" to select xml output directory
