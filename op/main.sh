#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# run main routine
python src/main.py
