#!/bin/bash

# go to project root
cd "$DGD" || exit

# activate python virtual env (the leading dot causes the script to be executed in current shell instead of sub-shell)
. .venv/bin/activate
