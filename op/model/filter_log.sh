#!/bin/bash

filter() {
  echo "$1" \
  | grep -v "is deprecated" \
  | grep -v "non-GPU devices" \
  | grep -v "num_readers has been reduced" \
  | grep -v "tf eager tensor" \
  | grep -v "tf.numpy_function" \
  | grep -v "tf.py_function" \
  | grep -v "Create a \`tf." \
  | grep -v "Instructions for updating" \
  | grep -v "Maybe overwriting" \
  | grep -v "Number of filenames to read" \
  | grep -v "Reading unweighted datasets" \
  | grep -v "Sets are not currently considered sequences" \
  | grep -v "Simply pass a True/False value to the \`training\` argument" \
  | grep -v "Skipping full serialization of Keras layer" \
  | grep -v "Unresolved object in checkpoint" \
  | grep -v "Unsupported signature for serialization" \
  | grep -v "Use \`tf." \
  | grep -v "Use fn_output_signature instead" \
  | grep -v "Using MirroredStrategy with devices" \
  | cat
}

if [ -p /dev/stdin ]; then
  # input from stdin
  while read -r LINE; do
    filter "$LINE"
  done
elif [ -f "$1" ]; then
  # input from file
  LINES=$(cat "$1")
  filter "$LINES"
fi
