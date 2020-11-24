#!/bin/bash

filter() {
  echo "$1" \
  | grep -v "I1112" \
  | grep -v "W1112" \
  | grep -v "Use \`tf." \
  | grep -v "Create a \`tf." \
  | grep -v "tf.numpy_function" \
  | grep -v "tf.py_function" \
  | grep -v "tf eager tensor" \
  | grep -v "Simply pass a True/False value to the \`training\` argument" \
  | grep -v "Use fn_output_signature instead" \
  | grep -v "non-GPU devices" \
  | grep -v "Using MirroredStrategy with devices" \
  | grep -v "Maybe overwriting" \
  | grep -v "Reading unweighted datasets" \
  | grep -v "Number of filenames to read" \
  | grep -v "num_readers has been reduced" \
  | grep -v "is deprecated" \
  | grep -v "Instructions for updating" \
  | grep -v "Unresolved object in checkpoint" \
  | grep -v "Unsupported signature for serialization" \
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
