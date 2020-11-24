#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
cd "$DGD" || exit
mkdir -p "$DGD_DIR_NET"
cd "$DGD_DIR_NET" || exit

# download pre-trained network checkpoint (expect to see "tar: Failed to set default locale" message)
DGD_NET_FN="$DGD_NET.tar.gz"
curl -O "http://download.tensorflow.org/models/object_detection/tf2/20200711/$DGD_NET_FN"
tar -xf "$DGD_NET_FN"
rm -f "$DGD_NET_FN"
mv "$DGD_NET" "$(basename "$DGD_DIR_NET_IMPORT")"

# clone config (do not overwrite existing config)
cd "$DGD" || exit
cp -n "$DGD_DIR_NET_IMPORT/pipeline.config" "$DGD_CONF_NET"
