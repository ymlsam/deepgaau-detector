#!/bin/bash

# prepare env
. "$DGD/op/env.sh" || exit

# prepare folder
cd "$DGD" || exit
mkdir -p "$DGD_NET"
cd "$DGD_NET" || exit

# download pre-trained network checkpoint (expect to see "tar: Failed to set default locale" message)
DGD_NET_NAME="$(basename "$DGD_NET")"
DGD_NET_FILE="$DGD_NET_NAME.tar.gz"
curl -O "http://download.tensorflow.org/models/object_detection/tf2/20200711/$DGD_NET_FILE"
tar -xf "$DGD_NET_FILE"
rm -f "$DGD_NET_FILE"
mv "$DGD_NET_NAME" "$(basename "$DGD_NET_IMPORT")"

# clone config (do not overwrite existing config)
cd "$DGD" || exit
cp -n "$DGD_NET_IMPORT/pipeline.config" "$DGD_NET_CONF"
