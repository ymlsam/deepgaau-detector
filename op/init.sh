#!/bin/bash

# create virtual env
op/venv/init.sh

# prepare env
. "$DGD/op/env.sh" || exit

# update pip
op/python/update_pip.sh

# install dependencies (via setup.py)
pip install -e .

# install protoc
PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

# install tensorflow object detection api
cd "$DGD" || exit
mkdir -p "$DGD_DIR_TF"
cd "$DGD_DIR_TF" || exit
git clone https://github.com/tensorflow/models.git

cd models/research || exit
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

# test tensorflow object detection api (expecting "OK (skipped=1)")
python object_detection/builders/model_builder_tf2_test.py
