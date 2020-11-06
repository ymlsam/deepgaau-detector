#!/bin/bash

# go to project root
PT=$DG/pricetag
cd "$PT" || exit

# check & reset shell script permission
find ./op -type f -name "*.sh" -exec ls -l {} \;
find ./op -type f -name "*.sh" -exec chmod 744 {} \;
find ./op -type f -name "*.sh" -exec ls -l {} \;

# create, activate & check virtual environment (the leading dot causes the script to be executed in current shell instead of sub-shell)
python -m venv .venv
. ./op/venv/activate.sh
pip -V

# install dependencies (via setup.py)
pip install -e .

# install protoc
PROTOC_ZIP=protoc-3.7.1-osx-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP

# install tensorflow object detection api
mkdir tensorflow
cd tensorflow || exit
git clone https://github.com/tensorflow/models.git
cd models/research || exit
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install --use-feature=2020-resolver .

# download checkpoint (expect to see "tar: Failed to set default locale" message)
cd "$PT/tensorflow" || exit
NET=ssd_resnet50_v1_fpn_640x640_coco17_tpu-8
mkdir -p "net/$NET"
curl -O "http://download.tensorflow.org/models/object_detection/tf2/20200711/$NET.tar.gz"
tar -xf "$NET.tar.gz"
mv "$NET/checkpoint" "net/$NET/"
rm -rf "$NET"
rm -f "$NET.tar.gz"

# create data directory & import testing images
cd "$PT" || exit
mkdir data
cd data || exit
cp -r "$PT/tensorflow/models/research/object_detection/test_images/ducky" .

# config for non-Jupyter environment in order to use plt.show()
echo "comment out the line with \"matplotlib.use('Agg')\" for non-Jupyter environment"
echo "> research/object_detection/utils/visualization_utils.py"
read -r -p "(press enter to proceed)"
vi research/object_detection/utils/visualization_utils.py
