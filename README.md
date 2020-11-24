# Installation
## Environment
* tested on macOS 10.15.7

## Python
* version: 3.8.6 (python-3.8.6-macosx10.9.pkg), tensorflow 2.3.1 is incompatible with python v3.9+
* open or double click the .pkg file to install
* create symbolic link
```
sudo ln -s python3 /usr/local/bin/python
```
* restart terminal
* check version (expecting "Python 3.8.6")
```
python --version
```

### Setup
* open terminal
* go to project root
* make shell scripts executable
```
find op -type f -name "*.sh" -exec chmod u+x {} \;
```
* install main environment variable $DGD
```
. op/profile.sh
```
* change env var value in "op/env.sh" as needed (keep env var name or predefined shell scripts will be broken)
* initialize project
```
op/init.sh
```

# Data & Preprocessing
## Image Preparation
* create data directory & import sample images
```
op/img/init.sh
```

## Label Map
* update label map accordingly
```
vi "$DGD_CONF_LABEL"
```

## Image Annotation
```
op/img/annotate.sh
```

## Dataset Splitting
```
op/img/split.sh
```

## XML to TFRecord Conversion
```
op/img/convert.sh
```

# Model Initialization & Configuration
* download pre-trained network checkpoint & import config
```
op/model/import.sh
```
* edit config
```
vi "$DGD_CONF_NET"
```
  * model.ssg.num_classes: 1
  * train_config.batch_size: 8 ("batch_size: 64" will probably result in memory outage in local machine, causing the training script to be killed without error message)
  * train_config.fine_tune_checkpoint: "$DGD_DIR_NET_IMPORT/checkpoint/ckpt-0"
  * train_config.fine_tune_checkpoint_type: "detection"
  * train_config.use_bfloat16: false
  * train_input_reader.label_map_path: "$DGD_CONF_LABEL"
  * train_input_reader.tf_record_input_reader.input_path: "$DGD_DIR_DATA/train/_train.tfrecord"
  * eval_input_reader.label_map_path: "$DGD_CONF_LABEL"
  * eval_input_reader.tf_record_input_reader.input_path: "$DGD_DIR_DATA/dev/_dev.tfrecord"

# Training & Evaluation
* start training & evaluation based on config & TFRecord files
```
op/model/train_eval.sh
```
* track progress via TensorBoard
```
op/model/report.sh
```

# Exporting
* export trained model
```
op/model/export.sh
```

# Acknowledgement
* Tensorflow Eager Few Shot Object Detection Colab
> https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
* TensorFlow 2 Object Detection API Tutorial - Training Custom Object Detector
> https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
