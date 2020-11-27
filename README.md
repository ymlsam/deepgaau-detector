# Deepgaau Detector
Deepgaau Detector is a custom object detector using Tensorflow v2 with few-shot learning. It is fully operational with:
* train/dev/test dataset management
* integration with image annotation tool
* choices of pre-trained models to fine-tune from
* customizable config tuned for eager learning
* helper scripts for training, evaluation & detection with python type hinting

I started this project due to frustration during my study in the Object Detection API. The motivation is to share a complete & reusable codebase using latest version of TF & OD API.

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

## Project Setup
* open terminal
* go to project root
* make shell scripts executable
```
find op -type f -name "*.sh" -exec chmod u+x {} \;
```
* install main environment variable $DGD

[`. op/setup/profile.sh`](op/setup/profile.sh)
* refer to env var in "[op/env.sh](op/env.sh)", change values as needed (keep env var name or predefined shell scripts will be broken)
* initialize project

[`op/setup/init.sh`](op/setup/init.sh)
* adopt ".venv/bin/python" as Python interpreter for your IDE

# Data & Preprocessing
## Image Preparation
* create data directory & import sample images

[`op/img/init.sh`](op/img/init.sh)

## Label Map
* create or update label map accordingly

[`vi "$DGD_DATA_LABEL"`](data/ducky/ducky.pbtxt)

## Image Annotation
[`op/img/annotate.sh`](op/img/annotate.sh)

## Dataset Splitting
[`op/img/split.sh`](op/img/split.sh)

## XML to TFRecord Conversion
[`op/img/convert.sh`](op/img/convert.sh)

# Object Detection Model
## Initialization
* make sure env var "$DGD_NET" in [op/env.sh](op/env.sh) is pointing to the network that you would to like to start fine-tuning from
* download pre-trained network checkpoint & import config

[`op/model/import.sh`](op/model/import.sh)

## Configuration
* prepare various paths using env var
```
echo "$DGD_NET_IMPORT/checkpoint/ckpt-0"
echo "$DGD_DATA_LABEL"
echo "$DGD_DATA/train/_train.tfrecord"
echo "$DGD_DATA/dev/_dev.tfrecord"
```
* edit config, replacing env var with actual value

[`vi "$DGD_NET_CONF"`](model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config)
* model.ssd.num_classes: 1
* model.ssd.freeze_batchnorm: true
* train_config.num_steps: 100
* train_config.batch_size: 8 ("batch_size: 64" may result in memory outage in local machine, causing training script to be killed without error message)
* train_config.add_regularization_loss: false
* train_config.optimizer.momentum_optimizer.learning_rate.constant_learning_rate.learning_rate: 0.01
* train_config.optimizer.momentum_optimizer.momentum_optimizer_value: 0.9
* remove "train_config.data_augmentation_options"
* train_config.fine_tune_checkpoint: "$DGD_NET_IMPORT/checkpoint/ckpt-0"
* train_config.fine_tune_checkpoint_type: "detection"
* train_config.update_trainable_variables: ["WeightSharedConvolutionalBoxHead", "WeightSharedConvolutionalClassHead"]
* train_config.use_bfloat16: false
* train_input_reader.label_map_path: "$DGD_DATA_LABEL"
* train_input_reader.tf_record_input_reader.input_path: "$DGD_DATA/train/_train.tfrecord"
* eval_input_reader.label_map_path: "$DGD_DATA_LABEL"
* eval_input_reader.tf_record_input_reader.input_path: "$DGD_DATA/dev/_dev.tfrecord"

## Training & Evaluation
* start training based on config & train set (training is cumulative, you may want to change or clean up training directory during config tuning)

[`op/model/train.sh`](op/model/train.sh)
* evaluate periodically based on config & dev set

[`op/model/eval.sh`](op/model/eval.sh)
* track progress via TensorBoard

[`op/model/report.sh`](op/model/report.sh)

## Exporting
* export trained model (you may want to change export directory first to avoid overwriting previously exported checkpoint or model)

[`op/model/export.sh`](op/model/export.sh)

## Detection
* detect objects from test set

[`op/detect.sh`](op/detect.sh)
* check "$DGD_DATA/output" for detected objects
* as a reference, for "[ssd_resnet50_v1_fpn_640x640_coco17_tpu-8](model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8)", detection starts working with around 100 epochs (num_steps) on the ducky dataset, with edge cases (e.g. ducky.0.jpg in test set) still fail
* change "in_path" in "[op/detect.sh](op/detect.sh)" to an image directory or a specific image file (.jpg)fff that you would like to detect
* change "out_dir" in "[op/detect.sh](op/detect.sh)" to alternative output directory

# Acknowledgement
* Tensorflow Eager Few Shot Object Detection Colab
> https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/eager_few_shot_od_training_tf2_colab.ipynb
* TensorFlow 2 Object Detection API Tutorial - Training Custom Object Detector
> https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html
