import glob
import numpy as np
import os
import random
import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.core.model import DetectionModel
from object_detection.utils import config_util
from tensorflow import Tensor, Variable
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from typing import Callable, Dict, List, Tuple

from .util import loader, plotter


def annotate_imgs(imgs: List[np.ndarray], num_classes: int) -> List[np.ndarray]:
    print('=== annotate images ===')
    # gt_boxes = []
    # colab_utils.annotate(train_images_np, box_storage_pointer=gt_boxes)
    
    gt_boxes = [
        np.array([[0.436, 0.591, 0.629, 0.712]], dtype=np.float32),
        np.array([[0.539, 0.583, 0.73, 0.71]], dtype=np.float32),
        np.array([[0.464, 0.414, 0.626, 0.548]], dtype=np.float32),
        np.array([[0.313, 0.308, 0.648, 0.526]], dtype=np.float32),
        np.array([[0.256, 0.444, 0.484, 0.629]], dtype=np.float32),
    ]
    
    return gt_boxes


def detect(model: DetectionModel, test_imgs: List[np.ndarray], label_id_offset: int, cat_idx: Dict, detect_img_dir: str = '') -> None:
    print('=== detect ===')
    boxes_list = []
    classes_list = []
    scores_list = []
    
    for i, img in enumerate(test_imgs):
        input_tensor = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)
        detections = detect_one(model, input_tensor)

        boxes_list.append(detections['detection_boxes'][0].numpy())
        classes_list.append(detections['detection_classes'][0].numpy().astype(np.uint32) + label_id_offset)
        scores_list.append(detections['detection_scores'][0].numpy())
    
    # wait for detections for all images
    
    # visualize detections
    plotter.plot_detectionss(test_imgs, boxes_list, classes_list, scores_list, cat_idx, out_img_dir=detect_img_dir)


@tf.function
def detect_one(model: DetectionModel, input_tensor: Tensor) -> Dict:
    """Run detection on an input image.
    
    Args:
        model: detection model
        input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
            Note that height and width can be anything since the image will be
            immediately resized according to the needs of the model within this
            function.
    
    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
            and `detection_scores`).
    """
    preprocessed_image, shapes = model.preprocess(input_tensor)
    prediction_dict = model.predict(preprocessed_image, shapes)
    
    return model.postprocess(prediction_dict, shapes)


def get_cat_idx() -> Dict:
    # by convention, non-background classes start counting at 1
    duck_class_id = 1
    
    return {
        'duck_class_id': {
            'id': duck_class_id,
            'name': 'rubber_ducky',
        }
    }


def get_train_step_fn(model: DetectionModel, batch_size: int, optimizer: OptimizerV2, vars_to_fine_tune: List[Variable]) -> Callable[[List[Tensor], List[Tensor], List[Tensor]], Tensor]:
    # Set up forward + backward pass for a single train step.
    """get a tf.function for training step"""
    
    # Use tf.function for a bit of speed.
    # Comment out the tf.function decorator if you want the inside of the function to run eagerly.
    @tf.function
    def train_step_fn(image_tensors: List[Tensor], groundtruth_boxes_list: List[Tensor], groundtruth_classes_list: List[Tensor]):
        """single training iteration
        
        Args:
            image_tensors: A list of [1, height, width, 3] Tensor of type tf.float32.
                Note that the height and width can vary across images, as they are
                reshaped within this function to be 640x640.
            groundtruth_boxes_list: A list of Tensors of shape [N_i, 4] with type
                tf.float32 representing groundtruth boxes for each image in the batch.
            groundtruth_classes_list: A list of Tensors of shape [N_i, num_classes]
                with type tf.float32 representing groundtruth boxes for each image in
                the batch.
            
        Returns:
          A scalar tensor representing the total loss for the input batch.
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list=groundtruth_boxes_list, groundtruth_classes_list=groundtruth_classes_list)
        
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([model.preprocess(image_tensor)[0] for image_tensor in image_tensors], axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
        
        return total_loss
    
    return train_step_fn


def init_model(num_classes: int) -> DetectionModel:
    print('=== init model ===', flush=True)
    tf.keras.backend.clear_session()
    
    network_name = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
    pipeline_config = 'tensorflow/models/research/object_detection/configs/tf2/' + network_name + '.config'
    checkpoint_path = 'tensorflow/net/' + network_name + '/checkpoint/ckpt-0'
    print(network_name)
    
    # Load pipeline config and build a detection model.
    #
    # Since we are working off of a COCO architecture which predicts 90
    # class slots by default, we override the `num_classes` field here to be just one (for our new rubber ducky class).
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=True)
    
    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression. We will
    # restore the box regression head but initialize the classification head
    # from scratch (we show the omission below by commenting out the line that
    # we would add if we wanted to restore both heads)
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        # _prediction_heads=detection_model._box_predictor._prediction_heads,  # (i.e., the classification head that we *will not* restore)
        _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(checkpoint_path).expect_partial()
    
    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    
    return detection_model


def load_imgs(img_dir: str, img_pattern: str = '*.jpg') -> List[np.ndarray]:
    # load images
    print('=== load images ===')
    imgs = []
    image_paths = glob.glob(os.path.join(img_dir, img_pattern))
    image_paths = sorted(image_paths)
    
    for image_path in image_paths:
        imgs.append(loader.load_img(image_path))
    
    # visualize images
    # plotter.plot_imgs(imgs, lmt=9)
    
    return imgs


def load_model() -> DetectionModel:
    #  TODO: load saved model
    print('TODO: load saved model')
    return None


def prepare_data(train_imgs: List[np.ndarray], gt_boxes: List[np.ndarray], num_classes: int, label_id_offset: int, cat_idx: Dict, annotate_img_dir: str = '') -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
    print('=== prepare data ===')
    
    # Convert class labels to one-hot; convert everything to tensors.
    # The `label_id_offset` here shifts all classes by a certain number of indices;
    # we do this here so that the model receives one-hot labels where non-background
    # classes start counting at the zeroth index. This is ordinarily just handled
    # automatically in our training binaries, but we need to reproduce it here.
    train_image_tensors = []
    gt_classes_one_hot_tensors = []
    gt_box_tensors = []
    
    for (img, boxes) in zip(train_imgs, gt_boxes):
        train_image_tensors.append(tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0))
        gt_box_tensors.append(tf.convert_to_tensor(boxes, dtype=tf.float32))
        zero_indexed_groundtruth_classes = tf.convert_to_tensor(np.ones(shape=[boxes.shape[0]], dtype=np.int32) - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))
    
    # visualize annotations
    # classes_list = []
    # scores_list = []
    #
    # for (img, boxes) in zip(train_imgs, gt_boxes):
    #     classes_list.append(np.ones(shape=[len(boxes)], dtype=np.int32))
    #     scores_list.append(np.array([1.0], dtype=np.float32))  # give boxes a dummy score of 100%
    #
    # plotter.plot_detectionss(train_imgs, gt_boxes, classes_list, scores_list, cat_idx, out_img_dir=annotate_img_dir, lmt=9)
    
    return train_image_tensors, gt_box_tensors, gt_classes_one_hot_tensors


def save_model(model: DetectionModel) -> None:
    # TODO: save trained model
    print('TODO: save trained model')


def train_model(model: DetectionModel, train_imgs: List[np.ndarray], gt_boxes: List[np.ndarray], num_classes: int, label_id_offset: int, cat_idx: Dict, annotate_img_dir: str = '') -> None:
    train_img_tensors, gt_box_tensors, gt_classes_one_hot_tensors = prepare_data(train_imgs, gt_boxes, num_classes, label_id_offset, cat_idx, annotate_img_dir)
    
    print('=== train model ===')
    tf.keras.backend.set_learning_phase(True)
    
    # These parameters can be tuned; since our training set has 5 images
    # it doesn't make sense to have a much larger batch size, though we could fit more examples in memory if we wanted to.
    m = len(train_img_tensors)
    batch_size = 4
    learning_rate = 0.01
    epochs = 100
    # learning_rate = 0.02
    # epochs = 1
    
    # Select variables in top layers to fine-tune.
    trainable_variables = model.trainable_variables
    to_fine_tune = []
    prefixes_to_train = ['WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead', 'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    
    for var in trainable_variables:
        if any([var.name.startswith(prefix) for prefix in prefixes_to_train]):
            to_fine_tune.append(var)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    train_step_fn = get_train_step_fn(model, batch_size, optimizer, to_fine_tune)
    
    for idx in range(epochs):
        # Grab keys for a random subset of examples
        all_keys = list(range(m))
        random.shuffle(all_keys)
        example_keys = all_keys[:batch_size]
        
        # Note that we do not do data augmentation in this demo.  If you want a
        # a fun exercise, we recommend experimenting with random horizontal flipping and random cropping :)
        gt_boxes_list = [gt_box_tensors[key] for key in example_keys]
        gt_classes_list = [gt_classes_one_hot_tensors[key] for key in example_keys]
        image_tensors = [train_img_tensors[key] for key in example_keys]
        
        # Training step (forward pass + backwards pass)
        total_loss = train_step_fn(image_tensors, gt_boxes_list, gt_classes_list)
        
        if idx % 10 == 0:
            print('batch ' + str(idx) + ' of ' + str(epochs) + ', loss=' + str(total_loss.numpy()), flush=True)


# main
def run() -> None:
    # config
    img_dir = 'data/ducky'
    train_img_dir = img_dir + '/train'
    test_img_dir = img_dir + '/test'
    
    out_img = True
    annotate_img_dir = img_dir + '/train_annotate' if out_img else ''
    detect_img_dir = img_dir + '/test_detect' if out_img else ''
    
    # build model
    num_classes = 1
    label_id_offset = 1
    cat_idx = get_cat_idx()
    
    model = init_model(num_classes)
    
    # load & annotate training images
    train_imgs = load_imgs(train_img_dir)
    gt_boxes = annotate_imgs(train_imgs, num_classes)
    
    # train
    train_model(model, train_imgs, gt_boxes, num_classes, label_id_offset, cat_idx, annotate_img_dir=annotate_img_dir)
    
    # detect
    test_imgs = load_imgs(test_img_dir)
    detect(model, test_imgs, label_id_offset, cat_idx, detect_img_dir=detect_img_dir)
