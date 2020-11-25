import numpy as np
import os
import tensorflow as tf

from glob import glob
from object_detection.builders import model_builder
from object_detection.core.model import DetectionModel
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util, label_map_util
from typing import Dict, Iterable, List, Tuple, TypedDict

from .util import cropper, loader, plotter


# typing
class Detections(TypedDict):
    detection_boxes: tf.Tensor
    detection_classes: tf.Tensor
    detection_scores: tf.Tensor


def detect_array(model: DetectionModel, imgs: List[np.ndarray]) -> (Iterable[np.ndarray], Iterable[np.ndarray], Iterable[np.ndarray]):
    print('=== detect ===', flush=True)
    
    # (for batch detection to work, images must be of same shape)
    # batch convert numpy array to tensor
    # img_tensors = tf.convert_to_tensor(imgs, dtype=tf.float32)
    
    # batch detect
    # detections = detect_tensor(model, img_tensors)
    
    # batch convert tensor to numpy array
    # boxes_list = detections['detection_boxes'].numpy()
    # clss_list = detections['detection_classes'].numpy()
    # scores_list = detections['detection_scores'].numpy()

    boxes_list = []
    clss_list = []
    scores_list = []

    for i, img in enumerate(imgs):
        # convert numpy array to tensor
        input_tensor = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=0)

        # detect
        detections = detect_tensor(model, input_tensor)

        # convert tensor to numpy array
        boxes_list.append(detections['detection_boxes'][0].numpy())
        clss_list.append(detections['detection_classes'][0].numpy())
        scores_list.append(detections['detection_scores'][0].numpy())
    
    return boxes_list, clss_list, scores_list


def detect_file(model: DetectionModel, img_path: str) -> (Iterable[np.ndarray], Iterable[np.ndarray], Iterable[np.ndarray], List[np.ndarray], List[str]):
    # load images into numpy array
    imgs, img_paths = load_imgs(img_path)
    
    # detect
    return detect_array(model, imgs) + (imgs,) + (img_paths,)


@tf.function
def detect_tensor(model: DetectionModel, imgs: tf.Tensor) -> Detections:
    """Run detection on input images.
    
    Args:
        model: detection model
        imgs: A [batch, height, width, 3] numpy array. Note that height and width can be anything since the images will
            be immediately resized according to the needs of the model within this function.
    
    Returns:
        A dict containing 3 Tensors (`detection_boxes`, `detection_classes`, and `detection_scores`).
    """
    
    preprocessed_imgs, shapes = model.preprocess(imgs)
    prediction_dict = model.predict(preprocessed_imgs, shapes)
    detections = model.postprocess(prediction_dict, shapes)
    
    label_id_offset = 1
    cls_field = fields.DetectionResultFields.detection_classes
    detections[cls_field] = tf.cast(detections[cls_field], tf.int32) + label_id_offset
    
    return detections


def get_cat_idx(label_map_path: str) -> Dict:
    # by convention, non-background classes start counting at 1
    return label_map_util.create_category_index_from_labelmap(label_map_path)


def is_file_path(path: str, exts: Tuple[str]) -> bool:
    for ext in exts:
        if path.endswith('.' + ext):
            return True
    
    return False


def load_imgs(img_path: str, img_exts: Tuple[str] = ('jpg',)) -> Tuple[List[np.ndarray], List[str]]:
    # load images
    print('=== load images ===', flush=True)
    
    is_file = is_file_path(img_path, img_exts)
    image_paths = []
    
    if is_file:
        # load a single image
        image_paths.extend(glob(img_path))
    else:
        # load all images from a directory
        for ext in img_exts:
            image_paths.extend(glob(os.path.join(img_path, '*.' + ext)))
        
        # sort by file name
        image_paths = sorted(image_paths)
    
    imgs = [loader.load_img(image_path) for image_path in image_paths]
    
    # visualize images
    # plotter.plot_imgs(imgs, lmt=9)
    
    return imgs, image_paths


def load_model(model_config: Dict, ckpt_path: str) -> DetectionModel:
    # SSDMetaArch > DetectionModel > tf.keras.layers.Layer
    print('=== load model ===', flush=True)
    tf.keras.backend.clear_session()
    
    model = model_builder.build(model_config=model_config, is_training=False)
    
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(ckpt_path).expect_partial()
    
    return model


def save_detectionss(
        imgs: List[np.ndarray],
        img_paths: List[str],
        boxes_list: Iterable[np.ndarray],
        clss_list: Iterable[np.ndarray],
        scores_list: Iterable[np.ndarray],
        min_score: float,
        cat_idx: Dict,
        out_dir: str,
) -> None:
    if not out_dir:
        return
    
    # make directory
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir, mode=0o755)
    
    # crop detections for each image
    for i, (img, img_path, boxes, clss, scores) in enumerate(zip(imgs, img_paths, boxes_list, clss_list, scores_list)):
        fn = os.path.splitext(os.path.basename(img_path))[0] if img_path else 'out.' + ('%d' % i)
        
        for j, (box, cls, score) in enumerate(zip(boxes, clss, scores)):
            if not cls or score < min_score:
                continue
            
            # crop image
            out_img = cropper.crop_img(img, box)
            
            if out_img is None:
                continue
            
            # format file name
            out_cls = ('.' + cat_idx[cls]['name']) if cls in cat_idx else ''
            out_path = out_dir + '/' + fn + '.' + ('%d' % j) + out_cls + '.jpg'
            
            # save image
            loader.save_img(out_img, out_path)


# main
def main(config_path: str, ckpt_path: str, in_path: str, out_dir: str) -> None:
    # config
    configs = config_util.get_configs_from_pipeline_file(config_path)
    cat_idx = get_cat_idx(configs['eval_input_config'].label_map_path)
    
    # model
    model = load_model(configs['model'], ckpt_path)
    
    # detect
    (boxes_list, clss_list, scores_list, imgs, img_paths) = detect_file(model, in_path)
    
    # scores outputted by different networks vary
    # scores of true positive detection from resnet can be as high as 0.9-1.0, while they are much lower in mobilenet
    min_score = 0.5
    
    # visualize detections
    plot_dir = os.path.join(out_dir, 'plot') if out_dir else ''
    plotter.plot_detectionss(imgs, img_paths, boxes_list, clss_list, scores_list, min_score, cat_idx, out_dir=plot_dir)
    
    # output detected objects as image files
    save_detectionss(imgs, img_paths, boxes_list, clss_list, scores_list, min_score, cat_idx, out_dir)
