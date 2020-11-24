import math
import matplotlib
import numpy as np

from matplotlib import pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
from typing import Dict, List, Optional


def config() -> None:
    # for display via Jupyter, "matplotlib.use('Agg')" is called during import of visualization_utils (and some other files)
    # this prevents plot to be shown on local machine
    # reset matplotlib backend to workaround
    if is_mac():
        matplotlib.use('MacOSX')
    
    # configure after fixing backend
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [14, 7]


def is_mac() -> bool:
    return True
    
    
def plot_detections(img: np.ndarray, boxes: np.ndarray, classes: np.ndarray, scores: Optional[np.ndarray], cat_idx: Dict, out_img_path: str = '') -> None:
    """to visualize detections
    
    Args:
        img         : uint8 numpy array with shape (img_height, img_width, 3)
        boxes       : numpy array of shape [N, 4]
        classes     : numpy array of shape [N]. Note that class indices are 1-based, and match the keys in the label map.
        scores      : numpy array of shape [N] or None. If scores=None, then this function assumes that the boxes to be plotted are groundtruth boxes and plot all boxes as black with no classes or scores.
        cat_idx     : dict containing category dictionaries (each holding category index `id` and category name `name`) keyed by category indices.
        out_img_path: name for the image file.
    """
    annotated_img = img.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(annotated_img, boxes, classes, scores, cat_idx, use_normalized_coordinates=True, min_score_thresh=0.8)
    
    if out_img_path:
        plt.imsave(out_img_path, annotated_img)
    else:
        plt.imshow(annotated_img)


def plot_detectionss(imgs: List[np.ndarray], boxes_list: List[np.ndarray], classes_list: List[np.ndarray], scores_list: List[Optional[np.ndarray]], cat_idx: Dict, out_img_dir: str = '', row: int = 0, col: int = 3, lmt: int = 0) -> None:
    if not out_img_dir:
        config()
    
    if not lmt:
        lmt = len(imgs)
    
    if not row:
        row = math.ceil(min(len(imgs), lmt) / col)
        
    for i, (img, boxes, classes, scores) in enumerate(zip(imgs, boxes_list, classes_list, scores_list)):
        # show detections for each image
        if i >= row * col or i >= lmt:
            break
        
        if not out_img_dir:
            plt.subplot(row, col, i + 1)
        
        out_img_path = (out_img_dir + '/out_' + ('%02d' % i) + '.jpg') if out_img_dir else ''
        
        plot_detections(img, boxes, classes, scores, cat_idx, out_img_path=out_img_path)
    
    if not out_img_dir:
        plt.show()


def plot_imgs(imgs: List[np.ndarray], row: int = 0, col: int = 3, lmt: int = 0) -> None:
    config()
    
    if not lmt:
        lmt = len(imgs)
    
    if not row:
        row = math.ceil(lmt / col)
    
    for i, img in enumerate(imgs):
        if i >= row * col:
            break
        
        plt.subplot(row, col, i + 1)
        plt.imshow(img)
    
    plt.show()
