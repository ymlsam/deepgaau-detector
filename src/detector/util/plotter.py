import math
import matplotlib
import numpy as np
import os

from matplotlib import pyplot as plt
from object_detection.utils import visualization_utils as viz_utils
from typing import Dict, Iterable, List, Optional


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
    
    
def plot_detections(
        img: np.ndarray,
        boxes: np.ndarray,
        clss: np.ndarray,
        scores: Optional[np.ndarray],
        min_score: float,
        cat_idx: Dict,
        out_path: str = '',
) -> None:
    """to visualize detections
    
    Args:
        img      : uint8 numpy array with shape (img_height, img_width, 3)
        boxes    : numpy array of shape [N, 4]
        clss     : numpy array of shape [N]. Note that class indices are 1-based, and match the keys in the label map.
        scores   : numpy array of shape [N] or None. If scores=None, then this function assumes that the boxes to be plotted are groundtruth boxes and plot all boxes as black with no classes or scores.
        min_score: minimum score threshold for a box or keypoint to be visualized.
        cat_idx  : dict containing category dictionaries (each holding category index `id` and category name `name`) keyed by category indices.
        out_path : name for the image file.
    """
    annotated_img = img.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(annotated_img, boxes, clss, scores, cat_idx, use_normalized_coordinates=True, min_score_thresh=min_score)
    
    if out_path:
        plt.imsave(out_path, annotated_img)
    else:
        plt.imshow(annotated_img)


def plot_detectionss(
        imgs: List[np.ndarray],
        img_paths: List[str],
        boxes_list: Iterable[np.ndarray],
        clss_list: Iterable[np.ndarray],
        scores_list: Iterable[Optional[np.ndarray]],
        min_score: float,
        cat_idx: Dict,
        out_dir: str = '',
        row: int = 0,
        col: int = 3,
        lmt: int = 0,
) -> None:
    if not out_dir:
        config()
    elif not os.path.isdir(out_dir):
        os.mkdir(out_dir, mode=0o755)
    
    if not lmt:
        lmt = len(imgs)
    
    if not row:
        row = math.ceil(min(len(imgs), lmt) / col)
        
    for i, (img, img_path, boxes, clss, scores) in enumerate(zip(imgs, img_paths, boxes_list, clss_list, scores_list)):
        # show detections for each image
        if i >= row * col or i >= lmt:
            break
        
        if not out_dir:
            plt.subplot(row, col, i + 1)
            
        fn = os.path.splitext(os.path.basename(img_path))[0] + '.out' if img_path else 'out.' + ('%d' % i)
        out_path = (out_dir + '/' + fn + '.jpg') if out_dir else ''
        
        plot_detections(img, boxes, clss, scores, min_score, cat_idx, out_path=out_path)
    
    if not out_dir:
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
