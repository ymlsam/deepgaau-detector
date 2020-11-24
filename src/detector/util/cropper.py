import numpy as np

from PIL import Image
from typing import Optional, Tuple


def crop_img(img: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Crop an image in numpy array format
  
    Args:
        img: A [height, width, 3] numpy array
        box: (start y ratio, start x ratio, end y ratio, end x ratio)
  
    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    
    h, w, c = img.shape
    
    y0 = max(int(round(h * box[0])), 0)
    x0 = max(int(round(w * box[1])), 0)
    y1 = min(int(round(h * box[2])), h)
    x1 = min(int(round(w * box[3])), w)
    
    if y0 >= y1 or x0 >= x1:
        return None
    
    return img[y0:y1, x0:x1, :]
