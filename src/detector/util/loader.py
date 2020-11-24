import numpy as np

from PIL import Image


def load_img(path: str) -> np.ndarray:
    """load an image from file into a numpy array
    
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
  
    Args:
        path: a file path.
  
    Returns:
        uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
