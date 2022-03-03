import numpy as np
import matplotlib.image as mpimg


def load_image(path: str) -> np.ndarray:
    img = mpimg.imread(path, format='jpg')
    return img