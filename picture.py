from typing import Tuple
import numpy as np
import matplotlib.image as mpimg
from scipy.interpolate import interpn


def _get_line_coords(line_points: np.ndarray, n_pts: int = 300) -> Tuple[np.ndarray, np.ndarray]:
  x = np.linspace(line_points[0, 0], line_points[1, 0], n_pts)
  y = np.linspace(line_points[0, 1], line_points[1, 1], n_pts)
  return x, y


def _get_rect_coords(rect_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  idxs_x = [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]]
  idxs_y = [[0, 1], [1, 1], [1, 1], [0, 1], [0, 1]]
  return np.array([rect_points[i[0]][i[1]] for i in idxs_x]), \
    np.array([rect_points[i[0]][i[1]] for i in idxs_y])


def load_image(path: str) -> np.ndarray:
    img = mpimg.imread(path, format='jpg')
    return img


def select_rectangle(temperatures: np.ndarray, rect_points) -> np.ndarray:
  return temperatures[rect_points[:, 1].min():rect_points[:, 1].max(), rect_points[:, 0].min():rect_points[:, 0].max()]


def select_line(temperatures: np.ndarray, line_points: np.ndarray) -> np.ndarray:
  x, y = _get_line_coords(line_points)
  points = (np.arange(temperatures.shape[0]), np.arange(temperatures.shape[1]))
  z = interpn(points, temperatures, np.array(list(zip(x, y))))
  return x, y, z