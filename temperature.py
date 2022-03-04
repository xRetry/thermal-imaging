from typing import Tuple
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator, LinearNDInterpolator, interpn


def _get_colors(image: np.ndarray, bar_location: str = 'bottom') -> np.ndarray:
  if bar_location == 'bottom':
    colors = image[-1, :, :]
  elif bar_location == 'top':
    colors = image[0, :, :]
  elif bar_location == 'left':
    colors = image[:, 0, :]
  elif bar_location == 'right':
    colors = image[:, -1, :]
  else:
    raise KeyError(f'Invalid bar location {bar_location}! Use \'top\', \'bottom\', \'left\' or \'right\'')
  return colors


def get_standard_uncertainty(tolerance: float = 0.07/2) -> float:
  # +- value in [K] for Compact XR
  std_u = tolerance / np.sqrt(3) # square distribution: tol / sqrt(3) -> 0.68


def _get_line_coords(line_points: np.ndarray, n_pts: int = 300) -> Tuple[np.ndarray, np.ndarray]:
  x = np.linspace(line_points[0, 0], line_points[1, 0], n_pts)
  y = np.linspace(line_points[0, 1], line_points[1, 1], n_pts)
  return x, y


def _get_rect_coords(rect_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  idxs_x = [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]]
  idxs_y = [[0, 1], [1, 1], [1, 1], [0, 1], [0, 1]]
  return np.array([rect_points[i[0]][i[1]] for i in idxs_x]), \
    np.array([rect_points[i[0]][i[1]] for i in idxs_y])


def convert_to_temperature(image: np.ndarray, t_min: float, t_max: float, bar_location: str = 'bottom', interp: str = 'nearest'):
  colors = _get_colors(image, bar_location)
  temperature_range = np.linspace(t_min, t_max, len(colors))
  if interp == 'nearest':
    interpolation = NearestNDInterpolator(colors, temperature_range)
  elif interp == 'rbf':
    interpolation = RBFInterpolator(colors, temperature_range)
  return interpolation(image)


def select_rectangle(temperatures: np.ndarray, rect_points) -> np.ndarray:
  return temperatures[rect_points[:, 1].min():rect_points[:, 1].max(), rect_points[:, 0].min():rect_points[:, 0].max()]


def select_line(temperatures: np.ndarray, line_points: np.ndarray) -> np.ndarray:
  x, y = _get_line_coords(line_points)
  # interpolation = NearestNDInterpolator(list(zip(np.arange(temperatures.shape[0]), np.arange(temperatures.shape[1]))), temperatures)
  # z = interpolation(x, y)
  # grid_x, grid_y = np.meshgrid(np.arange(temperatures.shape[0]), np.arange(temperatures.shape[1]))
  points = (np.arange(temperatures.shape[0]), np.arange(temperatures.shape[1]))
  z = interpn(points, temperatures, np.array(list(zip(x, y))))
  return x, y, z
