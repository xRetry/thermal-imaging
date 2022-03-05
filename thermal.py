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



def convert_to_temperature(image: np.ndarray, t_min: float, t_max: float, bar_location: str = 'bottom', interp: str = 'nearest'):
  colors = _get_colors(image, bar_location)
  temperature_range = np.linspace(t_min, t_max, len(colors))
  if interp == 'nearest':
    interpolation = NearestNDInterpolator(colors, temperature_range)
  elif interp == 'rbf':
    interpolation = RBFInterpolator(colors, temperature_range)
  return interpolation(image)



