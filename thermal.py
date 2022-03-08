from typing import Tuple, Dict
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator, LinearNDInterpolator, interpn, interp2d


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


def get_uncertainty(image: np.ndarray, bar_location: str, temperatures: np.ndarray, t_min: float, t_max: float, thermal_tolerance: float = 0.07) -> dict:
  n = len(_get_colors(image, bar_location))
  # +- value in [K] for Compact XR
  u_max = 0.5 / np.sqrt(3)
  u_min = u_max
  i = np.arange(n)
  u_scale = np.sqrt(((1 - i / (n-1))*u_min)**2 + ((i/(n-1)) * u_max)**2)
  u_interp = (t_max - t_min) / (4*np.sqrt(3)*n) 
  u = np.sqrt(u_scale**2 + u_interp**2)
  temperature_scale = np.linspace(t_min, t_max, n)
  mapping = dict(zip(temperature_scale, u))
  return mapping
  

def get_line_uncertainty(uncertainties: np.ndarray, x_line, y_line) -> np.ndarray:
  x = np.arange(uncertainties.shape[0]) 
  y = np.arange(uncertainties.shape[1])
  xy_line = np.column_stack([x_line, y_line])
  return interpn((x,y), uncertainties, xy_line)


def convert_to_temperature(image: np.ndarray, t_min: float, t_max: float, bar_location: str = 'bottom', interp: str = 'nearest'):
  colors = _get_colors(image, bar_location)
  temperature_range = np.linspace(t_min, t_max, len(colors))
  if interp == 'nearest':
    interpolation = NearestNDInterpolator(colors, temperature_range)
  elif interp == 'rbf':
    interpolation = RBFInterpolator(colors, temperature_range)
  return interpolation(image)



