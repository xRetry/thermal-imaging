from typing import Tuple, Dict, List, Callable
from functools import partial
import numpy as np
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator, LinearNDInterpolator, interpn, interp2d
from scipy.optimize import curve_fit
import picture


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


# TODO: Add thermal tolerance in calculation
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


def _combine_selections(grid: np.ndarray, lines: List[np.ndarray], rects: List[np.ndarray]) -> np.ndarray:
    line_values = np.array([picture.select_line(grid, line)[2] for line in lines])
    rect_values = np.array([picture.select_rectangle(grid, rect) for rect in rects])
    all_values = np.concatenate([line_values.flatten(), rect_values.flatten()])
    return all_values


def calibrate_temperatures(temperatures: np.ndarray, uncertainties: np.ndarray, line_selections: List[np.ndarray], 
rect_selections: List[np.ndarray], true_temperature: float, tolerance: float) -> None:
    temperatures_selected = _combine_selections(temperatures, line_selections, rect_selections)
    temp_diff = temperatures_selected.mean() - true_temperature
    temperatures -= temp_diff

    uncertainties_selected = _combine_selections(uncertainties, line_selections, rect_selections)
    n_selected = len(temperatures_selected)
    std_selected = np.sqrt(np.sum(temperatures_selected**2 + uncertainties_selected**2) / n_selected - temperatures_selected.mean()**2) 
    u_selected = std_selected / n_selected
    u_ref = tolerance / np.sqrt(3)
    uncertainties = np.sqrt(uncertainties**2 + u_selected**2 + u_ref**2)


def fit_curve(fit_func: Callable, y, uncertainty):
    x = np.arange(len(y))
    params, pcov = curve_fit(fit_func, x, y, sigma=uncertainty, absolute_sigma=False)
    std = np.sqrt(np.diagonal(pcov))
    return params, std