from typing import Tuple
import numpy as np
import matplotlib.image as mpimg
from scipy.interpolate import interpn
import imageio as iio
import core, thermal


def _get_line_coords(line_points: np.ndarray, n_pts: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(line_points[0, 0], line_points[1, 0], n_pts)
    y = np.linspace(line_points[0, 1], line_points[1, 1], n_pts)
    return x, y


def _get_rect_coords(rect_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idxs_x = [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]]
    idxs_y = [[0, 1], [1, 1], [1, 1], [0, 1], [0, 1]]
    return np.array([rect_points[i[0]][i[1]] for i in idxs_x]), \
        np.array([rect_points[i[0]][i[1]] for i in idxs_y])


def load_from_jpg(path: str, t_min: float, t_max: float, bar_location: str = 'bottom', interp: str = 'nearest', thermal_tolerance: float = 0) -> core.ThermalImage:
    image = mpimg.imread(path, format='jpg')
    temperatures = thermal.convert_to_temperature(image, t_min, t_max, bar_location, interp)
    uncertainty_mapping = thermal.get_uncertainty(image, bar_location, temperatures, t_min, t_max, thermal_tolerance=thermal_tolerance)
    func = np.vectorize(lambda x: uncertainty_mapping[x]) 
    uncertainties = func(temperatures)
    th_image = core.ThermalImage(
        image=image,
        temperatures=temperatures,
        uncertainties=uncertainties
    )
    return th_image


def load_from_tiff(path: str, thermal_tolerance) -> core.ThermalImage:
    img = iio.mimread(path, multifile=True)
    th_image = core.ThermalImage(
        image=img[0],
        temperatures=img[1],
        uncertainties=np.ones_like(img[1]) * thermal_tolerance / np.sqrt(3)
    )
    return th_image


def select_rectangle(temperatures: np.ndarray, rect_points) -> np.ndarray:
    return temperatures[rect_points[:, 1].min():rect_points[:, 1].max(), rect_points[:, 0].min():rect_points[:, 0].max()]


def select_line(temperatures: np.ndarray, line_points: np.ndarray) -> np.ndarray:
    x, y = _get_line_coords(line_points)
    points = (np.arange(temperatures.shape[0]), np.arange(temperatures.shape[1]))
    z = interpn(points, temperatures, np.array(list(zip(x, y))))
    return x, y, z