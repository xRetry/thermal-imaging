from typing import List, Tuple, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import picture, plotting, thermal


class ThermalImage:
    _image: np.ndarray
    _temperatures: np.ndarray
    _tolerance: float
    _colormap: matplotlib.colors.Colormap

    def __init__(self, path: str, t_min: float, t_max: float, bar_location: str = 'bottom', 
    interp: str = 'nearest', thermal_tolerance: Optional[float] = None, colormap: str = 'original'):
        self._image = picture.load_image(path)
        self._temperatures = thermal.convert_to_temperature(self._image, t_min, t_max, bar_location, interp)
        self._tolerance = thermal_tolerance / np.sqrt(3) * 2  # 95% uncertainty interval for rectangular tolerance
        if colormap == 'original':
            self._colormap = plotting.get_colormap(self._image, bar_location)
        else:
            self._colormap = plt.cm.get_cmap(colormap)

    def plot_image(self):
        plotting.plot_image(self._image)
    
    def plot_temperatures(self):
        plotting.plot_image(self._temperatures, self._colormap)

    def plot_line(self, x1: int, y1: int, x2: int, y2: int):
        line_points = np.array([[x1, y1], [x2, y2]])
        _, _, z_line = picture.select_line(self._temperatures, line_points)
        plotting.plot_image(self._temperatures, self._colormap, line_points)
        plotting.plot_line(z_line, self._tolerance)
    
    def plot_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        rect_points = np.array([[x1, y1], [x2, y2]])
        temp_selection = picture.select_rectangle(self._temperatures, rect_points)
        plotting.plot_image(self._temperatures, self._colormap, rect_points=rect_points)
        plotting.plot_image(temp_selection, self._colormap)
        plotting.plot_histogram(temp_selection)