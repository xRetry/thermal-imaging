from typing import List, Optional, Callable, Tuple
import numpy as np
import picture, plotting, thermal


class ThermalImage:
    _image: np.ndarray
    _temperatures: np.ndarray
    _uncertainties: np.ndarray
    _lines: List[np.ndarray]
    _rects: List[np.ndarray]
    _emissivity_camera: float
    _constant: float
    _emissivity_map: np.ndarray

    def __init__(self, image: np.ndarray, temperatures: np.ndarray, uncertainties: Optional[np.ndarray] = None):
        if uncertainties is None:
            uncertainties = np.zeros_like(temperatures)
        self._image = image
        self._temperatures = temperatures
        self._uncertainties = uncertainties
        self._lines, self._rects = [], []
        self._emissivity_camera = 1
        self._constant = 0
        self._emissivity_map = np.ones_like(temperatures)

    def set_camera_emissivity(self, emissivity_camera: float, constant: float):
        self._emissivity_camera = emissivity_camera
        self._constant = constant

    def plot_image(self, title=None):
        plotting.plot_image(self._image, title=title)
    
    def plot_temperatures(self, title=None, cbar_label=None):
        plotting.plot_image((self._temperatures - self._constant) / (self._emissivity_map * self._emissivity_camera), title=title, cbar_label=cbar_label)

    def add_line(self, x1: int, y1: int, x2: int, y2: int):
        line_points = np.array([[x1, y1], [x2, y2]])
        self._lines.append(line_points)
    
    def add_rectangle(self, x1: int, y1: int, x2: int, y2: int):
        rect_points = np.array([[x1, y1], [x2, y2]])
        self._rects.append(rect_points)

    def clear_selection(self):
        self._lines.clear()
        self._rects.clear()

    def plot_selection(self, fit_func: Optional[Callable] = None, title=None, cbar_label=None):
        line_coords = [picture._get_line_coords(l) for l in self._lines]
        rect_coords = [picture._get_rect_coords(r) for r in self._rects]
        plotting.plot_image(self._temperatures, line_coords=line_coords, rect_coords=rect_coords, title=title, cbar_label=cbar_label)
        for line in self._lines:
            x_line, y_line, z_line = picture.select_line(self._temperatures, line)
            sdu = thermal.get_line_uncertainty(self._uncertainties, x_line, y_line)
            params, tol = None, None
            if fit_func is not None:
                params, tol = thermal.fit_curve(fit_func, z_line, sdu)
            plotting.plot_line(z_line, sdu, fit_config=(fit_func, params, tol))

        for rect in self._rects:
            rect_select = picture.select_rectangle(self._temperatures, rect)
            plotting.plot_image(rect_select)
            plotting.plot_histogram(rect_select)

    def plot_emissivity(self, true_temperature: float, title=None, cbar_label=None):
        self._emissivity_map = (self._temperatures - self._constant) / (self._emissivity_camera * true_temperature)
        plotting.plot_image(self._emissivity_map, title=title, cbar_label=cbar_label)

    def calibrate_selection(self, true_temperature: float, tolerance: float = 0):
        temperatures_selected = picture.combine_selections(self._temperatures, self._lines, self._rects)
        uncertainties_selected = picture.combine_selections(self._uncertainties, self._lines, self._rects)
        plotting.plot_histogram(temperatures_selected)
        thermal.calibrate_temperatures(
            self._temperatures,
            self._uncertainties,
            temperatures_selected,
            uncertainties_selected,
            true_temperature,
            tolerance
        )

    def get_selection(self):
        image_selected = picture.combine_selections(self._image, self._lines, self._rects)
        temperatures_selected = picture.combine_selections(self._temperatures, self._lines, self._rects)
        uncertainties_selected = picture.combine_selections(self._uncertainties, self._lines, self._rects)
        return ThermalImage(image_selected, temperatures_selected, uncertainties_selected)

 
def determine_emissivity(img1: ThermalImage, img2: ThermalImage, true_temperature1: float, true_temperature2: float, true_emissivity: float = 1) -> None:
    img1_selected = img1.get_selection()
    img2_selected = img2.get_selection()
    t1_mean = np.mean(img1_selected._temperatures.flatten())
    t2_mean = np.mean(img2_selected._temperatures.flatten())

    emissivity_effective = (t1_mean - t2_mean) / (true_temperature1 - true_temperature2)
    emissivity_camera = emissivity_effective / true_emissivity
    constant = t1_mean - emissivity_effective * true_temperature1

    img1.set_camera_emissivity(emissivity_camera, constant)
    img2.set_camera_emissivity(emissivity_camera, constant)