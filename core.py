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
        self._emissivity_map = np.ones_like(temperatures)
        self._u_emissivity_map = np.zeros_like(temperatures)

    def set_camera_emissivity(self, emissivity_map: np.ndarray, u_emissivity_map: np.ndarray):
        self._emissivity_map = emissivity_map
        self._u_emissivity_map = u_emissivity_map

    def plot_image(self, title=None):
        plotting.plot_image(self._image, title=title)
    
    def plot_temperatures(self, title=None, cbar_label=None):
        plotting.plot_image(self._temperature, title=title, cbar_label=cbar_label)

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

    def plot_emissivity(self, title=None, cbar_label=None):
        line_coords = [picture._get_line_coords(l) for l in self._lines]
        rect_coords = [picture._get_rect_coords(r) for r in self._rects]
        plotting.plot_image(self._emissivity_map, line_coords=line_coords, rect_coords=rect_coords, title=title, cbar_label=cbar_label)
        emissivity_selected = picture.combine_selections(self._emissivity_map, self._lines, self._rects)
        uncertainties_selected = picture.combine_selections(self._u_emissivity_map, self._lines, self._rects)
        emissivity_mean = np.mean(emissivity_selected)
        u_mean = np.sqrt(np.sum((uncertainties_selected / len(uncertainties_selected))**2))
        print(emissivity_mean, u_mean)

    def get_selection(self):
        image_selected = picture.combine_selections(self._image, self._lines, self._rects)
        temperatures_selected = picture.combine_selections(self._temperatures, self._lines, self._rects)
        uncertainties_selected = picture.combine_selections(self._uncertainties, self._lines, self._rects)
        return ThermalImage(image_selected, temperatures_selected, uncertainties_selected)

 
def determine_emissivity(img1: ThermalImage, img2: ThermalImage, true_temperature1: float, true_temperature2: float, 
emissivity_true: float = 1, u_temperature1: float = 0, u_temperature2: float = 0, u_emissivity_true: float = 0) -> None:
    K = 273.15
    img1_selected = img1.get_selection()
    img2_selected = img2.get_selection()
    t1_mean = np.mean(img1_selected._temperatures.flatten()) + K
    t2_mean = np.mean(img2_selected._temperatures.flatten()) + K

    emissivity_effective = (t1_mean**4 - t2_mean**4) / ((true_temperature1+K)**4 - (true_temperature2+K)**4)
    emissivity_camera = emissivity_effective / emissivity_true
    constant = t1_mean**4 - emissivity_effective * (true_temperature1+K)**4

    emissivity_map1 = ((img1._temperatures+K)**4 - constant) / (emissivity_camera * (true_temperature1+K)**4)
    emissivity_map2 = ((img2._temperatures+K)**4 - constant) / (emissivity_camera * true_temperature2)

    u1_mean = np.sqrt(np.sum((img1_selected._uncertainties/len(img1_selected._uncertainties))**2))
    u2_mean = np.sqrt(np.sum((img2_selected._uncertainties/len(img2_selected._uncertainties))**2))

    u_emissivity = np.sqrt(
        ((4 * t1_mean**3)/(true_temperature1**4 - true_temperature2**4))**2 * u1_mean**2 + \
        ((4 * t2_mean**3)/(true_temperature1**4 - true_temperature2**4))**2 * u2_mean**2 + \
        ((4 * true_temperature1**3 * (t1_mean**4 - t2_mean**4)) / (true_temperature1**4 - true_temperature2**4))**2 * u_temperature1**2 + \
        ((4 * true_temperature2**3 * (t1_mean**4 - t2_mean**4)) / (true_temperature1**4 - true_temperature2**4))**2 * u_temperature2**2 
    )

    u_constant = np.sqrt(
        (4*t1_mean**3)**2 * u1_mean**2 + (4*emissivity_effective*true_temperature1**3)**2 * u_temperature1**2 + (true_temperature1**4)**2 * u_emissivity**2
    )

    u_emissivity_camera = np.sqrt(
        (1/emissivity_true)**2 * u_emissivity**2 + (emissivity_effective/emissivity_true**2)**2 * u_emissivity_true**2
    )

    u_emissivity_map1 = np.sqrt(
        ((4*img1._temperatures**3)/(emissivity_camera*true_temperature1**4))**2 * img1._uncertainties**2 + \
        (1/(emissivity_camera*true_temperature1**4))**2 * u_constant**2 + \
        ((img1._temperatures**4 - constant) / (emissivity_true * true_temperature1**4)**2)**2 * u_emissivity_camera**2 + \
        ((4*true_temperature1**3 * (img1._temperatures**4 - constant)) / (emissivity_true * true_temperature1**4)**2)**2 * u_temperature1**2
    )

    u_emissivity_map2 = np.sqrt(
        ((4*img2._temperatures**3)/(emissivity_camera*true_temperature2**4))**2 * img2._uncertainties**2 + \
        (1/(emissivity_camera*true_temperature2**4))**2 * u_constant**2 + \
        ((img2._temperatures**4 - constant) / (emissivity_true * true_temperature2**4)**2)**2 * u_emissivity_camera**2 + \
        ((4*true_temperature2**3 * (img2._temperatures**4 - constant)) / (emissivity_true * true_temperature2**4)**2)**2 * u_temperature2**2
    )

    img1.set_camera_emissivity(emissivity_map1, u_emissivity_map1)
    img2.set_camera_emissivity(emissivity_map2, u_emissivity_map2)