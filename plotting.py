from typing import Callable, Optional, Tuple
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from thermal import _get_colors


def get_colormap(image: np.ndarray, bar_location: str = 'bottom'):
    colors = _get_colors(image, bar_location) 
    cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, N=len(colors))
    return cm


def plot_image(image: np.ndarray, colormap=plt.get_cmap('turbo'), line_coords=None, point_coords=None, rect_coords=None, title: str = None, cbar_label = None):
    if colormap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=colormap)

    if point_coords is not None:
        plt.scatter(point_coords[:, 0], point_coords[:, 1], marker='x', c='m')

    if line_coords is not None:
        if not isinstance(line_coords, list):
            line_coords = [line_coords]
        for x_line, y_line in line_coords:
            plt.plot(x_line, y_line, color='m')
    
    if rect_coords is not None:
        if not isinstance(rect_coords, list):
            rect_coords = [rect_coords]
        for x_rect, y_rect in rect_coords:
            plt.plot(x_rect, y_rect, color='lightgrey')

    if cbar_label is not None:
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(cbar_label)
        cbar.ax.get_yaxis().set_ticks(np.round(np.linspace(image.min(), image.max(), 7), 2))

    if title is not None:
        plt.title(title)
    
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.tight_layout()
    plt.show()


def plot_line(temperatures: np.ndarray, sdu: float = None, title: str = None, fit_config: Optional[Tuple[Callable, np.ndarray, np.ndarray]] = None):
    x = np.arange(len(temperatures))
    plt.plot(x, temperatures)
    if sdu is not None:
        lower = temperatures - sdu * 2        
        upper = temperatures + sdu * 2
        plt.fill_between(x, lower, upper, alpha=0.2)
    if fit_config is not None:
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = [fit_config[0](x_, *fit_config[1]) for x_ in x_fit]
        y_lower = [fit_config[0](x_, *(fit_config[1] - 2*fit_config[2])) for x_ in x_fit]
        y_upper = [fit_config[0](x_, *(fit_config[1] + 2*fit_config[2])) for x_ in x_fit]
        plt.plot(x_fit, y_fit)
        plt.fill_between(x_fit, y_lower, y_upper, alpha=0.5)

    if title is not None:
        plt.title(title)
    plt.xlabel('Distanz')
    plt.ylabel('Temperatur [°C]')
    plt.show()


def plot_histogram(temperatures: np.ndarray, bins: int = 50, title: str = None, fit_normal: bool = True):
    plt.hist(temperatures.flatten(), bins=bins, density=True)
    if fit_normal:
        mean = np.mean(temperatures)
        std = np.std(temperatures.flatten())
        t = np.linspace(mean-3*std, temperatures.max(), 200)
        y = scipy.stats.norm.pdf(t, loc=mean, scale=std)
        plt.plot(t, y)       
    if title is not None:
        plt.title(title)
    plt.xlabel('Temperatur [°C]')
    plt.yticks([])
    plt.show()