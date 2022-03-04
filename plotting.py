import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from temperature import _get_rect_coords, _get_colors, _get_line_coords


def get_colormap(image: np.ndarray, bar_location: str = 'bottom'):
    colors = _get_colors(image, bar_location) 
    cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, N=len(colors))
    return cm


def plot_image(image: np.ndarray, colormap=None, line_points=None, point_points=None, rect_points=None, title: str = None):
    if colormap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=colormap)
    if point_points is not None:
        plt.scatter(point_points[:, 0], point_points[:, 1], marker='x', c='r')
    if line_points is not None:
        x_line, y_line = _get_line_coords(line_points)
        plt.plot(x_line, y_line, color='r')
    if rect_points is not None:
        x_rect, y_rect = _get_rect_coords(rect_points)
        plt.plot(x_rect, y_rect, color='r')
    if title is not None:
        plt.title(title)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.show()


def plot_line(temperatures: np.ndarray, tolerance: float = None, title: str = None):
    x = np.arange(len(temperatures))
    plt.plot(x, temperatures)
    if tolerance is not None:
        lower = [t-tolerance for t in temperatures]
        upper = [t+tolerance for t in temperatures]
        plt.fill_between(x, lower, upper, alpha=0.2)
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


def plot_distribution(temperatures: np.ndarray, standard_uncertainty: float = 0):
    mean = np.mean(temperatures)
    std = np.std(temperatures.flatten())
    t = np.linspace(mean-3*std, temperatures.max(), 200)
    y = scipy.stats.norm.pdf(t, loc=mean, scale=std)
    plt.plot(t, y)
    plt.show()