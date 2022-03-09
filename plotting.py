import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from thermal import _get_colors


def get_colormap(image: np.ndarray, bar_location: str = 'bottom'):
    colors = _get_colors(image, bar_location) 
    cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, N=len(colors))
    return cm


def plot_image(image: np.ndarray, colormap=None, line_coords=None, point_coords=None, rect_coords=None, title: str = None):
    if colormap is None:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=colormap)

    if point_coords is not None:
        plt.scatter(point_coords[:, 0], point_coords[:, 1], marker='x', c='r')

    if line_coords is not None:
        if not isinstance(line_coords, list):
            line_coords = [line_coords]
        for x_line, y_line in line_coords:
            plt.plot(x_line, y_line, color='r')
    
    if rect_coords is not None:
        if not isinstance(rect_coords, list):
            rect_coords = [rect_coords]
        for x_rect, y_rect in rect_coords:
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
        lower = [temperatures[i]-tolerance[i] for i in range(len(temperatures))]
        upper = [temperatures[i]+tolerance[i] for i in range(len(temperatures))]
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