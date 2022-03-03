import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from temperature import get_square_coords, get_colors


def get_colormap(image: np.ndarray, bar_location: str = 'bottom'):
    colors = get_colors(image, bar_location) 
    cm = LinearSegmentedColormap.from_list('', np.array(colors) / 255, N=len(colors))
    return cm


def plot_image(image: np.ndarray):
    plt.imshow(image)
    plt.show()


def plot_temperatures(temperatures: np.ndarray, colormap=plt.cm.get_cmap('inferno').reversed(), point_ll=None, point_ur=None):
    plt.imshow(temperatures, cmap=colormap)
    if point_ll is not None and point_ur is None:
        plt.scatter(*point_ll, marker='x', c='r')
    if point_ll is not None and point_ur is not None:
        x_rect, y_rect = get_square_coords(point_ll, point_ur)
        plt.plot(x_rect, y_rect, color='r')
    plt.show()