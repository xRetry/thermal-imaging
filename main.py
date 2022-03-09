import numpy as np
import picture

def fit_func(x, a, b, c) -> float:
    return a * np.exp(-b * x) + c

def fit_func2(x, a, b) -> float:
    return a + b * x

def run():
    img = picture.load_from_jpg(
        path='images/thermal_image.jpg',
        t_min=13,
        t_max=20,
        bar_location='bottom',
        interp='nearest',
        thermal_tolerance=0.035,
    )
    img2 = picture.load_from_tiff(
        path='images/tiff_sample.tiff',
        thermal_tolerance=0.035
    )
    # img.plot_image()
    # img.plot_temperatures()
    # img.add_line(50, 2, 50, 75)
    # img.add_rectangle(260, 136, 316, 77)
    img.add_line(416, 177, 728, 146)
    # img.add_rectangle(710, 320, 840, 258)
    # img.plot_selection()
    # img.calibrate_selection(-10, 1)
    # img.plot_temperatures()
    img.plot_selection(fit_func=fit_func2)

if __name__ == '__main__':
    run()