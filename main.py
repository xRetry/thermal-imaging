import numpy as np
import picture

def fit_func(x, a, b, c) -> float:
    return a * np.exp(-b * x) + c

def fit_func2(x, a, b) -> float:
    return a + b * x


def ref1_tiff():
    img = picture.load_from_tiff(
        path='images/img_thermal_12-16-53_80-7.tiff',
        thermal_tolerance=0.035
    )
    img.add_rectangle(47, 31, 51, 41)
    img.add_rectangle(48, 23, 53, 30)
    img.plot_selection()
    img.calibrate_selection(80.7, 1)
    # img.plot_temperatures()
    return img
 

def ref1_jpg():
    img = picture.load_from_jpg(
        path='images/img_thermal_12-16-53_80-7.jpg',
        t_min=25,
        t_max=58,
        bar_location='left',
        interp='linear',
        thermal_tolerance=0.035,
    )
    img.add_rectangle(286, 208, 333, 265)
    img.add_rectangle(304, 138, 336, 198)
    img.plot_selection()
    img.calibrate_selection(80.7, 1)
    return img
 

def run():
    # img = ref1_tiff()
    img2 = ref1_jpg()
    # img.plot_image()
    # img.plot_temperatures()
    # img.add_line(50, 2, 50, 75)
    # For 12-16-53
    # img.plot_selection(fit_func=fit_func2)

if __name__ == '__main__':
    run()