import numpy as np
import picture
import core

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
    # img.plot_selection()
    # img.plot_temperatures()
    # img = img.get_selection()
    return img

def ref2_tiff():
    img = picture.load_from_tiff(
        path='images/img_thermal_12-27-53_55-8.tiff',
        thermal_tolerance=0.035
    )
    img.add_rectangle(54, 36, 61, 47)
    img.add_rectangle(57, 24, 62, 34)
    # img.plot_selection()
    # img = img.get_selection()
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
    img1 = ref1_tiff()
    img2 = ref2_tiff()
    img1.plot_selection(title='Aufnahme 1 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    img2.plot_selection(title='Aufnahme 2 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    core.determine_emissivity(img1, img2, 80.7, 55.8, true_emissivity=0.95)
    img1.plot_emissivity(80.7, title='Aufnahme 1 - Emissionsgrad', cbar_label='Emissionsgrad')
    img2.plot_emissivity(55.8, title='Aufnahme 2 - Emissionsgrad', cbar_label='Emissionsgrad')
    img1.plot_temperatures(title='Temperatur des Würfels nach Kalibrierung', cbar_label='Temperatur [°C]')

if __name__ == '__main__':
    run()