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
    # img.plot_selection()
    return img

def ref2_jpg():
    img = picture.load_from_jpg(
        path='images/img_thermal_12-27-53_55-8.jpg',
        t_min=25,
        t_max=45,
        bar_location='left',
        interp='linear',
        thermal_tolerance=0.035,
    )
    img.add_rectangle(345, 209, 379, 276)
    img.add_rectangle(364, 147, 391, 197)
    # img.plot_selection()
    return img


def run():
    img1 = ref1_tiff()
    img2 = ref2_tiff()
    # img1.plot_selection(title='Aufnahme 1 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    # img2.plot_selection(title='Aufnahme 2 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    core.determine_emissivity(img1, img2, 80.7, 55.8, emissivity_true=0.95)
    img1.clear_selection()
    img2.clear_selection()
    img1.add_rectangle(71, 13, 110, 117)
    img2.add_rectangle(43, 64, 100, 126)
    img1.plot_emissivity(title='Aufnahme 1 - Emissionsgrad', cbar_label='Emissionsgrad')
    img2.plot_emissivity(title='Aufnahme 2 - Emissionsgrad', cbar_label='Emissionsgrad')

def run2():
    img1 = ref1_jpg()
    img2 = ref2_jpg()
    # img1.plot_selection(title='Aufnahme 1 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    # img2.plot_selection(title='Aufnahme 2 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    core.determine_emissivity(img1, img2, 80.7, 55.8, emissivity_true=0.95)
    img1.clear_selection()
    img2.clear_selection()
    img1.add_rectangle(417, 73, 721, 738)
    img2.add_rectangle(258, 386, 652, 770)
    img1.plot_emissivity(title='Aufnahme 1 - Emissionsgrad', cbar_label='Emissionsgrad')
    img2.plot_emissivity(title='Aufnahme 2 - Emissionsgrad', cbar_label='Emissionsgrad')


if __name__ == '__main__':
    # run()
    run2()