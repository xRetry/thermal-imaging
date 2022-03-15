import numpy as np
import picture
import core
import plotting

def fit_func(x, a, b, c) -> float:
    return a * np.exp(-b * x) + c

def fit_func2(x, a, b) -> float:
    return a + b * x


def ref1_tiff():
    img = picture.load_from_tiff(
        path='images/img_thermal_12-16-53_80-7.tiff',
        thermal_tolerance=0.07
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
        thermal_tolerance=0.07
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
        thermal_tolerance=0.07,
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
        thermal_tolerance=0.07,
    )
    img.add_rectangle(345, 209, 379, 276)
    img.add_rectangle(364, 147, 391, 197)
    # img.plot_selection()
    return img

def black_tiff():
    img = picture.load_from_tiff(
        path='images\img_thermo_12-29-30_55-7.tiff',
        thermal_tolerance=0.07
    )
    img.add_rectangle(47, 30, 140, 123)
    # img.add_rectangle(57, 24, 62, 34)
    # img.plot_selection()
    # img = img.get_selection()
    return img 


def run_tiff():
    img1 = ref1_tiff()
    img2 = ref2_tiff()
    img1.plot_selection(title='TIFF Aufnahme 1 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    img2.plot_selection(title='TIFF Aufnahme 2 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    core.determine_emissivity(img1, img2, 80.7, 55.8, emissivity_true=0.95)
    img1.clear_selection()
    img2.clear_selection()
    img1.add_rectangle(71, 13, 110, 117)
    img2.add_rectangle(43, 64, 100, 126)
    e1, std1 = img1.plot_emissivity(title='TIFF Aufnahme 1 - Emissionsgrad', cbar_label='Emissionsgrad')
    e2, std2 = img2.plot_emissivity(title='TIFF Aufnahme 2 - Emissionsgrad', cbar_label='Emissionsgrad')
    return [e1, e2], [std1, std2]

def run_jpg():
    img1 = ref1_jpg()
    img2 = ref2_jpg()
    # img_b = black_tiff()
    img1.plot_selection(title='JPEG Aufnahme 1 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    img2.plot_selection(title='JPEG Aufnahme 2 - Auswahl des Referenzbereichs', cbar_label='Temperatur [°C]')
    core.determine_emissivity(img1, img2, 80.7, 55.8, 
        emissivity_true=0.95, 
        u_emissivity_true=0.05/np.sqrt(3),
        u_temperature1=0.5/np.sqrt(3),
        u_temperature2=0.5/np.sqrt(3),
        # additional_images=[(img_b, 55.7, 0.5/np.sqrt(3))]
    )
    img1.clear_selection()
    img2.clear_selection()
    img1.add_rectangle(417, 73, 721, 738)
    img2.add_rectangle(258, 386, 652, 770)
    e1, std1 = img1.plot_emissivity(title='JPEG Aufnahme 1 - Emissionsgrad', cbar_label='Emissionsgrad')
    e2, std2 = img2.plot_emissivity(title='JPEG Aufnahme 2 - Emissionsgrad', cbar_label='Emissionsgrad')
    # e3, std3 = img_b.plot_emissivity(title='Schwarze Fläche - Emissionsgrad', cbar_label='Emissionsgrad')
    return [e1, e2], [std1, std2]

def plot():
    e_tiff, std_tiff = run_tiff()
    e_jpg, std_jpg = run_jpg()
    plotting.plot_emissivities(
        [e_tiff[0], e_jpg[0], e_tiff[1], e_jpg[1]], 
        [std_tiff[0], std_jpg[0], std_tiff[1], std_jpg[1]],
        ['TIFF Aufnahme 1', 'JPEG Aufnahme 1', 'TIFF Aufnahme 2', 'JPEG Aufnahme 2']
    )

if __name__ == '__main__':
    plot()