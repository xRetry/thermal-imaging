import numpy as np
import picture, thermal, plotting, core

def run():
    img = core.ThermalImage(
        path='images/thermal_image.jpg',
        t_min=13,
        t_max=20,
        bar_location='bottom',
        interp='nearest',
        thermal_tolerance=0.07,
        colormap='original'
    )
    img.plot_image()
    img.plot_temperatures()
    img.plot_line(416, 177, 728, 146)
    img.plot_rectangle(710, 320, 840, 258)
    


def run_old():
    img = picture.load_image('images/thermal_image.jpg')
    temps = thermal.convert_to_temperature(img, 13, 20)
    point = np.array([[100, 500]])
    line_points = np.array([[416, 177], [728, 146]])
    rect_points = np.array([[710, 320], [840, 258]])
    _, _, z_line = picture.select_line(temps, line_points)
    plotting.plot_image(img, title='Originales Bild')
    cm = plotting.get_colormap(img)
    plotting.plot_image(temps, title='Interpoliertes Bild', colormap=cm)
    plotting.plot_image(temps, title='Auswahlmöglichkeiten', colormap=cm, line_points=line_points, point_points=point, rect_points=rect_points)
    # plotting.plot_image(temps, title='Linienauswahl', colormap=cm, line_points=)
    plotting.plot_line(z_line, tolerance=0.07/np.sqrt(3)*2, title='Temperatur entlang Linie')
    temp_slice = picture.select_rectangle(temps, rect_points)
    plotting.plot_image(temp_slice, title='Detailansicht Rechtecksauswahl', colormap=cm)
    plotting.plot_histogram(temp_slice, bins=30, title='Temperaturverteilung in Rechtecksauswahl')
    # plotting.plot_distribution(temp_slice)


if __name__ == '__main__':
    run()