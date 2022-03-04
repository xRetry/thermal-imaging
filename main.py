import numpy as np
import picture, temperature, plotting

def run():
    img = picture.load_image('images/thermal_image.jpg')
    temps = temperature.convert_to_temperature(img, 13, 20)
    point = np.array([[100, 500]])
    line_points = np.array([[200, 500], [400, 900]])
    rect_points = np.array([[600, 400], [500, 700]])
    _, _, z_line = temperature.select_line(temps, line_points)
    # plotting.plot_image(img)
    # plotting.plot_image(temps, line_points=line_points, point_points=point, rect_points=rect_points)
    plotting.plot_line(z_line, tolerance=0.07)
    # temp_slice = temperature.select_rectangle(temps, rect_points)
    # plotting.plot_image(temp_slice)
    # plotting.plot_distribution(temp_slice)


if __name__ == '__main__':
    run()