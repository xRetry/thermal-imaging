import numpy as np
import picture, temperature, plotting

def run():
    img = picture.load_image('images/thermal_image.jpg')
    temps = temperature.convert_to_temperature(img, 13, 20)
    plotting.plot_image(img)
    plotting.plot_temperatures(temps)


if __name__ == '__main__':
    run()