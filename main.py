import numpy as np
import picture

def run():
    img = picture.load_from_jpg(
        path='images/thermal_image.jpg',
        t_min=13,
        t_max=20,
        bar_location='bottom',
        interp='nearest',
        thermal_tolerance=0.07,
    )
    # img.plot_image()
    # img.plot_temperatures()
    img.add_line(416, 177, 728, 146)
    img.add_rectangle(710, 320, 840, 258)
    img.plot_selection()
    

if __name__ == '__main__':
    run()