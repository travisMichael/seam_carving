from util import *
import time


def scale_image(image_to_scale, number_of_pixels_to_remove):

    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    for i in range(number_of_pixels_to_remove):
        h, w, _ = image_to_scale.shape
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        # dI = dx + dy
        dI = dx + dy

        start_time = time.time()
        # aggregated_energy_map = calculate_optimal_energy_map(dI)
        aggregated_energy_map = forward_energy(image_to_scale)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(aggregated_energy_map)
        image_to_scale = remove_seam(image_to_scale, seam)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print("Seams removed: ", i, dx_time, dy_time, path_time, removal_time)

    return image_to_scale
