from util import *
import time


def scale_image_up(image_to_scale, number_of_seams_to_add, with_mask=False, use_forward_energy=False):
    original_image = np.copy(image_to_scale)
    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    original_indices = calculate_original_seam_indices(image_to_scale)

    seams_to_insert = []

    for i in range(number_of_seams_to_add):
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        dI = dx + dy

        start_time = time.time()
        if use_forward_energy:
            aggregated_energy_map = forward_energy(image_to_scale)
        else:
            aggregated_energy_map = calculate_optimal_energy_map(dI)

        path_time += time.time() - start_time

        start_time = time.time()
        local_seam = calculate_seam(aggregated_energy_map)
        original_seam = get_original_seam(local_seam, original_indices)
        seams_to_insert.append(original_seam)
        image_to_scale = remove_seam(image_to_scale, local_seam)
        original_indices = remove_seam_from_original_indices(local_seam, original_indices)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print("Seams removed: ", i, dx_time, dy_time, path_time, removal_time)

    print("Inserting seams..")
    image_to_scale = insert_seams(original_image, seams_to_insert, with_mask)

    return image_to_scale
