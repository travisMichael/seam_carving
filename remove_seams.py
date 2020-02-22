from util import *


def remove_seams(image_to_scale, number_of_pixels_to_remove, with_mask=False, use_forward_energy=False):

    original_image = np.copy(image_to_scale)
    original_indices = calculate_original_seam_indices(image_to_scale)
    seams_to_insert = []

    for i in range(number_of_pixels_to_remove):
        h, w, _ = image_to_scale.shape

        if use_forward_energy:
            aggregated_energy_map = forward_energy(image_to_scale)
        else:
            energy = backward_energy(image_to_scale)
            aggregated_energy_map = calculate_optimal_energy_map(energy)

        local_seam = calculate_seam(aggregated_energy_map)
        original_seam = get_original_seam(local_seam, original_indices)
        seams_to_insert.append(original_seam)
        image_to_scale = remove_seam(image_to_scale, local_seam)
        original_indices = remove_seam_from_original_indices(local_seam, original_indices)

        if i % 10 == 0:
            print("Seams removed: ", i)

    if with_mask:
        image_to_scale = mask_seams(original_image, seams_to_insert)

    return image_to_scale
