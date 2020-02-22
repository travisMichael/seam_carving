from util import *


def insert_seams(image_to_scale, number_of_seams_to_add, with_mask=False, use_forward_energy=False):
    original_image = np.copy(image_to_scale)

    original_indices = calculate_original_seam_indices(image_to_scale)

    seams_to_insert = []

    for i in range(number_of_seams_to_add):

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

    print("Inserting seams..")
    image_to_scale = insert_seams_from_deleted_seam_indices(original_image, seams_to_insert, with_mask)

    return image_to_scale
