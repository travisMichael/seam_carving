import numpy as np


# One pitfall was figuring out how to calculate the x and y gradients

def calculate_gradients(image):
    height, width, channels = image.shape

    y_gradient = np.zeros((height, width, channels)).astype(float)
    x_gradient = np.zeros((height, width, channels)).astype(float)

    dy_0, dx_0 = np.gradient(image[:, :, 0])
    dy_1, dx_1 = np.gradient(image[:, :, 0])
    dy_2, dx_2 = np.gradient(image[:, :, 0])

    y_gradient[:, :, 0] = dy_0
    y_gradient[:, :, 0] = dy_1
    y_gradient[:, :, 0] = dy_2

    x_gradient[:, :, 0] = dx_0
    x_gradient[:, :, 0] = dx_1
    x_gradient[:, :, 0] = dx_2

    return y_gradient, x_gradient


def calculate_optimal_energy_map(energy_map):
    height, width = energy_map.shape
    y_map = np.zeros((height, width))
    y_map[0] = energy_map[0]
    m = np.zeros((height, width))

    for i in range(1, height):
        m[i] = dyanamic_programming_step(m, i, energy_map[i], energy_map[i], energy_map[i])

    return m


def dyanamic_programming_step(m, index, middle, left, right):
    _, w = m.shape
    m_minus_one = m[index - 1]
    m_left = np.roll(m_minus_one, 1)
    m_left[0] = m_minus_one[0]
    m_right = np.roll(m_minus_one, -1)
    m_right[w - 1] = m_minus_one[w - 1]

    m_prevous = np.array([m_minus_one, m_left, m_right])
    c_grouped = np.array([middle, left, right])
    m_total = m_prevous + c_grouped

    argmins = np.argmin(m_total, axis=0)
    return np.choose(argmins, m_total)


def calculate_original_seam_indices(image):
    h, w, _ = image.shape
    indices = np.zeros((h, w)).astype(int)

    for i in range(h):
        indices[i] = np.linspace(0, w-1, w).astype(int)

    return indices


def get_original_seam(local_seam, original_indices):
    new_seam = np.zeros(len(local_seam))

    for i in range(len(local_seam)):
        new_seam[i] = original_indices[i, local_seam[i]]
    return new_seam


def remove_seam_from_original_indices(optimal_seam, original_indices):
    h, w = original_indices.shape
    new_indices = np.zeros((h, w - 1))

    for i in range(h):
        new_indices[i] = np.delete(original_indices[i], optimal_seam[i])

    return new_indices


def remove_seam(temp_image, optimal_seam):
    height, width, _ = temp_image.shape
    updated_image = np.zeros((height, width - 1, 3))

    for i in range(height):
        updated_image[i, :, 0] = np.delete(temp_image[i, :, 0], [optimal_seam[i]])
        updated_image[i, :, 1] = np.delete(temp_image[i, :, 1], [optimal_seam[i]])
        updated_image[i, :, 2] = np.delete(temp_image[i, :, 2], [optimal_seam[i]])

    return updated_image


def calculate_seam(y_map):
    height, width = y_map.shape
    seam = np.zeros(height, dtype=int)
    i = height - 1
    seam[i] = np.argmin(y_map[i, :])
    choices = np.zeros(3, dtype=int)

    while i > 0:
        i = i - 1
        j = seam[i + 1]
        choices[0] = np.max([j - 1, 0])
        choices[1] = j
        choices[2] = np.min([j + 1, width - 1])
        a_min = np.argmin([y_map[i][choices[0]], y_map[i][j], y_map[i][choices[2]]])
        seam[i] = choices[a_min]

    return seam


def mask_seams(original_image, seams):

    i = 0
    while len(seams) > 0:
        next_optimal_seam = seams.pop(0)
        original_image = mask_single_seam(original_image, next_optimal_seam)
        i += 1
        if i % 10 == 0:
            print("Seams inserted: " + str(i))

    return original_image


def insert_seams_from_deleted_seam_indices(original_image, seams, with_mask):

    i = 0
    while len(seams) > 0:
        next_optimal_seam = seams.pop(0)
        original_image = insert_single_seam(original_image, next_optimal_seam, with_mask)
        seams = increment_seams(seams, next_optimal_seam)
        i += 1
        if i % 10 == 0:
            print("Seams inserted: " + str(i))

    return original_image


def mask_single_seam(temp_image, optimal_seam):
    height, width, _ = temp_image.shape

    for i in range(height):
        column = int(optimal_seam[i])
        temp_image[i, column] = np.array([0, 0, 255])

    return temp_image


def insert_single_seam(temp_image, optimal_seam, with_mask):
    height, width, _ = temp_image.shape
    new_constructed_image = np.zeros((height, width + 1, 3), dtype=float)

    for i in range(height):
        column = int(optimal_seam[i])
        if column == 0:
            neighboring_pixels_average = np.average(temp_image[i, 0:1, :], axis=0)
        else:
            neighboring_pixels_average = np.average(temp_image[i, column - 1: column + 1, :], axis=0)
        if with_mask:
            neighboring_pixels_average = np.array([0, 0, 255])
        new_constructed_image[i, column, :] = neighboring_pixels_average
        new_constructed_image[i, 0:column, :] = temp_image[i, 0:column, :]
        new_constructed_image[i, column+1:width+2, :] = temp_image[i, column:width+1, :]

    return new_constructed_image


def increment_seams(seams, seam):
    for i in range(len(seams)):
        for j in range(len(seam)):
            if round(seams[i][j]) >= round(seam[j]):
                seams[i][j] += 1

    return seams


def calculate_magnitude(image):
    squared = image * image
    sum = np.sum(squared, axis=2)
    return np.sqrt(sum)


def backward_energy(image):
    dy, dx = calculate_gradients(image)
    return calculate_magnitude(dx) + calculate_magnitude(dy)


def forward_energy(image):
    h, w, _ = image.shape
    j_plus_one = np.roll(image, -1, axis=1)
    j_minus_one = np.roll(image, 1, axis=1)
    i_minus_one = np.roll(image, 1, axis=0)

    c_U = calculate_magnitude(j_plus_one - j_minus_one)
    c_L = c_U + calculate_magnitude(i_minus_one - j_minus_one)
    c_R = c_U + calculate_magnitude(i_minus_one - j_minus_one)

    m = np.zeros((h, w))
    for i in range(1, h):
        m[i] = dyanamic_programming_step(m, i, c_U[i], c_L[i], c_R[i])

    return m
