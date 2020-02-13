import numpy as np
import cv2
from skimage import filters, color


# One pitfall was figuring out how to calculate the x and y gradients

def y_gradient_magnitudes(image):
    height, width, channels = image.shape

    a = np.delete(image, obj=[0, 1], axis=0)
    b = np.delete(image, obj=[height - 1, height - 2], axis=0)

    diff = (a - b)
    squared = diff * diff
    sum = np.sum(squared, axis=2)
    root = np.sqrt(sum)

    first_row_diff = abs(image[0] - image[1])
    last_row_diff = abs(image[height - 1] - image[height - 2])

    first_row_squared = first_row_diff * first_row_diff
    first_row_sum = np.sum(first_row_squared, axis=1)
    first_row_root = np.sqrt(first_row_sum)

    last_row_squared = last_row_diff * last_row_diff
    last_row_sum = np.sum(last_row_squared, axis=1)
    last_row_root = np.sqrt(last_row_sum)

    c = np.vstack((first_row_root, root))
    dx = np.vstack((c, last_row_root))

    return dx


def x_gradient_magnitudes(image):
    height, width, channels = image.shape

    a = np.delete(image, obj=[0, 1], axis=1)
    b = np.delete(image, obj=[width - 1, width - 2], axis=1)

    diff = abs(a - b)
    squared = diff * diff
    sum = np.sum(squared, axis=2)
    root = np.sqrt(sum)

    first_column_diff = abs(image[:, 0, :] - image[:, 1, :])
    last_column_diff = abs(image[:, width - 1, :] - image[:, width - 2, :])

    first_column_squared = first_column_diff * first_column_diff
    first_column_sum = np.sum(first_column_squared, axis=1)
    first_column_root = np.sqrt(first_column_sum)

    last_column_squared = last_column_diff * last_column_diff
    last_column_sum = np.sum(last_column_squared, axis=1)
    last_column_root = np.sqrt(last_column_sum)

    first_column = np.expand_dims(first_column_root, axis=1)
    last_column = np.expand_dims(last_column_root, axis=1)

    c = np.hstack((first_column, root))
    dy = np.hstack((c, last_column))

    return dy


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


def insert_seams(original_image, seams):

    i = 0
    while len(seams) > 0:
        next_optimal_seam = seams.pop(0)
        original_image = insert_single_seam(original_image, next_optimal_seam, i)
        seams = increment_seams(seams, next_optimal_seam)
        i += 1
        print(i)

    return original_image


def insert_single_seam(temp_image, optimal_seam):
    height, width, _ = temp_image.shape
    new_constructed_image = np.zeros((height, width + 1, 3), dtype=float)

    for i in range(height):
        for j in range(width+1):
            column = optimal_seam[i]
            if j == column+1:
                if j == 0:
                    neighboring_pixels_average = np.average(temp_image[i, j:1, :], axis=0)
                else:
                    neighboring_pixels_average = np.average(temp_image[i, j - 1: j + 1, :], axis=0)
                new_constructed_image[i, j, :] = neighboring_pixels_average

            elif j > column:
                new_constructed_image[i, j, :] = temp_image[i, j-1, :]
            else:
                new_constructed_image[i, j, :] = temp_image[i, j, :]

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
