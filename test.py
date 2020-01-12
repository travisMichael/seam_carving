from PIL import Image
import numpy as np
import time
import cv2
from algo import SeamCarver


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
    c = np.zeros((4, width))
    y_map[0] = energy_map[0]

    # kind of messy, but the vectorized code allows us to remove the nested for loop from the dynamic programming step
    for i in range(1, height):
        left_shift = np.delete(y_map[i-1], 0)
        right_shift = np.delete(y_map[i-1], width - 1)

        left_shift = np.concatenate((left_shift, [np.inf]))
        right_shift = np.concatenate(([np.inf], right_shift))

        # we want to shift left
        c[0] = np.less_equal(left_shift, y_map[i-1])
        c_0_i = np.where(c[0] == 1)
        zero_not = np.logical_not(c[0])

        # we want to shift right
        c[1] = np.less_equal(right_shift, y_map[i-1])
        c_1_i = np.where(c[1] == 1)
        one_not = np.logical_not(c[1])

        np.put(y_map[i], c_1_i, right_shift[c_1_i])
        np.put(y_map[i], c_0_i, left_shift[c_0_i])
        # indices to keep original values
        not_and = np.logical_and(zero_not, one_not)
        not_and_i = np.where(not_and == True)

        # if both shifting is true, then we check which shift is better
        c[2] = np.less_equal(left_shift, right_shift)
        c[3] = np.less_equal(right_shift, left_shift)

        intermediate = np.logical_and(c[0], c[1])
        override_left = np.logical_and(intermediate, c[2])
        override_left_i = np.where(override_left == True)

        intermediate = np.logical_and(c[0], c[1])
        override_right = np.logical_and(intermediate, c[3])
        override_right_i = np.where(override_right == True)

        np.put(y_map[i], override_left_i, left_shift[override_left_i])
        np.put(y_map[i], override_right_i, right_shift[override_right_i])
        np.put(y_map[i], not_and_i, y_map[i-1][not_and_i])
        y_map[i] = y_map[i] + energy_map[i]

    return y_map


# def calculate_x_path_map(energy_map):
#     width, height = energy_map.shape
#     x_map = np.zeros((width, height))
#
#     for i in range(height):
#         for j in range(width):
#             if i == 0:
#                 x_map[j][i] = energy_map[j][i]
#             else:
#                 upper = np.max([i - 1, 0])
#                 lower = np.min([i + 1, height - 1])
#                 x_map[j][i] = energy_map[j][i] + np.min([x_map[j-1, upper], x_map[j-1, i], x_map[j-1, lower]])
#
#     return x_map


def delete_seam(image, seam_idx):
    height, width, _ = image.shape
    output = np.zeros((height, width - 1, 3))

    for row in range(height):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(image[row, :, 0], [col])
        output[row, :, 1] = np.delete(image[row, :, 1], [col])
        output[row, :, 2] = np.delete(image[row, :, 2], [col])

    return output


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


def scale_image_up(image_to_scale):
    original_image = np.copy(image_to_scale)
    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    # seam_carver = SeamCarver("dolphin.png", 466, 350)

    seams_to_insert = []

    for i in range(120):
        # h, w, c = image_to_scale.shape
        # seam_carver.step()
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        # dI = dx + dy
        energy_map = dx + dy

        # energy_map = calculate_energy(image_to_scale)
        # energy_map = gradient_magnitude(dI)
        start_time = time.time()
        # y_map = calculate_y_path_map(energy_map)
        y_map = calculate_optimal_energy_map(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(y_map)

        seams_to_insert.append(seam)
        image_to_scale = delete_seam(image_to_scale, seam)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print(i, dx_time, dy_time, path_time, removal_time)

    print("Inserting seams..")
    image_to_scale = insert_seams(original_image, seams_to_insert)

    return image_to_scale


def insert_seams(original_image, seams):

    while len(seams) > 0:
        next_optimal_seam = seams.pop(0)
        original_image = insert_single_seam(original_image, next_optimal_seam)
        seams = increment_seams(seams, next_optimal_seam)

    return original_image


def insert_single_seam(temp_image, optimal_seam):
    height, width, _ = temp_image.shape
    new_constructed_image = np.zeros((height, width + 1, 3), dtype='uint8')

    for i in range(height):
        column = optimal_seam[i]
        if column != 0:
            average = np.average(temp_image[i, column - 1: column + 1, :], axis=0)
            new_constructed_image[i, : column, :] = temp_image[i, : column, :]
            new_constructed_image[i, column, :] = average
            new_constructed_image[i, column + 1:, :] = temp_image[i, column:, :]
        else:
            average = np.average(temp_image[i, column: column + 2, :], axis=0)
            new_constructed_image[i, column, :] = temp_image[i, column, :]
            new_constructed_image[i, column + 1, :] = average
            new_constructed_image[i, column + 1:, :] = temp_image[i, column:, :]

    return new_constructed_image


def increment_seams(seams, seam):
    for i_seam in seams:
        for j in range(len(i_seam)):
            if i_seam[j] >= seam[j]:
                i_seam[j] += 2

    return seams


def scale_image(image_to_scale):

    seam_carver = SeamCarver("island_original.png", 466, 350)

    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    # 71 + 100 + 100
    for i in range(350):
        h, w, _ = image_to_scale.shape
        # seam_carver.step()
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        # dI = dx + dy
        energy_map = dx + dy

        # energy_map = calculate_energy(image_to_scale)

        # for j in range(h):
        #     for k in range(w):
        #         if energy_map[j][k] != seam_carver.energy_map[j][k]:
        #             print("Wrong")
        # energy_map = gradient_magnitude(dI)
        start_time = time.time()
        y_map = calculate_optimal_energy_map(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(y_map)
        image_to_scale = delete_seam(image_to_scale, seam)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print("Pixels removed: ", i, dx_time, dy_time, path_time, removal_time)

    return image_to_scale


# img = Image.open('island_original.png')
#
# img_array = cv2.imread('island_original.png').astype(np.float64)
#
# print(img.format)
#
# new_image = scale_image(img_array.astype(float))
#
# cv2.imwrite("island_result.png", new_image)


img_array = cv2.imread('dolphin.png').astype(np.float64)

new_image = scale_image_up(img_array)

cv2.imwrite("dolphin_50_result.png", new_image)

print("done")
