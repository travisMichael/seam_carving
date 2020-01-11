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
    # squared = diff * diff
    sum = np.sum(diff, axis=2)
    # root = np.sqrt(sum)

    first_row_diff = abs(image[0] - image[1])
    last_row_diff = abs(image[height - 1] - image[height - 2])

    # first_row_squared = first_row_diff * first_row_diff
    first_row_sum = np.sum(first_row_diff, axis=1)
    # first_row_root = np.sqrt(first_row_sum)

    # last_row_squared = last_row_diff * last_row_diff
    last_row_sum = np.sum(last_row_diff, axis=1)
    # last_row_root = np.sqrt(last_row_sum)

    # first_row = np.sqrt(np.sum(image[0]*image[0], axis=1))
    # final_row = np.sqrt(np.sum(image[height - 1]*image[height - 1], axis=1))

    c = np.vstack((first_row_sum, sum))
    dx = np.vstack((c, last_row_sum))

    return dx


def x_gradient_magnitudes(image):
    height, width, channels = image.shape

    a = np.delete(image, obj=[0, 1], axis=1)
    b = np.delete(image, obj=[width - 1, width - 2], axis=1)

    diff = abs(a - b)
    # squared = diff * diff
    sum = np.sum(diff, axis=2)
    # root = np.sqrt(sum)

    first_column_diff = abs(image[:, 0, :] - image[:, 1, :])
    last_column_diff = abs(image[:, width - 1, :] - image[:, width - 2, :])

    # first_column_squared = first_column_diff * first_column_diff
    first_column_sum = np.sum(first_column_diff, axis=1)
    # first_column_root = np.sqrt(first_column_sum)

    # last_column_squared = last_column_diff * last_column_diff
    last_column_sum = np.sum(last_column_diff, axis=1)
    # last_column_root = np.sqrt(last_column_sum)

    first_column = np.expand_dims(first_column_sum, axis=1)
    last_column = np.expand_dims(last_column_sum, axis=1)

    c = np.hstack((first_column, sum))
    dy = np.hstack((c, last_column))

    return dy


def calculate_y_path_map(energy_map):
    height, width = energy_map.shape
    y_map = np.zeros((height, width))
    c = np.zeros((4, width))
    y_map[0] = energy_map[0]

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


def cumulative_map_backward(energy_map):
    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output

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


def calculate_energy(image):
    b, g, r = cv2.split(image)
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy
    # b_0 = image[:,:,0]
    # c_0 = image[:,:,1]
    # d_0 = image[:,:,2]
    #
    # b_energy_0 = np.absolute(cv2.Scharr(b_0, -1, 1, 0)) + np.absolute(cv2.Scharr(b_0, -1, 0, 1))
    # c_energy_0 = np.absolute(cv2.Scharr(c_0, -1, 1, 0)) + np.absolute(cv2.Scharr(c_0, -1, 0, 1))
    # d_energy_0 = np.absolute(cv2.Scharr(d_0, -1, 1, 0)) + np.absolute(cv2.Scharr(d_0, -1, 0, 1))

    # return b_energy_0 + c_energy_0 + d_energy_0



def delete_seam(image, seam_idx):
    height, width, _ = image.shape
    output = np.zeros((height, width - 1, 3))

    for row in range(height):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(image[row, :, 0], [col])
        output[row, :, 1] = np.delete(image[row, :, 1], [col])
        output[row, :, 2] = np.delete(image[row, :, 2], [col])

    return output


# def calculate_seam(image, y_map):
#     height, width, _ = image.shape
#     seam = np.zeros(height, dtype=int)
#     i = height - 1
#     seam[i] = np.argmin(y_map[i, :])
#     choices = np.zeros(3, dtype=int)
#
#     while i > 0:
#         i = i - 1
#         j = seam[i + 1]
#         choices[0] = np.max([j - 1, 0])
#         choices[1] = j
#         choices[2] = np.min([j + 1, width - 1])
#         a_min = np.argmin([y_map[i][choices[0]], y_map[i][j], y_map[i][choices[2]]])
#         seam[i] = choices[a_min]
#
#     return seam


def calculate_seam(cumulative_map):
    m, n = cumulative_map.shape
    output = np.zeros((m,), dtype=np.uint32)
    output[-1] = np.argmin(cumulative_map[-1])
    for row in range(m - 2, -1, -1):
        prv_x = output[row + 1]
        if prv_x == 0:
            output[row] = np.argmin(cumulative_map[row, : 2])
        else:
            output[row] = np.argmin(cumulative_map[row, prv_x - 1: min(prv_x + 2, n - 1)]) + prv_x - 1
    return output

# def remove_vertical_seam(image, y_map):
#     height, width, _ = image.shape
#     to_be_removed_index = 0
#     to_be_removed = np.zeros(height * 3)
#     choices = np.zeros(3, dtype=int)
#
#     i = height - 1
#     j = np.argmin(y_map[i, :])
#
#     start_index = 3 * width * i + j * 3
#     to_be_removed[to_be_removed_index] = start_index
#     to_be_removed[to_be_removed_index + 1] = start_index + 1
#     to_be_removed[to_be_removed_index + 2] = start_index + 2
#     to_be_removed_index += 3
#
#     while i > 0:
#         i = i - 1
#         choices[0] = np.max([j - 1, 0])
#         choices[1] = j
#         choices[2] = np.min([j + 1, width - 1])
#         a_min = np.argmin([y_map[i][choices[0]], y_map[i][j], y_map[i][choices[2]]])
#         j = choices[a_min]
#         start_index = 3 * width * i + j * 3
#         to_be_removed[to_be_removed_index] = start_index
#         to_be_removed[to_be_removed_index + 1] = start_index + 1
#         to_be_removed[to_be_removed_index + 2] = start_index + 2
#         to_be_removed_index += 3
#
#     intermediate_image = np.delete(image, to_be_removed, None)
#
#     new_image = intermediate_image.reshape((height, width - 1, 3))
#
#     return new_image

def scale_image_up(image_to_scale):
    original_image = np.copy(image_to_scale)
    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    seam_carver = SeamCarver("dolphin.png", 466, 350)

    seams_to_insert = []

    for i in range(120):
        h, w, c = image_to_scale.shape
        seam_carver.step()
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        # dI = dx + dy
        # energy_map = dx + dy

        energy_map = calculate_energy(image_to_scale)
        # energy_map = gradient_magnitude(dI)
        start_time = time.time()
        # y_map = calculate_y_path_map(energy_map)
        y_map = cumulative_map_backward(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(y_map)
        for j in range(w):
            if y_map[h-1][j] != seam_carver.cumulative_map[h-1][j]:
                print("j")

        for j in range(len(seam)):
            if seam[j] != seam_carver.seam_idx[j]:
                print("j")

        seams_to_insert.append(seam)
        image_to_scale = delete_seam(image_to_scale, seam)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print(i, dx_time, dy_time, path_time, removal_time)

    print("Inserting seams..")
    image_to_scale = insert_seams(original_image, seams_to_insert)
    cv2.imwrite("dolphin_final.png", image_to_scale)

    return image_to_scale


def insert_seams(original_image, seams):

    while len(seams) > 0:
        seam = seams.pop(0)
        original_image = insert_single_seam(original_image, seam)
        seams = increment_seam_indices(seams, seam)

    return original_image


def insert_single_seam(image, seam):
    m, n, _ = image.shape
    output = np.zeros((m, n + 1, 3), dtype='uint8')
    for row in range(m):
        col = seam[row]
        for ch in range(3):
            if col == 0:
                p = np.average(image[row, col: col + 2, ch])
                output[row, col, ch] = image[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = image[row, col:, ch]
            else:
                p = np.average(image[row, col - 1: col + 1, ch])
                output[row, : col, ch] = image[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = image[row, col:, ch]

    return output


# def increment_seam_indices(seams, seam):
#     for seam_i in seams:
#         for j in range(len(seam_i)):
#             if seam_i[j] >= seam[j]:
#                 seam_i[j] += 2
#
#     return seams

def increment_seam_indices(remaining_seams, current_seam):
    output = []
    for seam in remaining_seams:
        seam[np.where(seam >= current_seam)] += 2
        output.append(seam)
    return output


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
        y_map = calculate_y_path_map(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(y_map)
        image_to_scale = delete_seam(image_to_scale, seam)
        # image_to_scale = remove_vertical_seam(image_to_scale, y_map)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            print(i, dx_time, dy_time, path_time, removal_time)
            cv2.imwrite("pics/island_" + str(i) + ".png", image_to_scale)

    return image_to_scale


img = Image.open('island_original.png')

img_array = cv2.imread('island_original.png').astype(np.float64)

print(img.format)

new_image = scale_image(img_array.astype(float))


# img_array = cv2.imread('dolphin.png').astype(np.float64)
#
# new_image = scale_image_up(img_array)
#
# image_new_image = Image.fromarray(new_image)
#
# image_new_image.show()

print()