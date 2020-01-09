from PIL import Image
import numpy as np
import time
import cv2

# One pitfall was figuring out how to calculate the x and y gradients


def x_gradient_magnitudes(image):
    height, width, channels = image.shape

    a = np.delete(image, obj=[0, 1], axis=0)
    b = np.delete(image, obj=[height - 1, height - 2], axis=0)

    diff = (a - b)

    squared = diff * diff

    sum = np.sum(squared, axis=2)

    root = np.sqrt(sum)

    first_row = np.sqrt(np.sum(image[0]*image[0], axis=1))
    final_row = np.sqrt(np.sum(image[height - 1]*image[height - 1], axis=1))

    c = np.vstack((first_row, root))
    dx = np.vstack((c, final_row))

    return dx


def y_gradient_magnitudes(image):
    height, width, channels = image.shape

    a = np.delete(image, obj=[0, 1], axis=1)
    b = np.delete(image, obj=[width - 1, width - 2], axis=1)

    diff = abs(a - b)

    squared = diff * diff

    sum = np.sum(squared, axis=2)

    root = np.sqrt(sum)

    first_column = np.expand_dims(np.sqrt(np.sum(image[:, 0]*image[:, 0], axis=1)), axis=1)
    last_colum = np.expand_dims(np.sqrt(np.sum(image[:, width - 1]*image[:, width - 1], axis=1)), axis=1)

    c = np.hstack((first_column, root))
    dy = np.hstack((c, last_colum))

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


def calculate_x_path_map(energy_map):
    width, height = energy_map.shape
    x_map = np.zeros((width, height))

    for i in range(height):
        for j in range(width):
            if i == 0:
                x_map[j][i] = energy_map[j][i]
            else:
                upper = np.max([i - 1, 0])
                lower = np.min([i + 1, height - 1])
                x_map[j][i] = energy_map[j][i] + np.min([x_map[j-1, upper], x_map[j-1, i], x_map[j-1, lower]])

    return x_map


def calculate_energy(image):
    b_0 = image[:,:,0]
    c_0 = image[:,:,0]
    d_0 = image[:,:,0]

    b_energy_0 = np.absolute(cv2.Scharr(b_0, -1, 1, 0)) + np.absolute(cv2.Scharr(b_0, -1, 0, 1))
    c_energy_0 = np.absolute(cv2.Scharr(c_0, -1, 1, 0)) + np.absolute(cv2.Scharr(c_0, -1, 0, 1))
    d_energy_0 = np.absolute(cv2.Scharr(d_0, -1, 1, 0)) + np.absolute(cv2.Scharr(d_0, -1, 0, 1))

    return b_energy_0 + c_energy_0 + d_energy_0


def delete_seam(image, seam_idx):
    height, width, _ = image.shape
    output = np.zeros((height, width - 1, 3), dtype='uint8')

    for row in range(height):
        col = seam_idx[row]
        output[row, :, 0] = np.delete(image[row, :, 0], [col])
        output[row, :, 1] = np.delete(image[row, :, 1], [col])
        output[row, :, 2] = np.delete(image[row, :, 2], [col])

    return output


def calculate_seam(image, y_map):
    height, width, _ = image.shape
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


def scale_image(image_to_scale):

    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    # 71 + 100 + 100
    for i in range(350):
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
        y_map = calculate_y_path_map(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        seam = calculate_seam(image_to_scale, y_map)
        image_to_scale = delete_seam(image_to_scale, seam)
        # image_to_scale = remove_vertical_seam(image_to_scale, y_map)
        removal_time += time.time() - start_time

        if i % 10 == 0:
            image_new_image = Image.fromarray(image_to_scale)
            image_new_image.save("pics/islands_" + str(i) + ".png")
            print(i, dx_time, dy_time, path_time, removal_time)

    return image_to_scale


img = Image.open('island_original.png')

img_array = np.asarray(img)

print(img.format)

new_image = scale_image(img_array)

image_new_image = Image.fromarray(new_image)

image_new_image.show()

print()