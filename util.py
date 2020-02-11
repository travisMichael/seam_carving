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

    while len(seams) > 0:
        next_optimal_seam = seams.pop(0)
        original_image = insert_single_seam(original_image, next_optimal_seam)
        seams = increment_seams(seams, next_optimal_seam)

    return original_image


def insert_single_seam(temp_image, optimal_seam, with_mask=False):
    height, width, _ = temp_image.shape
    new_constructed_image = np.zeros((height, width + 1, 3), dtype=float)

    for i in range(height):
        for j in range(width+1):
            column = optimal_seam[i]
            if j == column:
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

    # for i in range(height):
    #     column = optimal_seam[i]
    #     if column != 0:
    #         # np.array([0, 0, 255])
    #         neighboring_pixels_average = np.average(temp_image[i, column - 1: column + 1, :], axis=0)
    #         new_constructed_image[i, : column, :] = temp_image[i, : column, :]
    #         new_constructed_image[i, column, :] = neighboring_pixels_average
    #         new_constructed_image[i, column + 1:, :] = temp_image[i, column:, :]
    #     else:
    #         neighboring_pixels_average = np.average(temp_image[i, column: column + 2, :], axis=0)
    #         new_constructed_image[i, column, :] = temp_image[i, column, :]
    #         new_constructed_image[i, column + 1, :] = neighboring_pixels_average
    #         new_constructed_image[i, column + 1:, :] = temp_image[i, column:, :]
    #
    # return new_constructed_image


def increment_seams(seams, seam):
    for i_seam in seams:
        for j in range(len(i_seam)):
            if i_seam[j] >= seam[j]:
                i_seam[j] += 2

    return seams


def calc_energy_map(image):
        b, g, r = cv2.split(image)
        b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + np.absolute(cv2.Scharr(b, -1, 0, 1))
        g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + np.absolute(cv2.Scharr(g, -1, 0, 1))
        r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + np.absolute(cv2.Scharr(r, -1, 0, 1))
        return b_energy + g_energy + r_energy


def calculate_magnitude(image):
    squared = image * image
    sum = np.sum(squared, axis=2)
    return np.sqrt(sum)


# def forward_energy(image, e_2=None):
#     img_copy = np.copy(image)
#     h, w, _ = image.shape
#     if e_2 == None:
#         e_2 = np.zeros((h, w))
#     I_plus_one = np.roll(img_copy, 1, axis=0)
#     J_plus_one = np.roll(img_copy, 1, axis=1)
#     I_minus_one = np.roll(img_copy, -1, axis=0)
#     J_minus_one = np.roll(img_copy, -1, axis=1)
#
#     I_plus_one[0, :] = image[0, :]
#     I_plus_one[h-1, :] = image[h-1, :]
#
#     I_minus_one[0, :] = image[0, :]
#     I_minus_one[h-1, :] = image[h-1, :]
#
#     J_plus_one[:, 0] = image[:, 0]
#     J_plus_one[:, w-1] = image[:, w-1]
#
#     J_minus_one[:, 0] = image[:, 0]
#     J_minus_one[:, w-1] = image[:, w-1]
#
#     temp = calculate_magnitude(J_plus_one - J_minus_one)
#     C_L = temp + calculate_magnitude(I_minus_one - J_minus_one)
#     C_U = temp
#     C_R = temp + calculate_magnitude(I_minus_one - J_plus_one)
#
#     M = np.zeros((h, w))
#     for i in range(1, h):
#         mU = M[i-1]
#         mL = np.roll(mU, 1)
#         mR = np.roll(mU, -1)
#
#         mULR = np.array([mU, mL, mR])
#         cULR = np.array([C_U[i], C_L[i], C_R[i]])
#         mULR += cULR
#
#         argmins = np.argmin(mULR, axis=0)
#         m[i] = np.choose(argmins, mULR)
#         energy[i] = np.choose(argmins, cULR)
#
#     return energy

def forward_energy(img, flag=False):
    height = img.shape[0]
    width = img.shape[1]
    I = color.rgb2gray(img)

    energy = np.zeros((height, width))
    m = np.zeros((height, width))

    U = np.roll(I, 1, axis=0)
    L = np.roll(I, 1, axis=1)
    R = np.roll(I, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, height):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)

    return m


def cumulative_map_forward(image, energy_map):
    kernel_x = np.array([[0., 0., 0.], [-1., 0., 1.], [0., 0., 0.]], dtype=np.float64)
    kernel_y_left = np.array([[0., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float64)
    kernel_y_right = np.array([[0., 0., 0.], [1., 0., 0.], [0., -1., 0.]], dtype=np.float64)

    matrix_x = calc_neighbor_matrix(image, kernel_x)
    matrix_y_left = calc_neighbor_matrix(image, kernel_y_left)
    matrix_y_right = calc_neighbor_matrix(image, kernel_y_right)

    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            if col == 0:
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_right, e_up)
            elif col == n - 1:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_left, e_up)
            else:
                e_left = output[row - 1, col - 1] + matrix_x[row - 1, col - 1] + matrix_y_left[row - 1, col - 1]
                e_right = output[row - 1, col + 1] + matrix_x[row - 1, col + 1] + matrix_y_right[row - 1, col + 1]
                e_up = output[row - 1, col] + matrix_x[row - 1, col]
                output[row, col] = energy_map[row, col] + min(e_left, e_right, e_up)
    return output


def calc_neighbor_matrix(image, kernel):
    b, g, r = cv2.split(image)
    output = np.absolute(cv2.filter2D(b, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(g, -1, kernel=kernel)) + \
             np.absolute(cv2.filter2D(r, -1, kernel=kernel))
    return output
