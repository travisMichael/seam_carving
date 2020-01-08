from PIL import Image
import numpy as np
import time

# One pitfall was figuring out how to calculate the x and y gradients


def x_gradient_magnitudes(image):
    height, width, channels = image.shape
    # dx = np.copy(image)
    dx = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1:
                dx[i][j] = np.linalg.norm(image[i][j])
            else:
                temp = image[i+1][j] - image[i-1][j]
                dx[i][j] = np.sqrt(np.sum(temp * temp))

    return dx


def y_gradient_magnitudes(image):
    height, width, channels = image.shape
    # dy = np.copy(image)
    dy = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if j == 0 or j == width - 1:
                dy[i][j] = np.linalg.norm(image[i][j])
            else:
                temp = image[i][j+1] - image[i][j-1]
                dy[i][j] = np.sqrt(np.sum(temp * temp))

    return dy


# def gradient_magnitude(image):
#
#     height, width, _ = image.shape
#
#     delta = np.zeros((height, width))
#
#     for i in range(len(image)):
#         for j in range(len(image[0])):
#             delta[i][j] = np.sum(image[i][j] * image[i][j])
#
#     return delta


def calculate_y_path_map(energy_map):
    height, width = energy_map.shape
    y_map = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            if i == 0:
                y_map[i][j] = energy_map[i][j]
            else:
                left = np.max([j - 1, 0])
                right = np.min([j + 1, width - 1])
                y_map[i][j] = energy_map[i][j] + np.min([y_map[i-1][left], y_map[i-1][j], y_map[i-1][right]])

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


def remove_vertical_seam(image, y_map):
    height, width, _ = image.shape
    to_be_removed_index = 0
    to_be_removed = np.zeros(height * 3)
    choices = np.zeros(3, dtype=int)

    i = height - 1
    j = np.argmin(y_map[i, :])

    start_index = 3 * width * i + j * 3
    to_be_removed[to_be_removed_index] = start_index
    to_be_removed[to_be_removed_index + 1] = start_index + 1
    to_be_removed[to_be_removed_index + 2] = start_index + 2
    to_be_removed_index += 3

    while i > 0:
        i = i - 1
        choices[0] = np.max([j - 1, 0])
        choices[1] = j
        choices[2] = np.min([j + 1, width - 1])
        a_min = np.argmin([y_map[i][choices[0]], y_map[i][j], y_map[i][choices[2]]])
        j = choices[a_min]
        start_index = 3 * width * i + j * 3
        to_be_removed[to_be_removed_index] = start_index
        to_be_removed[to_be_removed_index + 1] = start_index + 1
        to_be_removed[to_be_removed_index + 2] = start_index + 2
        to_be_removed_index += 3

    intermediate_image = np.delete(image, to_be_removed, None)

    new_image = intermediate_image.reshape((height, width - 1, 3))

    return new_image


def scale_image(image_to_scale):

    dx_time = 0.0
    dy_time = 0.0
    path_time = 0.0
    removal_time = 0.0

    # 71 + 100 + 100
    for i in range(79):
        start_time = time.time()
        dx = x_gradient_magnitudes(image_to_scale)
        dx_time += time.time() - start_time

        start_time = time.time()
        dy = y_gradient_magnitudes(image_to_scale)
        dy_time += time.time() - start_time
        # dI = dx + dy
        energy_map = dx + dy
        # energy_map = gradient_magnitude(dI)
        start_time = time.time()
        y_map = calculate_y_path_map(energy_map)
        path_time += time.time() - start_time

        start_time = time.time()
        image_to_scale = remove_vertical_seam(image_to_scale, y_map)
        removal_time += time.time() - start_time

        image_new_image = Image.fromarray(image_to_scale)
        image_new_image.save("pics/islands_" + str(i) + ".png")
        print(i, dx_time, dy_time, path_time, removal_time)

    return image_to_scale


# choices = np.zeros(3, dtype=int)
# Read image
img = Image.open('pics/islands_99.png')

# img.save('island_original_copy.png')

img_array = np.asarray(img)

# prints format of image
print(img.format)

# dx = x_gradient(img_array)
# dy = y_gradient(img_array)
#
# dI = dx + dy
# energy_map = gradient_magnitude(dI)
#
# y_map = calculate_y_path_map(energy_map)
#
# new_image = remove_vertical_seam(img_array, y_map)
new_image = scale_image(img_array)

# image_dx = Image.fromarray(dx)
# image_dy = Image.fromarray(dy)
# image_dI = Image.fromarray(dI)
# image_m = Image.fromarray(energy_map, 'L')
image_new_image = Image.fromarray(new_image)

image_new_image.show()
# image_dx.show()
# image_dy.show()
# image_dI.show()
# image_m.show()

print()