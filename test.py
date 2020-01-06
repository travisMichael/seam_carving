from PIL import Image
import numpy as np


# One pitfall was figuring out how to calculate the x and y gradients


def x_gradient(image):
    dx = np.copy(image)

    for i in range(1, len(image) - 1):
        for j in range(len(image[0])):
            temp = dx[i+1][j] - dx[i-1][j]
            # abs(dy[i][j+1] - dy[i][j-1])
            dx[i][j] = temp

    return dx


def y_gradient(image):
    dy = np.copy(image)

    for i in range(len(image)):
        for j in range(1, len(image[0]) - 1):
            temp = dy[i][j+1] - dy[i][j-1]
            # abs(dy[i][j+1] - dy[i][j-1])
            dy[i][j] = temp

    return dy


def gradient_magnitude(image):

    height, width, _ = image.shape

    delta = np.zeros((height, width))

    for i in range(len(image)):
        for j in range(len(image[0])):
            delta[i][j] = np.sum(image[i][j] * image[i][j])

    return delta


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
    index_to_be_removed_list = []
    to_be_removed_index = 0
    to_be_removed = np.zeros(height * 3)

    i = height - 1
    j = np.argmin(y_map[i, :])

    start_index = 3 * width * i + j * 3
    to_be_removed[to_be_removed_index] = start_index
    index_to_be_removed_list.append(start_index)
    index_to_be_removed_list.append(start_index + 1)
    index_to_be_removed_list.append(start_index + 2)

    while i > 0:
        i = i - 1
        left = np.max([j - 1, 0])
        right = np.min([j + 1, width - 1])
        choices = [left, j, right]
        arg = np.argmin([y_map[i][left], y_map[i][j], y_map[i][right]])
        j = choices[arg]
        start_index = 3 * width * i + j * 3
        index_to_be_removed_list.append(start_index)
        index_to_be_removed_list.append(start_index + 1)
        index_to_be_removed_list.append(start_index + 2)

    intermediate_image = np.delete(image, index_to_be_removed_list, None)

    new_image = intermediate_image.reshape((height, width - 1, 3))

    return new_image


def scale_image(image_to_scale):

    for i in range(1):
        print(i)
        dx = x_gradient(image_to_scale)
        dy = y_gradient(image_to_scale)
        dI = dx + dy

        energy_map = gradient_magnitude(dI)
        y_map = calculate_y_path_map(energy_map)
        image_to_scale = remove_vertical_seam(image_to_scale, y_map)
        image_new_image = Image.fromarray(image_to_scale)
        image_new_image.save("island_" + str(i) + ".png")

    return image_to_scale


# Read image
img = Image.open('island_original.png')

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