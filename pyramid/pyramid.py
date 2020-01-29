import cv2
import numpy as np
import math


def reduce_channel(image_channel_current, h_next, w_next):
    w = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ]) / 273

    dst = cv2.copyMakeBorder(image_channel_current, 2, 2, 2, 2, cv2.BORDER_REPLICATE, None)

    M = 5
    N = 5

    image_channel_1 = np.zeros((h_next, w_next), dtype="uint8")

    # this implementation is based off of the formula from http://persci.mit.edu/pub_pdfs/spline83.pdf page 222.
    for i in range(h_next):
        for j in range(w_next):
            temp = 0
            for m in range(M):
                for n in range(N):
                    temp += dst[2 * i + m][2 * j + n] * w[m][n]
                    # handle overflow
            if temp > 255:
                temp = 255
                # print(temp)
            image_channel_1[i][j] = int(temp)

    return image_channel_1


def reduce_layer(image):

    h_current, w_current, _ = image.shape
    h_prev = math.ceil(h_current / 2)
    w_prev = math.ceil(w_current / 2)

    g_next = np.zeros((h_prev, w_prev, 3), dtype="uint8")

    channel_0 = reduce_channel(image[:, :, 0], h_prev, w_prev)
    channel_1 = reduce_channel(image[:, :, 1], h_prev, w_prev)
    channel_2 = reduce_channel(image[:, :, 2], h_prev, w_prev)

    g_next[:, :, 0] = channel_0
    g_next[:, :, 1] = channel_1
    g_next[:, :, 2] = channel_2
    return g_next


# (193, 192, 3)
# (97, 96, 3)
# (49, 48, 3)
# (25, 24, 3)
# (13, 12, 3)
# (7, 6, 3)

img = cv2.imread("car_original.png")
img_2 = cv2.imread("r_0.jpg")

r = reduce_layer(img)

print(r.shape)
cv2.imwrite("r_0.jpg", r)
# Gaussian Pyramid
# layer = img.copy()
# gaussian_pyramid = [layer]
# for i in range(6):
#     layer = cv2.pyrDown(layer)
#     print(layer.shape)
#     cv2.imwrite("pyramid_" + str(i) + ".jpg", layer)
#     gaussian_pyramid.append(layer)


print("done")



