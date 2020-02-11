import cv2
import numpy as np
import math


def reduce_channel(image_channel_current, h_next, w_next):
    w = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256

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


def expand_channel(image_channel, h_next, w_next):
    w = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]) / 256

    channel_next = np.zeros((h_next, w_next), dtype="uint8")

    dst = cv2.copyMakeBorder(image_channel, 2, 2, 2, 2, cv2.BORDER_REPLICATE, None)

    for i in range(h_next):
        for j in range(w_next):
            summation = 0
            for m in range(-2, 3):
                for n in range(-2, 3):
                    pixeli = (i - m)/2
                    pixelj = (j - n)/2
                    if (math.floor(pixeli) == pixeli) and (math.floor(pixelj) == pixelj):
                        pixeli = pixeli + 2
                        pixelj = pixelj + 2
                        tmpval =  dst[int(pixeli)][int(pixelj)] * w[m + 2][n + 2]
                        summation += tmpval
            if summation > 255:
                summation = 255
            channel_next[i][j] = 4 * summation

    return channel_next


def expand_layer(image):

    h_current, w_current, _ = image.shape
    h_next = math.ceil(h_current * 2)
    w_next = math.ceil(w_current * 2)

    g_next = np.zeros((h_next, w_next, 3), dtype="uint8")

    channel_0 = expand_channel(image[:, :, 0], h_next, w_next)
    channel_1 = expand_channel(image[:, :, 1], h_next, w_next)
    channel_2 = expand_channel(image[:, :, 2], h_next, w_next)

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


# img = cv2.imread("co_image_0.jpg")
# img_2 = cv2.imread("r_0.jpg")
#
# r = expand_layer(img_2)
#
# print(r.shape)
# cv2.imwrite("t_0.jpg", r)

# img = cv2.imread("car_original.png")
# img_2 = cv2.imread("t_0.jpg")
#
# l = img_2 - img
#
# cv2.imwrite("laplacian_2.jpg", l)
#
# print("done")



