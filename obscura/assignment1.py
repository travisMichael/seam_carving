""" Camera Obscura - Post-processing
This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

Notes
-----
You are only allowed to use cv2.imread, c2.imwrite and cv2.copyMakeBorder from
cv2 library. You should implement convolution on your own.
GENERAL RULES:
    1. DO NOT INCLUDE code that saves, shows, displays, writes the image that
    you are being passed in. Do that on your own if you need to save the images
    but these functions should NOT save the image to disk.
    2. DO NOT import any other libraries aside from those that we provide.
    You should be able to complete the assignment with the given libraries
    (and in many cases without them).
    3. DO NOT change the format of this file. You may NOT change function
    type signatures (not even named parameters with defaults). You may add
    additional code to this file at your discretion, however it is your
    responsibility to ensure that the autograder accepts your submission.
    4. This file has only been tested in the course virtual environment.
    You are responsible for ensuring that your code executes properly in the
    virtual machine environment, and that any changes you make outside the
    areas annotated for student code do not impact your performance on the
    autograder system.
"""
import numpy as np
import cv2


def single_channel_convolution(image_channel, f):

    height, width = image_channel.shape
    filter_channel_result = np.zeros((height, width), dtype=float)

    f_h, f_w = f.shape
    f_h_2 = int(f_h / 2)
    f_w_2 = int(f_w / 2)
    top = f_w_2
    bottom = f_w_2
    left = f_h_2
    right = f_h_2

    if f_h % 2 == 0:
        bottom = bottom - 1
    if f_w % 2 == 0:
        right = right - 1

    dst = cv2.copyMakeBorder(image_channel, top, bottom, left, right, cv2.BORDER_REPLICATE, None)

    for x in range(f_h_2, height + f_h_2):
        for y in range(f_w_2, width + f_w_2):
            summation = 0.0
            f_i = 0
            for i in range(x - top, x + bottom + 1):
                f_j = 0
                for j in range(y - left, y + right + 1):
                    summation += dst[i][j] * f[f_i][f_j]
                    f_j += 1
                f_i += 1

            # handle overflow
            if summation > 255:
                summation = 255
            if summation < 0:
                summation = 0
            filter_channel_result[x - f_h_2][y - f_w_2] = round(summation)

    return filter_channel_result.astype("uint8")


# Used example code from cv2 website for padding images
# https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
def applyConvolution(image, filter):
    """Apply convolution operation on image with the filter provided.
    Pad the image with cv2.copyMakeBorder and cv2.BORDER_REPLICATE to get an output image of the right size
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    filter: numpy.ndarray
        A numpy array of dimensions (N,M) and type np.float64
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """

    h, w, _ = image.shape

    img_out = np.zeros((h, w, 3), dtype="uint8")

    channel_0 = single_channel_convolution(image[:, :, 0], filter)
    channel_1 = single_channel_convolution(image[:, :, 1], filter)
    channel_2 = single_channel_convolution(image[:, :, 2], filter)

    img_out[:, :, 0] = channel_0
    img_out[:, :, 1] = channel_1
    img_out[:, :, 2] = channel_2

    return img_out


def median_filter_channel_convolution(image_channel, m, n):

    height, width = image_channel.shape
    filter_median_result = np.zeros((height, width))

    median_value_tracker = np.zeros((m, n))
    f_h_2 = int(m / 2)
    f_w_2 = int(n / 2)
    top = f_w_2
    bottom = f_w_2
    left = f_h_2
    right = f_h_2

    if m % 2 == 0:
        bottom = bottom - 1
    if n % 2 == 0:
        right = right - 1

    dst = cv2.copyMakeBorder(image_channel, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)

    for x in range(f_h_2, height + f_h_2):
        # if x % 100 == 0:
        #     print(str(x) + '...')
        for y in range(f_w_2, width + f_w_2):
            f_i = 0
            for i in range(x - top, x + bottom + 1):
                f_j = 0
                for j in range(y - left, y + right + 1):
                    median_value_tracker[f_i][f_j] = dst[i][j]
                    f_j += 1
                f_i += 1

            filter_median_result[x - f_h_2][y - f_w_2] = int(np.median(median_value_tracker))

    return filter_median_result.astype("uint8")


# Used example code from cv2 website for padding images
# https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html
def applyMedianFilter(image, filterdimensions):
    """Apply median filter on image after padding it with zeros around the edges using cv2.copyMakeBorder
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    filterdimensions: list<int>
        List of length 2 that represents the filter size M x N
    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    M, N = filterdimensions

    h, w, _ = image.shape

    img_out = np.zeros((h, w, 3), dtype="uint8")

    channel_0 = median_filter_channel_convolution(image[:, :, 0], M, N)
    channel_1 = median_filter_channel_convolution(image[:, :, 1], M, N)
    channel_2 = median_filter_channel_convolution(image[:, :, 2], M, N)

    img_out[:, :, 0] = channel_0
    img_out[:, :, 1] = channel_1
    img_out[:, :, 2] = channel_2
    return img_out


def applyFilter1(image):
    """Filter noise from the image by using applyConvolution() and an averaging filter
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8

    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    average_filter = np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ])

    return applyConvolution(image, average_filter)


def applyFilter2(image):
    """Filter noise from the image by using applyConvolution() and a gaussian filter
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8

    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.

    gaussian_filter = np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ])

    return applyConvolution(image, gaussian_filter)


def sharpenImage(image):
    """Sharpen the image. Call applyConvolution with an image sharpening kernel
    Parameters
    ----------
    image : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8

    Returns
    -------
    output : numpy.ndarray
        A numpy array of dimensions (HxWx3) and type np.uint8
    """
    # WRITE YOUR CODE HERE.
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, -0]
    ])

    return applyConvolution(image, sharpen_kernel)


if __name__ == "__main__":
    # WRITE YOUR CODE HERE.
    # Read co_image_0.jpg and pass it to applyFilter1(), applyFilter2(), applyMedianFilter() and sharpenImage()
    # src = cv2.imread('co_image_0.jpg')
    # src = cv2.imread('setup_0.jpg')

    # r = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite("setup_0.jpg", src)

    # result_1 = applyFilter1(src)
    # result_2 = applyFilter2(src)
    # result_3 = applyMedianFilter(src, [3, 3])
    # result_4 = sharpenImage(src)

    # cv2.imwrite("averaging_filtered_image_0.jpg", result_1)
    # cv2.imwrite("gaussian_filtered_image_0.jpg", result_2)
    # cv2.imwrite("median_filtered_image_0.jpg", result_3)
    # cv2.imwrite("sharp_image_0.jpg", result_4)

    pass
