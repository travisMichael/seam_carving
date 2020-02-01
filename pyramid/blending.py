""" Pyramid Blending

This file has a number of functions that you need to fill out in order to
complete the assignment. Please write the appropriate code, following the
instructions on which functions you may or may not use.

References
----------
See the following papers, available on T-square under references:

(1) "The Laplacian Pyramid as a Compact Image Code"
        Burt and Adelson, 1983

(2) "A Multiresolution Spline with Application to Image Mosaics"
        Burt and Adelson, 1983

Notes
-----
    You may not use cv2.pyrUp or cv2.pyrDown anywhere in this assignment.

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
import scipy as sp
import scipy.signal  # one option for a 2D convolution library
import cv2


def ceiling(number):
    number_to_int = int(number)
    if number_to_int == number:
        return number_to_int
    return number_to_int + 1


# def reduce_channel(image_channel_current, h_next, w_next):
#     w = np.array([
#         [1, 4, 6, 4, 1],
#         [4, 16, 24, 16, 4],
#         [6, 24, 36, 24, 6],
#         [4, 16, 24, 16, 4],
#         [1, 4, 6, 4, 1],
#     ]) / 256

    # dst = cv2.copyMakeBorder(image_channel_current, 2, 2, 2, 2, cv2.BORDER_REPLICATE, None)
    #
    # M = 5
    # N = 5
    #
    # image_channel_1 = np.zeros((h_next, w_next), dtype="uint8")
    #
    # # this implementation is based off of the formula from http://persci.mit.edu/pub_pdfs/spline83.pdf page 222.
    # for i in range(h_next):
    #     for j in range(w_next):
    #         temp = 0
    #         for m in range(M):
    #             for n in range(N):
    #                 temp += dst[2 * i + m][2 * j + n] * w[m][n]
    #                 # handle overflow
    #         if temp > 255:
    #             temp = 255
    #         image_channel_1[i][j] = int(temp)
    #
    # return image_channel_1


# def expand_channel(image_channel, h_next, w_next):
#
#     channel_next = np.zeros((h_next, w_next), dtype="uint8")
#
#     dst = cv2.copyMakeBorder(image_channel, 2, 2, 2, 2, cv2.BORDER_REPLICATE, None)
#
#     for i in range(h_next):
#         for j in range(w_next):
#             summation = 0
#             for m in range(-2, 3):
#                 for n in range(-2, 3):
#                     pixeli = (i - m)/2
#                     pixelj = (j - n)/2
#                     if (int(pixeli) == pixeli) and (int(pixelj) == pixelj):
#                         pixeli = pixeli + 2
#                         pixelj = pixelj + 2
#                         tmpval =  dst[int(pixeli)][int(pixelj)] * w[m + 2][n + 2]
#                         summation += tmpval
#             if summation > 255:
#                 summation = 255
#             channel_next[i][j] = 4 * summation
#
#     return channel_next


def generatingKernel(a):
    """Return a 5x5 generating kernel based on an input parameter (i.e., a
    square "5-tap" filter.)

    Parameters
    ----------
    a : float
        The kernel generating parameter in the range [0, 1] used to generate a
        5-tap filter kernel.

    Returns
    -------
    output : numpy.ndarray
        A 5x5 array containing the generated kernel
    """
    # DO NOT CHANGE THE CODE IN THIS FUNCTION
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def reduce_layer(image, kernel=generatingKernel(0.4)):
    """Convolve the input image with a generating kernel and then reduce its
    width and height each by a factor of two.

    For grading purposes, it is important that you use a reflected border
    (i.e., padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid
    region (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution. Subsampling must include the first
    row and column, skip the second, etc.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                          fefghg
        abcd     Pad      babcdc   Convolve   ZYXW   Subsample   ZX
        efgh   ------->   fefghg   -------->  VUTS   -------->   RP
        ijkl    BORDER    jijklk     keep     RQPO               JH
        mnop   REFLECT    nmnopo     valid    NMLK
        qrst              rqrsts              JIHG
                          nmnopo

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the reduce function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data type
        (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data type (e.g.,
        np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (ceil(r/2), ceil(c/2)). For instance, if the input is
        5x7, the output will be 3x4.
    """
    # dst = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REFLECT101, None)
    h, w = image.shape
    h_new = ceiling(h / 2)
    w_new = ceiling(w / 2)

    blurred = cv2.filter2D(image, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101) # np.convolve(dst, kernel)
    image_new = np.zeros((h_new, w_new))

    for i in range(h_new):
        image_new[i][:] = blurred[i*2][0:w + 1: 2]

    return image_new.astype("uint8")

    # M = 5
    # N = 5



    # new_image = np.zeros((h, w), dtype="uint8")
    #
    # # this implementation is based off of the formula from http://persci.mit.edu/pub_pdfs/spline83.pdf page 222.
    # for i in range(h_new):
    #     for j in range(w_new):
    #         temp = 0
    #         for m in range(M):
    #             for n in range(N):
    #                 temp += dst[2 * i + m][2 * j + n] * kernel[m][n]
    #                 # handle overflow
    #         if temp > 255:
    #             temp = 255
    #         new_image[i][j] = int(temp)

    # return new_image

    # WRITE YOUR CODE HERE.
    # h_current, w_current, _ = image.shape
    # h_prev = ceiling(h_current / 2)
    # w_prev = ceiling(w_current / 2)

    # g_next = np.zeros((h_prev, w_prev))
    #
    # channel_0 = reduce_channel(image[:, :, 0], h_prev, w_prev)
    # channel_1 = reduce_channel(image[:, :, 1], h_prev, w_prev)
    # channel_2 = reduce_channel(image[:, :, 2], h_prev, w_prev)
    #
    # g_next[:, :, 0] = channel_0
    # g_next[:, :, 1] = channel_1
    # g_next[:, :, 2] = channel_2
    # g_next = cv2.pyrDown(image)
    # return g_next
    # raise NotImplementedError


def expand_layer(image, kernel=generatingKernel(0.4)):
    """Upsample the image to double the row and column dimensions, and then
    convolve it with a generating kernel.

    Upsampling the image means that every other row and every other column will
    have a value of zero (which is why we apply the convolution after). For
    grading purposes, it is important that you use a reflected border (i.e.,
    padding equivalent to cv2.BORDER_REFLECT101) and only keep the valid region
    (i.e., the convolution operation should return an image of the same
    shape as the input) for the convolution.

    Finally, multiply your output image by a factor of 4 in order to scale it
    back up. If you do not do this (and you should try it out without that)
    you will see that your images darken as you apply the convolution.
    You must explain why this happens in your submission PDF.

    Example (assuming 3-tap filter and 1-pixel padding; 5-tap is analogous):

                                          000000
             Upsample   A0B0     Pad      0A0B0B   Convolve   zyxw
        AB   ------->   0000   ------->   000000   ------->   vuts
        CD              C0D0    BORDER    0C0D0D     keep     rqpo
        EF              0000   REFLECT    000000    valid     nmlk
                        E0F0              0E0F0F              jihg
                        0000              000000              fedc
                                          0E0F0F

                NOTE: Remember to multiply the output by 4.

    A "3-tap" filter means a 3-element kernel; a "5-tap" filter has 5 elements.
    Please consult the lectures for a more in-depth discussion of how to
    tackle the expand function.

    Parameters
    ----------
    image : numpy.ndarray
        A grayscale image of shape (r, c). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    kernel : numpy.ndarray (Optional)
        A kernel of shape (N, N). The array may have any data
        type (e.g., np.uint8, np.float64, etc.)

    Returns
    -------
    numpy.ndarray(dtype=np.float64)
        An image of shape (2*r, 2*c). For instance, if the input is 3x4, then
        the output will be 6x8.
    """

    h, w = image.shape

    image_new = np.zeros((h * 2, w * 2))
    for i in range(h):
        image_new[i*2][0:w*2 + 1:2] = image[i]

    blurred = cv2.filter2D(image, -1, kernel=kernel, borderType=cv2.BORDER_REFLECT101)

    return blurred

    # WRITE YOUR CODE HERE.
    # h, w = image.shape
    # channel_next = np.zeros((h, w), dtype="uint8")
    #
    # dst = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REFLECT101, None)

    # for i in range(h):
    #     for j in range(w):
    #         summation = 0
    #         for m in range(-2, 3):
    #             for n in range(-2, 3):
    #                 pixeli = (i - m)/2
    #                 pixelj = (j - n)/2
    #                 if (int(pixeli) == pixeli) and (int(pixelj) == pixelj):
    #                     pixeli = pixeli + 2
    #                     pixelj = pixelj + 2
    #                     tmpval =  dst[int(pixeli)][int(pixelj)] * kernel[m + 2][n + 2]
    #                     summation += tmpval
    #         if summation > 255:
    #             summation = 255
    #         channel_next[i][j] = 4 * summation

    # return channel_next
    # return g_next


def gaussPyramid(image, levels):
    """Construct a pyramid from the image by reducing it by the number of
    levels specified by the input.

    You must use your reduce_layer() function to generate the pyramid.

    Parameters
    ----------
    image : numpy.ndarray
        An image of dimension (r, c).

    levels : int
        A positive integer that specifies the number of reductions to perform.
        For example, levels=0 should return a list containing just the input
        image; levels = 1 should perform one reduction and return a list with
        two images. In general, len(output) = levels + 1.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list of arrays of dtype np.float. The first element of the list
        (output[0]) is layer 0 of the pyramid (the image itself). output[1] is
        layer 1 of the pyramid (image reduced once), etc.
    """
    # WRITE YOUR CODE HERE.

    layers_of_pyramid = []

    image_to_reduce = np.copy(image)
    layers_of_pyramid.append(image_to_reduce)

    for i in range(levels):
        print('Reducing...')
        image_to_reduce = reduce_layer(image_to_reduce)
        layers_of_pyramid.append(image_to_reduce)

    return layers_of_pyramid
    # raise NotImplementedError


def laplPyramid(gaussPyr):
    """Construct a Laplacian pyramid from a Gaussian pyramid; the constructed
    pyramid will have the same number of levels as the input.

    You must use your expand_layer() function to generate the pyramid. The
    Gaussian Pyramid that is passed in is the output of your gaussPyramid
    function.

    Parameters
    ----------
    gaussPyr : list<numpy.ndarray(dtype=np.float)>
        A Gaussian Pyramid (as returned by your gaussPyramid function), which
        is a list of numpy.ndarray items.

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of the same size as gaussPyr. This pyramid should
        be represented in the same way as guassPyr, as a list of arrays. Every
        element of the list now corresponds to a layer of the laplacian
        pyramid, containing the difference between two layers of the gaussian
        pyramid.

        NOTE: The last element of output should be identical to the last layer
              of the input pyramid since it cannot be subtracted anymore.

    Notes
    -----
        (1) Sometimes the size of the expanded image will be larger than the
        given layer. You should crop the expanded image to match in shape with
        the given layer. If you do not do this, you will get a 'ValueError:
        operands could not be broadcast together' because you can't subtract
        differently sized matrices.

        For example, if my layer is of size 5x7, reducing and expanding will
        result in an image of size 6x8. In this case, crop the expanded layer
        to 5x7.
    """
    # WRITE YOUR CODE HERE.
    gaussPyr_length = len(gaussPyr)

    lapl_pyramid_layers = []
    for i in range(gaussPyr_length - 1):
        print("Expanding")
        current = gaussPyr[i]
        next = gaussPyr[i + 1]
        # expanded = expand_layer(next)
        expanded = expand_layer(next) # cv2.pyrUp(next)

        h, w = current.shape
        h_, w_ = expanded.shape
        if h != h_ or w != w_:
            expanded = cv2.resize(expanded, (w, h))
        laplacian = cv2.subtract(current.astype("uint8"), expanded.astype("uint8"))
        lapl_pyramid_layers.append(laplacian)

    lapl_pyramid_layers.append(gaussPyr[gaussPyr_length - 1])
    return lapl_pyramid_layers


def blend(laplPyrWhite, laplPyrBlack, gaussPyrMask):
    """Blend two laplacian pyramids by weighting them with a gaussian mask.

    You should return a laplacian pyramid that is of the same dimensions as the
    input pyramids. Every layer should be an alpha blend of the corresponding
    layers of the input pyramids, weighted by the gaussian mask.

    Therefore, pixels where current_mask == 1 should be taken completely from
    the white image, and pixels where current_mask == 0 should be taken
    completely from the black image.

    (The variables `current_mask`, `white_image`, and `black_image` refer to
    the images from each layer of the pyramids. This computation must be
    performed for every layer of the pyramid.)

    Parameters
    ----------
    laplPyrWhite : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of an image constructed by your laplPyramid
        function.

    laplPyrBlack : list<numpy.ndarray(dtype=np.float)>
        A laplacian pyramid of another image constructed by your laplPyramid
        function.

    gaussPyrMask : list<numpy.ndarray(dtype=np.float)>
        A gaussian pyramid of the mask. Each value should be in the range
        [0, 1].

    Returns
    -------
    list<numpy.ndarray(dtype=np.float)>
        A list containing the blended layers of the two laplacian pyramids

    Notes
    -----
        (1) The input pyramids will always have the same number of levels.
        Furthermore, each layer is guaranteed to have the same shape as
        previous levels.
    """

    # WRITE YOUR CODE HERE.
    blended_pyramed = []
    number_of_layers = len(laplPyrWhite)

    for i in range(number_of_layers):
        print("Blending layer: " + str(i))
        lapl_layer_i_white = laplPyrWhite[i]
        lapl_layer_i_black = laplPyrBlack[i]
        gauss_layer_i_mask = gaussPyrMask[i]

        mask_inverse = 255 - gauss_layer_i_mask
        new_lapl_white = np.multiply(lapl_layer_i_white, gauss_layer_i_mask / 255).astype("uint8")
        new_lapl_black = np.multiply(lapl_layer_i_black, mask_inverse / 255).astype("uint8")

        blended_layer = cv2.add(new_lapl_white.astype("uint8"), new_lapl_black.astype("uint8"))
        blended_pyramed.append(blended_layer)

    return blended_pyramed


def collapse(pyramid):
    """Collapse an input pyramid.

    Approach this problem as follows: start at the smallest layer of the
    pyramid (at the end of the pyramid list). Expand the smallest layer and
    add it to the second to smallest layer. Then, expand the second to
    smallest layer, and continue the process until you are at the largest
    image. This is your result.

    Parameters
    ----------
    pyramid : list<numpy.ndarray(dtype=np.float)>
        A list of numpy.ndarray images. You can assume the input is taken
        from blend() or laplPyramid().

    Returns
    -------
    numpy.ndarray(dtype=np.float)
        An image of the same shape as the base layer of the pyramid.

    Notes
    -----
        (1) Sometimes expand will return an image that is larger than the next
        layer. In this case, you should crop the expanded image down to the
        size of the next layer. Look into numpy slicing to do this easily.

        For example, expanding a layer of size 3x4 will result in an image of
        size 6x8. If the next layer is of size 5x7, crop the expanded image
        to size 5x7.
    """

    # WRITE YOUR CODE HERE.
    aggregated = pyramid[len(pyramid) - 1]
    for i in range(len(pyramid) - 2, -1, -1):
        print(i)
        current = pyramid[i]
        exp = expand_layer(aggregated.astype("uint8"))
        # exp = cv2.pyrUp(aggregated.astype("uint8"))
        h, w = current.shape
        h_, w_ = exp.shape
        if h != h_ or w != w_:
            exp = cv2.resize(exp, (w, h))
        aggregated = cv2.add(current.astype("uint8"), exp.astype("uint8"))

    return aggregated

