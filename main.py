from insert_seams import scale_image_up
from remove_seams import scale_image
import numpy as np
import cv2
import sys


# Seam removal: Figure 5 from the 2007 paper -- you do not need to show scaling or cropping
def island_down():
    img_array = cv2.imread('island_original.png').astype(np.float64)
    new_image = scale_image(img_array.astype(float), 350)
    cv2.imwrite("island_result.png", new_image)

# Seam insertion: Figure 8 from the 2007 paper -- parts c, d, and f only
# c
def dolphin_up_with_mask():
    img_array = cv2.imread('dolphin.png').astype(np.float64)
    new_image = scale_image_up(img_array, 120, use_mask=True)
    cv2.imwrite("dolphin_up_mask.png", new_image)


#d
def dolphin_up():
    img_array = cv2.imread('dolphin.png').astype(np.float64)
    new_image = scale_image_up(img_array, 120)
    cv2.imwrite("dolphin_up.png", new_image)


# f
def dolphin_up_up():
    img_array = cv2.imread('dolphin_up.png').astype(np.float64)
    new_image = scale_image_up(img_array, 120)
    cv2.imwrite("dolphin_up_up.png", new_image)


# Seam removal: Figure 8 (bench) from the 2008 paper -- recreate the comparison images, including seam removal depictions, for both backward and forward energies
def bench_backward():
    img_array = cv2.imread('bench_original.png').astype(np.float64)
    new_image = scale_image(img_array.astype(float), 256)
    cv2.imwrite("bench_backward_result.png", new_image)


def bench_backward_with_mask():
    img_array = cv2.imread('bench_original.png').astype(np.float64)
    new_image = scale_image(img_array.astype(float), 256, with_mask=True)
    cv2.imwrite("bench_backward_mask_result.png", new_image)


def bench_forward():
    img_array = cv2.imread('bench_original.png').astype(np.float64)
    new_image = scale_image(img_array.astype(float), 256, True)
    cv2.imwrite("bench_forward_result.png", new_image)


def bench_forward_with_mask():
    img_array = cv2.imread('bench_original.png').astype(np.float64)
    new_image = scale_image(img_array.astype(float), 256, with_mask=True, use_forward_energy=True)
    cv2.imwrite("bench_forward_mask_result.png", new_image)


# Seam insertion: Figure 9 (elongated car) from the 2008 paper -- recreate the comparison images for both backward and forward energies
def car_forward():
    img_array = cv2.imread('car_original.png').astype(np.float64)
    new_image = scale_image_up(img_array, 190, use_forward_energy=True)
    cv2.imwrite("car_up_forward.png", new_image)


def car_backward():
    img_array = cv2.imread('car_original.png').astype(np.float64)
    new_image = scale_image_up(img_array, 190)
    cv2.imwrite("car_up_backward.png", new_image)


if __name__ == '__main__':
    a = sys.argv[0]
    bench_forward_with_mask()
    print("done")
