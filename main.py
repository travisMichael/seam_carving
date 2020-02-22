from insert_seams import insert_seams
from remove_seams import remove_seams
import numpy as np
import cv2
import sys


# Seam removal: Figure 5 from the 2007 paper -- you do not need to show scaling or cropping
def island_down():
    # runtime is roughly 40 seconds
    img_array = cv2.imread('/island/island_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 350)
    cv2.imwrite("island/island_result.png", new_image)


# Seam insertion: Figure 8 from the 2007 paper -- parts c, d, and f only
# c
def dolphin_stretch_1_with_mask():
    img_array = cv2.imread('/dolphin/dolphin.png').astype(np.float64)
    new_image = insert_seams(img_array, 120, with_mask=True)
    cv2.imwrite("dolphin/dolphin_stretch_1_mask.png", new_image)


# d
def dolphin_stretch_1():
    img_array = cv2.imread('dolphin/dolphin.png').astype(np.float64)
    new_image = insert_seams(img_array, 119)
    cv2.imwrite("dolphin/dolphin_stretch_1_result.png", new_image)


# f
def dolphin_stretch_2():
    img_array = cv2.imread('dolphin/dolphin_stretch_1_result.png').astype(np.float64)
    new_image = insert_seams(img_array, 121)
    cv2.imwrite("dolphin/dolphin_stretch_2_result.png", new_image)


# Seam removal: Figure 8 (bench) from the 2008 paper -- recreate the comparison images,
# including seam removal depictions, for both backward and forward energies
def bench_backward():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256)
    cv2.imwrite("bench/bench_backward_result.png", new_image)


def bench_backward_with_mask():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256, with_mask=True)
    cv2.imwrite("bench/bench_backward_mask_result.png", new_image)


def bench_forward():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = insert_seams(img_array.astype(float), 256, use_forward_energy=True)
    cv2.imwrite("bench/bench_forward_result.png", new_image)


def bench_forward_with_mask():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = insert_seams(img_array.astype(float), 256, with_mask=True, use_forward_energy=True)
    cv2.imwrite("bench/bench_forward_mask_result.png", new_image)


# Seam insertion: Figure 9 (elongated car) from the 2008 paper -- recreate the comparison images
#  for both backward and forward energies
def car_forward():
    img_array = cv2.imread('car/car_original.png').astype(np.float64)
    new_image = insert_seams(img_array, 192, use_forward_energy=True)
    cv2.imwrite("car/car_stretch_forward.png", new_image)


def car_backward():
    img_array = cv2.imread('car/car_original.png').astype(np.float64)
    new_image = insert_seams(img_array, 192)
    cv2.imwrite("car/car_stretch_backward.png", new_image)


if __name__ == '__main__':
    a = sys.argv[0]
    # island_down()
    # bench_backward()
    # bench_forward()
    # bench_backward_with_mask()
    # car_forward()
    car_backward()
    print("done")
