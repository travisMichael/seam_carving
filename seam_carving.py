from insert_seams import insert_seams
from remove_seams import remove_seams
import numpy as np
import cv2
import sys
import time


# Seam removal: Figure 5 from the 2007 paper -- you do not need to show scaling or cropping
def island_down():
    # runtime is roughly 40 seconds
    img_array = cv2.imread('island/island_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 350)
    cv2.imwrite("island/island_result.png", new_image)
    cv2.imwrite("fig5.jpg", new_image)


# Seam insertion: Figure 8 from the 2007 paper -- parts c, d, and f only
# c
def dolphin_stretch_1_with_mask():
    img_array = cv2.imread('dolphin/dolphin.png').astype(np.float64)
    new_image = insert_seams(img_array, 120, with_mask=True)
    cv2.imwrite("dolphin/dolphin_stretch_1_mask.png", new_image)
    cv2.imwrite("fig8c_07.jpg", new_image)


# d
def dolphin_stretch_1():
    img_array = cv2.imread('dolphin/dolphin.png').astype(np.float64)
    new_image = insert_seams(img_array, 119)
    cv2.imwrite("dolphin/dolphin_stretch_1_result.png", new_image)
    cv2.imwrite("fig8d_07.jpg", new_image)


# f
def dolphin_stretch_2():
    img_array = cv2.imread('dolphin/dolphin_stretch_1_result.png').astype(np.float64)
    new_image = insert_seams(img_array, 121)
    cv2.imwrite("dolphin/dolphin_stretch_2_result.png", new_image)
    cv2.imwrite("fig8f_07.jpg", new_image)


# Seam removal: Figure 8 (bench) from the 2008 paper -- recreate the comparison images,
# including seam removal depictions, for both backward and forward energies
def bench_backward():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256)
    cv2.imwrite("bench/bench_backward_result.png", new_image)
    cv2.imwrite("fig8Comp_backward_08.jpg", new_image)


def bench_backward_with_mask():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256, with_mask=True)
    cv2.imwrite("bench/bench_backward_mask_result.png", new_image)
    cv2.imwrite("fig8Seam_backward_08.jpg", new_image)


def bench_forward():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256, use_forward_energy=True)
    cv2.imwrite("bench/bench_forward_result.png", new_image)
    cv2.imwrite("fig8Comp_forward_08.jpg", new_image)


def bench_forward_with_mask():
    img_array = cv2.imread('bench/bench_original.png').astype(np.float64)
    new_image = remove_seams(img_array.astype(float), 256, with_mask=True, use_forward_energy=True)
    cv2.imwrite("bench/bench_forward_mask_result.png", new_image)
    cv2.imwrite("fig8Seam_forward_08.jpg", new_image)


# Seam insertion: Figure 9 (elongated car) from the 2008 paper -- recreate the comparison images
#  for both backward and forward energies
def car_forward():
    img_array = cv2.imread('car/car_original.png').astype(np.float64)
    new_image = insert_seams(img_array, 192, use_forward_energy=True)
    cv2.imwrite("car/car_stretch_forward.png", new_image)
    cv2.imwrite("fig9Comp_forward_08.jpg", new_image)


def car_backward():
    img_array = cv2.imread('car/car_original.png').astype(np.float64)
    new_image = insert_seams(img_array, 192)
    cv2.imwrite("car/car_stretch_backward.png", new_image)
    cv2.imwrite("fig9Comp_backward_08.jpg", new_image)


def diff():
    island_result = cv2.imread("fig5.jpg", 0).astype(float)
    island_target = cv2.imread("fig5_extra2.jpg", 0).astype(float)

    dolphin_result = cv2.imread("fig8d_07.jpg", 0).astype(float)
    dolphin_target = cv2.imread("fig8d_07_extra2.jpg", 0).astype(float)

    dolphin_stretch_2_result = cv2.imread("fig8f_07.jpg", 0).astype(float)
    dolphin_stretch_2_target = cv2.imread("fig8f_07_extra2.jpg", 0).astype(float)

    bench_backward_result = cv2.imread("fig8Comp_backward_08.jpg", 0).astype(float)
    bench_backward_target = cv2.imread("fig8Comp_backward_08_extra2.jpg", 0).astype(float)

    bench_forward_result = cv2.imread("fig8Comp_forward_08.jpg", 0).astype(float)
    bench_forward_target = cv2.imread("fig8Comp_forward_08_extra2.jpg", 0).astype(float)

    car_backward_result = cv2.imread("fig9Comp_backward_08.jpg", 0).astype(float)
    car_backward_target = cv2.imread("fig9Comp_backward_08_extra2.jpg", 0).astype(float)

    car_forward_result = cv2.imread("fig9Comp_forward_08.jpg", 0).astype(float)
    car_forward_target = cv2.imread("fig9Comp_forward_08_extra2.jpg", 0).astype(float)

    island_diff = island_result - island_target
    dolphin_diff_1 = dolphin_result - dolphin_target
    dolphin_diff_2 = dolphin_stretch_2_result - dolphin_stretch_2_target
    bench_backward_diff = bench_backward_result - bench_backward_target
    bench_forward_diff = bench_forward_result - bench_forward_target
    car_backward_diff = car_backward_result - car_backward_target
    car_forward_diff = car_forward_result - car_forward_target

    cv2.imwrite("island/island_diff.png", island_diff)
    cv2.imwrite("fig5_extra1.jpg", island_diff)
    cv2.imwrite("dolphin/dolphin_diff_1.png", dolphin_diff_1)
    cv2.imwrite("fig8d_07_extra1.jpg", dolphin_diff_1)
    cv2.imwrite("dolphin/dolphin_diff_2.png", dolphin_diff_2)
    cv2.imwrite("fig8f_07_extra1.jpg", dolphin_diff_2)
    cv2.imwrite("bench/bench_backward_diff.png", bench_backward_diff)
    cv2.imwrite("fig8Comp_backward_08_extra1.jpg", bench_backward_diff)
    cv2.imwrite("bench/bench_forward_diff.png", bench_forward_diff)
    cv2.imwrite("fig8Comp_forward_08_extra1.jpg", bench_forward_diff)
    cv2.imwrite("car/car_backward_diff.png", car_backward_diff)
    cv2.imwrite("fig9Comp_forward_08_extra1.jpg", car_backward_diff)
    cv2.imwrite("car/car_forward_diff.png", car_forward_diff)
    cv2.imwrite("fig9Comp_backward_08_extra1.jpg", car_forward_diff)


if __name__ == '__main__':
    a = sys.argv[0]
    b = time.time()
    if (len(sys.argv) >= 2):
        input = sys.argv[1]
        if input == "island_down":
            island_down()
        if input == "dolphin_stretch_1":
            dolphin_stretch_1()
        if input == "dolphin_stretch_1_with_mask":
            dolphin_stretch_1_with_mask()
        if input == "dolphin_stretch_2":
            dolphin_stretch_2()
        if input == "bench_backward":
            bench_backward()
        if input == "bench_backward_with_mask":
            bench_backward_with_mask()
        if input == "bench_forward":
            bench_forward()
        if input == "bench_forward_with_mask":
            bench_forward_with_mask()
        if input == "car_forward":
            car_forward()
        if input == "car_backward":
            car_backward()
        if input == "all":
            start = time.time()
            island_down()
            dolphin_stretch_1()
            dolphin_stretch_1_with_mask()
            dolphin_stretch_2()
            bench_backward()
            bench_backward_with_mask()
            bench_forward()
            bench_forward_with_mask()
            car_forward()
            car_backward()
            end = time.time() - start
            print(end)
        if input == "diff":
            diff()

    print("done")
