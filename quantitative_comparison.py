import cv2
import numpy as np


def calculate_similarity(diff):
    h, w = diff.shape
    total = h * w
    sum = np.sum(diff) / 255
    avg = sum / total
    return (1 - avg) * 100


island_diff = cv2.imread("fig5_extra1.jpg", 0).astype(float)
dolphin_diff_1 = cv2.imread("fig8d_07_extra1.jpg", 0).astype(float)
dolphin_diff_2 = cv2.imread("fig8f_07_extra1.jpg", 0).astype(float)
bench_backward_diff = cv2.imread("fig8Comp_backward_08_extra1.jpg", 0).astype(float)
bench_forward_diff = cv2.imread("fig8Comp_forward_08_extra1.jpg", 0).astype(float)
car_backward_diff = cv2.imread("fig9Comp_forward_08_extra1.jpg", 0).astype(float)
car_forward_diff = cv2.imread("fig9Comp_backward_08_extra1.jpg", 0).astype(float)


result = calculate_similarity(island_diff)
print("Island Similarity %: " + str(result))

result = calculate_similarity(dolphin_diff_1)
print("Dolphin Stretch 1 Similarity %: " + str(result))

result = calculate_similarity(dolphin_diff_2)
print("Dolphin Stretch 2 Similarity %: " + str(result))

result = calculate_similarity(bench_backward_diff)
print("Bench Backward Similarity %: " + str(result))

result = calculate_similarity(bench_forward_diff)
print("Bench Forward Similarity %: " + str(result))

result = calculate_similarity(car_backward_diff)
print("Car Backward Similarity %: " + str(result))

result = calculate_similarity(car_forward_diff)
print("Bench Forward Similarity %: " + str(result))