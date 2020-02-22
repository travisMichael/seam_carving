import numpy as np
import cv2


# bench_backward_target = cv2.imread("bench/bench_forward_target.png").astype(float)
#
# bench_backward_target = bench_backward_target[0:342, :, :]
#
# cv2.imwrite("bench/bench_forward_target.png", bench_backward_target)

island_result = cv2.imread("island/island_result.png", 0).astype(float)
island_target = cv2.imread("island/expected_island.png", 0).astype(float)

dolphin_result = cv2.imread("dolphin/dolphin_stretch_1_result.png", 0).astype(float)
dolphin_target = cv2.imread("dolphin/dolphinStretch1.png", 0).astype(float)

dolphin_stretch_2_result = cv2.imread("dolphin/dolphin_stretch_2_result.png", 0).astype(float)
dolphin_stretch_2_target = cv2.imread("dolphin/dolphinStretch2.png", 0).astype(float)

bench_backward_result = cv2.imread("bench/bench_backward_result.png", 0).astype(float)
bench_backward_target = cv2.imread("bench/bench_backward_target.png", 0).astype(float)

bench_forward_result = cv2.imread("bench/bench_forward_result.png", 0).astype(float)
bench_forward_target = cv2.imread("bench/bench_forward_target.png", 0).astype(float)

island_diff = island_result - island_target
dolphin_diff_1 = dolphin_result - dolphin_target
dolphin_diff_2 = dolphin_stretch_2_result - dolphin_stretch_2_target
bench_backward_diff = bench_backward_result - bench_backward_target
bench_forward_diff = bench_forward_result - bench_forward_target

cv2.imwrite("island/island_diff.png", island_diff)
cv2.imwrite("dolphin/dolphin_diff_1.png", dolphin_diff_1)
cv2.imwrite("dolphin/dolphin_diff_2.png", dolphin_diff_2)
cv2.imwrite("bench/bench_backward_diff.png", bench_backward_diff)
cv2.imwrite("bench/bench_forward_diff.png", bench_forward_diff)

print()