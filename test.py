import numpy as np
import cv2


island_result = cv2.imread("island/island_result.png", 0).astype(float)
island_target = cv2.imread("island/expected_island.png", 0).astype(float)

dolphin_result = cv2.imread("dolphin/dolphin_stretch_1_result.png", 0).astype(float)
dolphin_target = cv2.imread("dolphin/dolphinStretch1.png", 0).astype(float)

dolphin_stretch_2_result = cv2.imread("dolphin/dolphin_stretch_2_result.png", 0).astype(float)
dolphin_stretch_2_target = cv2.imread("dolphin/dolphinStretch2.png", 0).astype(float)

island_diff = island_result - island_target
dolphin_diff_1 = dolphin_result - dolphin_target
dolphin_diff_2 = dolphin_stretch_2_result - dolphin_stretch_2_target

cv2.imwrite("island/island_diff.png", island_diff)
cv2.imwrite("dolphin/dolphin_diff_1.png", dolphin_diff_1)
cv2.imwrite("dolphin/dolphin_diff_2.png", dolphin_diff_2)

print()