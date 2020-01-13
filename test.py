from insert_seams import scale_image_up
from remove_seams import scale_image
import numpy as np
import cv2


# One pitfall was figuring out how to calculate the x and y gradients


# img_array = cv2.imread('bench_original.png').astype(np.float64)
#
# new_image = scale_image(img_array.astype(float), 256)
#
# cv2.imwrite("bench_forward_result.png", new_image)

# 4
img_array = cv2.imread('car_original.png').astype(np.float64)
img_array = cv2.imread('elongated_car_forward_result.png').astype(np.float64)

new_image = scale_image_up(img_array, 30)

cv2.imwrite("elongated_car_forward_result.png", new_image)

print("done")
