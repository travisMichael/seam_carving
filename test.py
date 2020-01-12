from insert_seams import scale_image_up
from remove_seams import scale_image
import numpy as np
import cv2


# One pitfall was figuring out how to calculate the x and y gradients


# img = Image.open('island_original.png')
#
# img_array = cv2.imread('island_original.png').astype(np.float64)
#
# print(img.format)
#
# new_image = scale_image(img_array.astype(float))
#
# cv2.imwrite("island_result.png", new_image)


img_array = cv2.imread('dolphin.png').astype(np.float64)

new_image = scale_image_up(img_array, 120)

cv2.imwrite("dolphin_50_result.png", new_image)

print("done")
