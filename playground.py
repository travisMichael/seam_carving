import numpy as np
import time as time
import cv2
from algo import SeamCarver

start_time = time.time()

s = SeamCarver('car_original.png', 385, 384 + 70)

cv2.imwrite("elongated_car_forward_result_2.png", s.out_image)

elapsed_time = time.time() - start_time
# .astype(int)
x = np.arange(9.)
r = np.where( x == 5 )

print(elapsed_time)
print(c)