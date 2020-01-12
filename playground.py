import numpy as np
import time as time
import cv2
# from test import x_gradient_magnitudes

start_time = time.time()

original = np.array([
    [[1,2,3], [0,-1,0], [4,5,6]],
    [[6,2,-3], [6,-1,0], [6,5,6]],
    [[7,2,-3], [7,-1,0], [7,5,6]]
])

seam = np.zeros(3)

seam[0] = 0
seam[1] = 1
seam[2] = 2

new_image = np.zeros((3, 4, 3), dtype="uint8")

elapsed_time = time.time() - start_time
# .astype(int)
x = np.arange(9.)
r = np.where( x == 5 )

print(elapsed_time)
print(c)