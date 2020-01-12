import numpy as np
import time as time
import cv2
# from test import x_gradient_magnitudes
from algo import SeamCarver
from PIL import Image

seam_carver = SeamCarver("island_original.png", 466, 350)

# seam_carver.start()

a = seam_carver.out_image

image_new_image = Image.fromarray(a.astype("uint8"))

# image_new_image.show()
image_new_image.save("pics/final_island.png")

print("")

# import fib
from PIL import Image

# fib.fib(10)

# img = Image.open('island_original.png')
#
# im_rgba = img.copy()
# im_rgba.putalpha(255)
# im_rgba.save('pics/transparent_island_255.png')
