import numpy as np
import time as time

from PIL import Image

# img = Image.open('island_original.png')
#
# im_rgba = img.copy()
# im_rgba.putalpha(255)
# im_rgba.save('pics/transparent_island_255.png')

start_time = time.time()

original = np.array([[[1,1,1], [2,2,2]],
                     [[3,3,3], [6,6,6]],
                     [[5,5,5], [9,9,9]],
                     [[1,2,3], [4,5,6]]
                     ] )

height, width, channels = original.shape

a = np.delete(original, obj=[0, 1], axis=0)
b = np.delete(original, obj=[height - 1, height - 2], axis=0)

diff = (a - b)

squared = diff * diff

sum = np.sum(squared, axis=2)

root = np.sqrt(sum)

# np.concatenate((a,[[5,5,5,5]]), axis=1)
first_row = np.sqrt(np.sum(original[0]*original[0], axis=1))
final_row = np.sqrt(np.sum(original[height - 1]*original[height - 1], axis=1))

c = np.vstack((first_row, root))
final = np.vstack((c, final_row))
# e = np.delete(c, obj=2, axis=0)
# your code
elapsed_time = time.time() - start_time

print(elapsed_time)
print(c)