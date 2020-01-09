import numpy as np
import time as time
# import fib
from PIL import Image

# fib.fib(10)

# img = Image.open('island_original.png')
#
# im_rgba = img.copy()
# im_rgba.putalpha(255)
# im_rgba.save('pics/transparent_island_255.png')

start_time = time.time()

original = np.array([[1,2,3,0,1,3],
                     [0,-1,0,0,0,0],
                     [1,2,-3,0,1,3]
                     ] )

# original[1, :] = original[0, :]


height, width = original.shape

c = np.zeros((3, 6))

# original[np.argwhere(original) > 3] = 2
left_shift = np.delete(original[0], 0)
right_shift = np.delete(original[0], width - 1)

left_shift = np.concatenate((left_shift, [np.inf]))
right_shift = np.concatenate(([np.inf], right_shift))

result = np.zeros(len(original[0]))
# np.put(a, [0, 2], [-44, -55])
# we want to shift left
c[0] = np.less_equal(left_shift, original[0])
c_0_i = np.where(c[1] == 1)
zero_not = np.logical_not(c[0])

# we want to shift right
c[1] = np.less_equal(right_shift, original[0])
c_1_i = np.where(c[1] == 1)
one_not = np.logical_not(c[1])

np.put(result, c_1_i, right_shift[c_1_i])
np.put(result, c_0_i, left_shift[c_0_i])
# indices to keep original values
not_and = np.logical_and(zero_not, one_not)
not_and_i = np.where(not_and == True)

# if both shifting is true, then we check which shift is better
c[2] = np.less_equal(left_shift, right_shift)

intermediate = np.logical_and(c[0], c[1])
override_left = np.logical_and(intermediate, c[2])
override_left_i = np.where(override_left == True)

np.put(result, override_left_i, left_shift[override_left_i])
np.put(result, not_and_i, original[0][not_and_i])

# np.put(c[0], [0, 2], [-44, -55])
# e = np.delete(c, obj=2, axis=0)
# your code
elapsed_time = time.time() - start_time
# .astype(int)
x = np.arange(9.)
r = np.where( x == 5 )

print(elapsed_time)
print(c)