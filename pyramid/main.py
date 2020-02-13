from pyramid.util import *
import cv2
from pyramid.blending import *



#
# # 114
img_2 = cv2.imread("original/burt_apple.png")
h, w, c = img_2.shape
i_2 = img_2[:, 114:w - 114, :]
i_2 = cv2.resize(i_2, (512, 512))
cv2.imwrite("original/apple.jpg", i_2)

mask = np.zeros((512, 512))
mask[:, 256: 513] = 255

# gauss_list = gaussPyramid(mask, 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/mask/gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
img = cv2.imread("original/apple.jpg")
# gauss_list = gaussPyramid(img[:, :, 0], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/apple/0_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
# gauss_list = gaussPyramid(img[:, :, 1], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/apple/1_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
#
# gauss_list = gaussPyramid(img[:, :, 2], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/apple/2_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
# img = cv2.imread("original/orange.jpg")
# gauss_list = gaussPyramid(img[:, :, 0], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/orange/0_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
# gauss_list = gaussPyramid(img[:, :, 1], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/orange/1_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
#
# gauss_list = gaussPyramid(img[:, :, 2], 5)
#
# for i in range(len(gauss_list)):
#     cv2.imwrite("pics/orange/2_gauss_pyramid" + str(i) + ".jpg", gauss_list[i])
# ----------------------------------------------------------------------------
# gauss_list = []
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/apple/0_gauss_pyramid" + str(i) + ".jpg")[:, :, 0])
#
# lapl_list = laplPyramid(gauss_list)
# for i in range(len(lapl_list)):
#      cv2.imwrite("pics/apple/0_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])

# gauss_list = []
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/apple/1_gauss_pyramid" + str(i) + ".jpg")[:, :, 1])
#
# lapl_list = laplPyramid(gauss_list)
# for i in range(len(lapl_list)):
#     cv2.imwrite("pics/apple/1_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])
# gauss_list = []
#
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/apple/2_gauss_pyramid" + str(i) + ".jpg")[:, :, 2])
#
# lapl_list = laplPyramid(gauss_list)
#
# for i in range(len(lapl_list)):
#     cv2.imwrite("pics/apple/2_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])
#
#
# gauss_list = []
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/orange/0_gauss_pyramid" + str(i) + ".jpg")[:, :, 0])
#
# lapl_list = laplPyramid(gauss_list)
# for i in range(len(lapl_list)):
#     cv2.imwrite("pics/orange/0_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])
# gauss_list = []
#
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/orange/1_gauss_pyramid" + str(i) + ".jpg")[:, :, 1])
#
# lapl_list = laplPyramid(gauss_list)
# for i in range(len(lapl_list)):
#     cv2.imwrite("pics/orange/1_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])
# gauss_list = []
#
# for i in range(6):
#     gauss_list.append(cv2.imread("pics/orange/2_gauss_pyramid" + str(i) + ".jpg")[:, :, 2])
#
# lapl_list = laplPyramid(gauss_list)
#
# for i in range(len(lapl_list)):
#     cv2.imwrite("pics/orange/2_lapl_pyramid_apple" + str(i) + ".jpg", lapl_list[i])
# ----------------------------------------------------------------------------
# lapl_apple_list = []
# for i in range(6):
#     lapl_apple_list.append(cv2.imread("pics/apple/0_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# lapl_orange_list = []
# for i in range(6):
#     lapl_orange_list.append(cv2.imread("pics/orange/0_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# mask_list = []
# for i in range(6):
#     mask_list.append(cv2.imread("pics/mask/gauss_pyramid" + str(i) + ".jpg")[:, :, 0])
#
# blended_pyramid_0 = blend(lapl_apple_list, lapl_orange_list, mask_list)
# for i in range(6):
#     cv2.imwrite("pics/blend/0" + str(i) + ".jpg", blended_pyramid_0[i])
#
# lapl_apple_list = []
# for i in range(6):
#     lapl_apple_list.append(cv2.imread("pics/apple/1_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# lapl_orange_list = []
# for i in range(6):
#     lapl_orange_list.append(cv2.imread("pics/orange/1_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# mask_list = []
# for i in range(6):
#     mask_list.append(cv2.imread("pics/mask/gauss_pyramid" + str(i) + ".jpg")[:, :, 0])
#
# blended_pyramid_0 = blend(lapl_apple_list, lapl_orange_list, mask_list)
# for i in range(6):
#     cv2.imwrite("pics/blend/1" + str(i) + ".jpg", blended_pyramid_0[i])
#
# lapl_apple_list = []
# for i in range(6):
#     lapl_apple_list.append(cv2.imread("pics/apple/2_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# lapl_orange_list = []
# for i in range(6):
#     lapl_orange_list.append(cv2.imread("pics/orange/2_lapl_pyramid_apple" + str(i) + ".jpg")[:, :, 2])
# mask_list = []
# for i in range(6):
#     mask_list.append(cv2.imread("pics/mask/gauss_pyramid" + str(i) + ".jpg")[:, :, 0])
#
# blended_pyramid_0 = blend(lapl_apple_list, lapl_orange_list, mask_list)
# for i in range(6):
#     cv2.imwrite("pics/blend/2" + str(i) + ".jpg", blended_pyramid_0[i])
# ----------------------------------------------------------------------------
blended_pyramid_0 = []
for i in range(6):
    blended_pyramid_0.append(cv2.imread("pics/blend/0" + str(i) + ".jpg")[:, :, 0])

blended_pyramid_1 = []
for i in range(6):
    blended_pyramid_1.append(cv2.imread("pics/blend/1" + str(i) + ".jpg")[:, :, 1])

blended_pyramid_2 = []
for i in range(6):
    blended_pyramid_2.append(cv2.imread("pics/blend/2" + str(i) + ".jpg")[:, :, 2])

agg_0 = collapse(blended_pyramid_0)
agg_1 = collapse(blended_pyramid_1)
agg_2 = collapse(blended_pyramid_2)

a = np.zeros((256, 256, 3))
a[:, :, 0] = agg_0
a[:, :, 1] = agg_1
a[:, :, 2] = agg_2

cv2.imwrite("pics/bob_tree_art_2.jpg", a)
print("done")