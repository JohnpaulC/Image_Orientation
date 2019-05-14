import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

from image_utils import angle_cal, create_images

np.set_printoptions(suppress=True)

img_base = cv.imread('figures/base.jpg')
img_rotate = cv.imread('figures/rotate_15.jpg')
img_base = cv.cvtColor(img_base, cv.COLOR_BGR2GRAY)
img_rotate = cv.cvtColor(img_rotate, cv.COLOR_BGR2GRAY)

w, h = img_base.shape[:2]
rotate_angle = -15
scale_ratio = 7
img_base = cv.resize(img_base, (int(h/scale_ratio), int(w/scale_ratio)))
img_rotate = cv.resize(img_rotate, (int(h/scale_ratio), int(w/scale_ratio)))
w, h = img_base.shape[:2]
center = (h / 2, w / 2)
M = cv.getRotationMatrix2D(center, rotate_angle, 1)
img_rotate_base = cv.warpAffine(img_base, M, (h, w))
if True:
    print("The resize image shape is ")
    print(img_base.shape)
    plt.figure()
    plt.subplot(3,1,1), plt.imshow(img_base, "Greys_r")
    plt.subplot(3,1,2), plt.imshow(img_rotate, "Greys_r")
    plt.subplot(3,1,3), plt.imshow(img_rotate_base, "Greys_r")
    plt.show()

# Result Calculation
median, mean, time = angle_cal(img_base, img_rotate, "SIFT", show_all_results= 0)
print("SIFT Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))
median, mean, time = angle_cal(img_base, img_rotate_base, "SIFT", show_all_results= 0)
print("SIFT Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))