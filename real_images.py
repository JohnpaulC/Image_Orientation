import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

from image_utils import angle_cal, create_images

np.set_printoptions(suppress=True)

img_base = cv.imread('figures/0.jpeg')
img_rotate_orig = cv.imread('figures/10.jpeg')
img_base = cv.cvtColor(img_base, cv.COLOR_BGR2GRAY)
img_rotate_orig = cv.cvtColor(img_rotate_orig, cv.COLOR_BGR2GRAY)
img_rotate = cv.GaussianBlur(img_rotate_orig,(3,3),0)

w, h = img_base.shape[:2]
rotate_angle = -15
scale_ratio = 5
img_base = cv.resize(img_base, (int(h/scale_ratio), int(w/scale_ratio)))
img_rotate = cv.resize(img_rotate, (int(h/scale_ratio), int(w/scale_ratio)))
w, h = img_base.shape[:2]
center = (h / 2, w / 2)
M = cv.getRotationMatrix2D(center, rotate_angle, 1)
img_rotate_base = cv.warpAffine(img_base, M, (h, w))

show_hist = False
if show_hist:
    #hist = cv.calcHist([img_rotate], [0], None, [256], [0,256])
    plt.figure()
    plt.subplot(3,1,1)
    plt.hist(img_rotate_orig.ravel(),256,[0,256])
    plt.subplot(3,1,2)
    plt.hist(img_rotate.ravel(),256,[0,256])
    plt.ylim([0, 50000])
    plt.subplot(3,1,3)
    plt.hist(img_rotate_base.ravel(),256,[0,256])
    plt.show()

if False:
    print("The resize image shape is ")
    print(img_base.shape)
    plt.figure()
    plt.subplot(3,1,1), plt.imshow(img_base, "Greys_r")
    plt.subplot(3,1,2), plt.imshow(img_rotate, "Greys_r")
    plt.subplot(3,1,3), plt.imshow(img_rotate_base, "Greys_r")
    plt.show()

# Result Calculation
median, mean, time = angle_cal(img_base, img_rotate, "SIFT", show_all_results= 0)
print("SIFT real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))
median, mean, time = angle_cal(img_base, img_rotate_base, "SIFT", show_all_results= 0)
print("SIFT fake Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))