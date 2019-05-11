import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from image_utils import angle_cal, create_images

np.set_printoptions(suppress=True)

img_base = cv.imread('base.jpg')
img_rotate = cv.imread('rotate_15.jpg')
img_base = cv.cvtColor(img_base, cv.COLOR_BGR2GRAY)
img_rotate = cv.cvtColor(img_rotate, cv.COLOR_BGR2GRAY)
print(img_base.shape)
plt.figure()
plt.subplot(2,1,1), plt.imshow(img_base, "Greys_r")
plt.subplot(2,1,2), plt.imshow(img_rotate, "Greys_r")
plt.show()

print("The Original shape is ")
print(img_base.shape)

w, h = img_base.shape[0:2]
print(w)
print(h)

img_base = cv.resize(img_base, (int(h/5), int(w/5)))
img_rotate = cv.resize(img_rotate, (int(h/5), int(w/5)))

print("The image shape is ")
print(img_base.shape)
plt.figure()
plt.subplot(2,1,1), plt.imshow(img_base, "Greys_r")
plt.subplot(2,1,2), plt.imshow(img_rotate, "Greys_r")
plt.show()


# Result Calculation
mean, time = angle_cal(img_base, img_rotate, "SIFT", show_all_results= True)
print("SIFT Result: {0:6.3f} in {1:.3f}".format(mean, time))
mean, time = angle_cal(img_base, img_rotate, "SURF", show_all_results= True)
print("SURF Result: {0:6.3f} in {1:.3f}".format(mean, time))
mean, time = angle_cal(img_base, img_rotate, "ORB", show_all_results= True)
print("ORB Result: {0:6.3f} in {1:.3f}".format(mean, time))