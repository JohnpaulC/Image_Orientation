import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

print(cv.__version__)

# Change one image and getting from.

img_orig = cv.imread('dog.jpg', cv.IMREAD_COLOR)
(h, w) = img_orig.shape[:2]
print(h, w)
img_gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
plt.subplot(3, 2, 1), plt.imshow(img_gray, cmap='Greys_r'), plt.title("gray image")

center = (w / 2, h / 2)
angle = 5
M = cv.getRotationMatrix2D(center, angle, 1)
print(M)
img_rotate = cv.warpAffine(img_gray, M, (w, h))
print(img_rotate.shape)
plt.subplot(3, 2, 2), plt.imshow(img_rotate, cmap='Greys_r'), plt.title("rotate image")

# Do the translation
# x_translation, y_translation = 20, 20
# T = np.float32([
#     [1, 0, x_translation],
#     [0, 1, y_translation]])
# img_rotate = cv.warpAffine(img_rotate, T, (w, h))
# cv.imshow('fxx3', img_rotate)

# pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
# pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
# M = cv.getPerspectiveTransform(pts2, pts1)
# print(M)
# img_perceptive = cv.warpPerspective(img_gray, M, (w, h))
# cv.imshow('fxx4', img_perceptive)

plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
