import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print(cv.__version__)

if True:
    img_base = cv.imread('dog.jpg', cv.IMREAD_COLOR)
    (h, w) = img_base.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, 10, 1)
    img_rotate = cv.warpAffine(img_base, M, (w, h))

elif True:
    img_base = cv.imread("./base_element.jpg", cv.IMREAD_COLOR)
    img_rotate = cv.imread("./rotate_element.jpg", cv.IMREAD_COLOR)

else:
    img_base = cv.imread("./1.jpeg", cv.IMREAD_COLOR)
    img_rotate = cv.imread("./2.jpeg", cv.IMREAD_COLOR)


img_base = cv.cvtColor(img_base, cv.COLOR_BGR2GRAY)
img_rotate = cv.cvtColor(img_rotate, cv.COLOR_BGR2GRAY)

if False:
    cv.imshow("Orig", img_base)
    cv.imshow("Rotate", img_rotate)
    cv.waitKey(0)
    cv.destroyAllWindows()

print(img_base.shape)
print(img_rotate.shape)

# Sift Create and calculate
sift = cv.xfeatures2d.SIFT_create()
# Kp is key points
# des is descriptor
# len(des) = len(Kp) * 128
kp1, des1 = sift.detectAndCompute(img_base, None)
kp2, des2 = sift.detectAndCompute(img_rotate, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck = True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)
img = cv.drawMatches(img_base, kp1, img_rotate, kp2, matches[:1], None, flags = 2)
cv.imwrite('oh.jpg', img)

if False:
    plt.imshow(img), plt.show()
    print(type(matches))
    print(len(matches))
    print(type(matches[1]))

rotate_angle = []
for i in range(20):
    num = i
    img_index1 = matches[num].queryIdx
    img_index2 = matches[num].trainIdx
    if False:
        print("-" * 20)
        print(matches[num])
        print(matches[num].distance)
        print(matches[num].imgIdx)
        print(matches[num].queryIdx)
        print(matches[num].trainIdx)
        print(kp1[img_index1].pt)
        print(kp1[img_index1].angle)
        print(kp2[img_index2].pt)
        print(kp2[img_index2].angle)
    rotate_angle.append(kp2[img_index2].angle - kp1[img_index1].angle)


rotate_angle = np.array(rotate_angle)
rotate_angle = (360 * (rotate_angle < 0) + rotate_angle)
print("Final result: ")
print(rotate_angle)
mean = np.mean(rotate_angle)
print("Mean result: ")
print(mean)
