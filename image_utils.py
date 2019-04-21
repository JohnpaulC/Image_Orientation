import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

print(cv.__version__)


def creat_images():
    # Change one image and getting from.
    top_x = 100
    top_y = 100
    bottom_x = 600
    bottom_y = 400

    img_orig = cv.imread('dog.jpg', cv.IMREAD_COLOR)
    print(img_orig.shape)
    (h, w) = img_orig.shape[:2]
    print("The image size is {0:d} * {1:d}".format(h, w))
    img_gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
    plt.subplot(3, 2, 1), plt.imshow(img_gray, cmap='Greys_r'), plt.title("gray image")

    center = (w / 2, h / 2)
    angle = 5
    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rotate = cv.warpAffine(img_gray, M, (w, h))
    print(img_rotate.shape)
    plt.subplot(3, 2, 2), plt.imshow(img_rotate, cmap='Greys_r'), plt.title("rotate image")

    # Do the translation
    x_translation, y_translation = 20, 20
    T = np.float32([
        [1, 0, x_translation],
        [0, 1, y_translation]])
    img_rotate_translation = cv.warpAffine(img_rotate, T, (w, h))
    plt.subplot(3, 2, 3), plt.imshow(img_rotate_translation, cmap='Greys_r'), plt.title("Translate rotate image")

    pts1 = np.float32([[0, 0], [w, 0], [0, h], [h, w]])
    pts2 = np.float32([[0, 50], [w, 0], [0, h - 100], [h, w]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    print(M)

    img_perspective = cv.warpPerspective(img_gray, M, (w, h))
    plt.subplot(3, 2, 4), plt.imshow(img_perspective, cmap='Greys_r'), plt.title("Perceptive image")

    result_orig = img_gray[top_y:bottom_y, top_x:bottom_x]
    result_rotate = img_rotate[top_y:bottom_y, top_x:bottom_x]
    result_rotate_translation = img_rotate_translation[top_y:bottom_y, top_x:bottom_x]
    result_perspective = img_perspective[top_y:bottom_y, top_x:bottom_x]
    if False:
        cv.imshow("1", result_orig)
        cv.imshow("2", result_rotate)
        cv.imshow("3", result_rotate_translation)
        cv.imshow("123", result_perspective)

        plt.show()
        cv.waitKey(0)
        cv.destroyAllWindows()

    return result_orig, result_rotate, result_rotate_translation, result_perspective


def angleCal(img_base, img_rotate):
    start = time.time()
    # Sift Create and calculate
    sift = cv.xfeatures2d.SIFT_create()
    # Kp is key points
    # des is descriptor
    # len(des) = len(Kp) * 128
    kp1, des1 = sift.detectAndCompute(img_base, None)
    kp2, des2 = sift.detectAndCompute(img_rotate, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if False:
        img = cv.drawMatches(img_base, kp1, img_rotate, kp2, matches[:1], None, flags=2)
        cv.imwrite('oh.jpg', img)
        plt.imshow(img), plt.show()
        print(type(matches))
        print(len(matches))
        print(type(matches[1]))

    rotate_angle = []
    num_keypoint = 20
    for i in range(num_keypoint):
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
        rotate_angle.append(kp1[img_index1].angle - kp2[img_index2].angle)
    end_time = time.time() - start
    print("Total time: " + str(end_time))

    rotate_angle = np.array(rotate_angle)
    rotate_angle = (360 * (rotate_angle < 0) + rotate_angle)
    if False:
        print("Final result: ")
        print(rotate_angle)

    mean = np.mean(rotate_angle)
    print("Mean result: ")
    print(mean)

    return mean