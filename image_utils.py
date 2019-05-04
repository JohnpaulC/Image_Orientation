import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt

print(cv.__version__)


def create_images(angle):
    # Change one image and getting from.
    top_x = 100
    top_y = 100
    bottom_x = 500
    bottom_y = 400

    img_orig = cv.imread('office.jpg', cv.IMREAD_COLOR)
    (h, w) = img_orig.shape[:2]
    #print("The image size is {0:d} * {1:d}".format(h, w))
    img_gray = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)


    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rotate = cv.warpAffine(img_gray, M, (w, h))

    # Do the translation on rotated image
    x_translation, y_translation = 20, 20
    T = np.float32([
        [1, 0, x_translation],
        [0, 1, y_translation]])
    img_rotate_translation = cv.warpAffine(img_rotate, T, (w, h))


    ### Do the perspective on Original Image
    num_distortion = 50
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, num_distortion], [w, 0], [0, h - num_distortion], [w, h]])
    M_p = cv.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv.warpPerspective(img_gray, M_p, (w, h))

    M_n = cv.getPerspectiveTransform(pts2, pts1)
    img_correction = cv.warpPerspective(img_perspective, M_n, (w, h))

    img_rotation_per = cv.warpPerspective(img_rotate, M_p, (w, h))
    img_rotation_per_cor = cv.warpPerspective(img_rotation_per, M_n, (w, h))

    ### Get the slice of whole image
    result_orig = img_gray[top_y:bottom_y, top_x:bottom_x]
    result_rotate = img_rotate[top_y:bottom_y, top_x:bottom_x]
    result_rotate_translation = img_rotate_translation[top_y:bottom_y, top_x:bottom_x]
    result_perspective = img_perspective[top_y:bottom_y, top_x:bottom_x]
    result_correction = img_correction[top_y:bottom_y, top_x:bottom_x]
    result_rotation_per = img_rotation_per[top_y:bottom_y, top_x:bottom_x]
    result_rotation_per_cor = img_rotation_per_cor[top_y:bottom_y, top_x:bottom_x]


    # DEBUG: Image show
    if False:
        cv.imshow("result_orig", result_orig)
        cv.imshow("result_rotate", result_rotate)
        cv.imshow("result_rotate_translation", result_rotate_translation)
        cv.imshow("result_perspective", result_perspective)
        cv.imshow("result_correction", result_correction)

        cv.waitKey(0)
        cv.destroyAllWindows()

    elif True:
        plt.subplot(3, 2, 1), plt.axis('off'), plt.imshow(img_gray, cmap='Greys_r'), plt.title("gray image")
        plt.subplot(3, 2, 2), plt.axis('off'), plt.imshow(img_rotate, cmap='Greys_r'), plt.title("rotate image")

        plt.subplot(3, 2, 3), plt.axis('off'), plt.imshow(img_perspective, cmap='Greys_r'), plt.title("Perceptive image")
        plt.subplot(3, 2, 4), plt.axis('off'), plt.imshow(img_correction, cmap='Greys_r'), plt.title("Correction image")

        plt.subplot(3, 2, 5), plt.axis('off'), plt.imshow(img_rotation_per, cmap='Greys_r'), plt.title("Rotate Perceptive image")
        plt.subplot(3, 2, 6), plt.axis('off'), plt.imshow(img_rotation_per_cor, cmap='Greys_r'), plt.title("Rotate Correction image")
        plt.show()

    return result_orig, result_rotate, result_rotate_translation,\
           result_perspective, result_correction,\
           result_rotation_per, result_rotation_per_cor


def angleCal(img_base, img_rotate, show_all_results = False):
    start = time.time()
    # Sift Create and calculate
    sift = cv.xfeatures2d.SIFT_create()
    # Kp is key points
    # des is feature descriptor
    # len(des) = len(Kp) * 128
    kp1, des1 = sift.detectAndCompute(img_base, None)
    kp2, des2 = sift.detectAndCompute(img_rotate, None)

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if show_all_results:
        img = cv.drawMatches(img_base, kp1, img_rotate, kp2, matches[:10], None, flags=2)
        cv.imshow('match', img)
        cv.waitKey()
        cv.destroyWindow('match')

    if False:
        plt.imshow(img), plt.show()
        print(type(matches))
        print(len(matches))
        print(type(matches[1]))

    rotate_angle = []
    num_keypoint = 10
    for i in range(num_keypoint):
        num = i
        img_index1 = matches[num].queryIdx
        img_index2 = matches[num].trainIdx
        rotate_angle.append(kp1[img_index1].angle - kp2[img_index2].angle)

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

    end_time = time.time() - start
    print("Total time: " + str(end_time))

    rotate_angle = np.array(rotate_angle)
    rotate_angle = (360 * (rotate_angle < 0) + rotate_angle)
    rotate_angle = np.abs((rotate_angle > 359) * 360 - rotate_angle)
    mean = np.mean(rotate_angle)

    if show_all_results:
        print("Final result: ")
        print(rotate_angle)
        print("Mean result: ")
        print(mean)

    return mean

def plot_result(y, constant = None):
    x = np.arange(0, len(y))
    y = np.array(y)
    plt.figure()
    if constant is not None:
        plt.plot(x, 0 * x + constant)
    plt.plot(x, y)
    plt.show()

def plot_result_bar(y, constant = None):
    x = np.arange(0, len(y))
    y = np.array(y)
    plt.figure()
    if constant is not None:
        plt.plot(x, 0 * x + constant)
    plt.plot(x, y)
    plt.show()