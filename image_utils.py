import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt


def create_images(angle, show_image = False):
    # Change one image and getting from.
    top_x = 100
    top_y = 100
    bottom_x = 500
    bottom_y = 400

    img_orig = cv.imread('figures/office.jpg', cv.IMREAD_COLOR)
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
    img_perspective = cv.warpPerspective(img_rotate, M_p, (w, h))

    M_n = cv.getPerspectiveTransform(pts2, pts1)
    img_correction = cv.warpPerspective(img_perspective, M_n, (w, h))

    ### Get the slice of whole image
    result_orig = img_gray[top_y:bottom_y, top_x:bottom_x]
    result_rotate = img_rotate[top_y:bottom_y, top_x:bottom_x]
    result_rotate_translation = img_rotate_translation[top_y:bottom_y, top_x:bottom_x]
    result_perspective = img_perspective[top_y:bottom_y, top_x:bottom_x]
    result_correction = img_correction[top_y:bottom_y, top_x:bottom_x]


    if show_image:
        plt.subplot(2, 2, 1), plt.axis('off'), plt.imshow(img_gray, cmap='Greys_r'), plt.title("gray image")
        plt.subplot(2, 2, 2), plt.axis('off'), plt.imshow(img_rotate, cmap='Greys_r'), plt.title("rotate image")

        plt.subplot(2, 2, 3), plt.axis('off'), plt.imshow(img_perspective, cmap='Greys_r'), plt.title("Perceptive image")
        plt.subplot(2, 2, 4), plt.axis('off'), plt.imshow(img_correction, cmap='Greys_r'), plt.title("Correction image")
        plt.show()

    return result_orig, result_rotate, result_rotate_translation,\
           result_perspective, result_correction

def angle_cal(img_base, img_rotate, mode = "SURF", show_all_results = False):
    start = time.time()
    if mode == "SIFT":
        # Sift Create and calculate
        sift = cv.xfeatures2d.SIFT_create()
        # Kp is key points
        # des is feature descriptor
        # len(des) = len(Kp) * 128
        kp1, des1 = sift.detectAndCompute(img_base, None)
        kp2, des2 = sift.detectAndCompute(img_rotate, None)

        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    elif mode == "ORB":
        orb = cv.ORB_create()

        kp1, des1 = orb.detectAndCompute(img_base, None)
        kp2, des2 = orb.detectAndCompute(img_rotate, None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    else:
        surf = cv.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(img_base, None)
        kp2, des2 = surf.detectAndCompute(img_rotate, None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    # Cal the Orientation
    rotate_angle = []
    key_point = 25
    # Show the debug file
    if False:
        img = cv.drawMatches(img_base, kp1, img_rotate, kp2, matches[:key_point], None, flags=2)
        cv.imwrite("results/mathing.png", img)
        cv.imshow('match', img), cv.waitKey(), cv.destroyWindow('match')
    for i in range(key_point):
        num = i
        img_index1 = matches[num].queryIdx
        img_index2 = matches[num].trainIdx
        rotate_angle.append(kp1[img_index1].angle - kp2[img_index2].angle)

    end_time = time.time() - start

    # Change the local
    rotate_angle = np.array(rotate_angle)
    rotate_angle = (360 * (rotate_angle < 0) + rotate_angle)
    rotate_angle = np.abs((rotate_angle > 180) * 360 - rotate_angle)
    # Mean result
    mean = np.mean(rotate_angle)
    # Median calculation for the filter the noise
    median =np.median(rotate_angle)
    print("Before " + str(median))
    limit = 1
    median_index = (rotate_angle > max(median - limit, 0)) & (rotate_angle < median + limit)
    median_index = np.array(median_index + 0)
    median = np.mean(rotate_angle[np.nonzero(median_index)])
    print("After " + str(median))

    if show_all_results:
        print("Final result: ")
        print(rotate_angle)
        plot_result_bar(range(len(rotate_angle)), rotate_angle, mean)

    return median, mean, end_time


def plot_result_bar(name_result, result, base_value = None):
    """
    This function is ploting the SIFT, SURF, ORB results bar.
    """
    plt.figure(figsize=(12.8, 9.6))
    plt.bar(name_result, result, width=0.25)
    if base_value is not None:
        plt.plot(name_result, len(name_result) * [base_value], 'r--', linewidth=0.5)
    plt.title("Result in angle " + str(base_value))
    for x, y in zip(name_result, result):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')
    plt.show()
    #plt.savefig("results/" + str(base_value) + ".png")

def plot_result(y, constant = None):
    """
    It plot the list result in sequence as polyline.
    """
    x = np.arange(0, len(y))
    y = np.array(y)
    plt.figure()
    if constant is not  None:
        plt.plot(x, 0 * x + constant)
    plt.plot(x, y)
    plt.show()

def plot_double_result(y1, y2, constant=None):
    """
    It plot the list results in double sequence as polyline.
    """
    x1 = np.arange(0, len(y1))
    y1 = np.array(y1)
    x2 = np.arange(0, len(y2))
    y2 = np.array(y2)
    plt.figure()
    plt.subplot(3, 1, 1)
    if constant is not None:
        plt.plot(x1, 0 * x1 + constant)
    plt.plot(x1, y1)
    plt.subplot(3, 1, 2)
    if constant is not None:
        plt.plot(x2, 0 * x2 + constant)
    plt.plot(x2, y2)
    plt.subplot(3, 1, 3)
    if constant is not None:
        plt.plot(x1, 0 * x1 + constant)
    plt.plot(x1, y1, y2)
    plt.show()

def plot_hist(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()