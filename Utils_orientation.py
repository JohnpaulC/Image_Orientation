import cv2
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from utils.datasets import *
from utils.utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
Utils function
"""
def create_images(angle, show_image = False):
    # Change one image and getting from.
    top_x = 100
    top_y = 100
    bottom_x = 500
    bottom_y = 400

    img_orig = cv2.imread('figures/office.jpg', cv2.IMREAD_COLOR)
    (h, w) = img_orig.shape[:2]
    #print("The image size is {0:d} * {1:d}".format(h, w))
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rotate = cv2.warpAffine(img_gray, M, (w, h))

    # Do the translation on rotated image
    x_translation, y_translation = 20, 20
    T = np.float32([
        [1, 0, x_translation],
        [0, 1, y_translation]])
    img_rotate_translation = cv2.warpAffine(img_rotate, T, (w, h))


    ### Do the perspective on Original Image
    num_distortion = 50
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[0, num_distortion], [w, 0], [0, h - num_distortion], [w, h]])
    M_p = cv2.getPerspectiveTransform(pts1, pts2)
    img_perspective = cv2.warpPerspective(img_rotate, M_p, (w, h))

    M_n = cv2.getPerspectiveTransform(pts2, pts1)
    img_correction = cv2.warpPerspective(img_perspective, M_n, (w, h))

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

def angle_cal(img_base, img_rotate, mode = "SURF", show_results = False, show_images = False):
    """
        This is a Orietation calculation function using Descriptor SIFT, SURF or ORB.

        Parameters:
            img_base, img_rotate: two numpy image file which have a orientation difference.
            mode(SIFT, SURF, ORB): the basic descriptor and matching mode
            show_results: Print the processing results and plot the bar graph
            show_images: show the processing match image

        Returns:
            Median: the processed median value
            Mean: Mean value
            time: Processing of whole process
        """
    start = time.time()
    # Here the difference mode will using different key points description and matching method
    if mode == "SIFT":
        # Sift Create and calculate
        sift = cv2.xfeatures2d.SIFT_create()
        # Kp is key points
        # des is feature descriptor
        # len(des) = len(Kp) * 128
        kp1, des1 = sift.detectAndCompute(img_base, None)
        kp2, des2 = sift.detectAndCompute(img_rotate, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif mode == "ORB":
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img_base, None)
        kp2, des2 = orb.detectAndCompute(img_rotate, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        surf = cv2.xfeatures2d.SURF_create()
        kp1, des1 = surf.detectAndCompute(img_base, None)
        kp2, des2 = surf.detectAndCompute(img_rotate, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match the key points and sort them by distance
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Cal the Orientation
    rotate_angle = []
    key_point = 25
    limit = 2

    for i in range(key_point):
        num = i
        img_index1 = matches[num].queryIdx
        img_index2 = matches[num].trainIdx
        rotate_angle.append(kp1[img_index1].angle - kp2[img_index2].angle)

    # Change the results into 0~180
    rotate_angle = np.array(rotate_angle)
    rotate_angle = (360 * (rotate_angle < 0) + rotate_angle)
    rotate_angle = np.abs((rotate_angle > 180) * 360 - rotate_angle)
    # Mean result
    mean = np.mean(rotate_angle)
    # Median calculation for the filter the noise
    median =np.median(rotate_angle)
    median_index = (rotate_angle > max(median - limit, 0)) & (rotate_angle < median + limit)
    median_index = np.array(median_index + 0)
    median = np.mean(rotate_angle[np.nonzero(median_index)])
    end_time = time.time() - start

    if show_images:
        img = cv2.drawMatches(img_base, kp1, img_rotate, kp2, matches[:key_point], None, flags=2)
        #cv2.imwrite("results/mathing.png", img)
        cv2.imshow('matching images', img), cv2.waitKey(), cv2.destroyWindow('match')
    if show_results:
        print("Final result: ")
        print(rotate_angle)
        plot_result_bar(range(len(rotate_angle)), rotate_angle, mean)

    return median, mean, end_time

def object_detection(model, img_file):
    """
    This is a Object detection function using CNN.

    Parameters:
        model: The YOLO darknet model
        img_file: image file path the pre and post processing is in the function

    Returns:
        The scalable detection parameter which is top_left and bottom_right corner
        And the confidence and classification confidence
    """
    conf_thres = 0.5
    nms_thres = 0.5
    # Preprocess the image
    img0 = cv2.imread(img_file)
    # Padded resize
    img, _, _, _ = letterbox(img0)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    # Get detections
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(img)
    detections = non_max_suppression(pred, conf_thres, nms_thres)[0]
    # Rescale the detection into original size
    scale_coords(416, detections[:, :4], img0.shape).round()
    return detections

def object_capture(base_file, rotate_file,
                   bool_cap = False, detection_index = 0,
                   detection_base = None, detection_rotate = None):
    if not bool_cap:
        img_base = cv2.imread(base_file)
        img_rotate = cv2.imread(rotate_file)
    else:
        img_base = cv2.imread(base_file)
        img_rotate = cv2.imread(rotate_file)
        # Finding the corresponding object
        object_index = np.where(detection_rotate[:, 6] == detection_base[detection_index, 6])
        # Base image capture
        x1, y1, x2, y2 = np.uint64(detection_base[detection_index, 0:4])
        img_base = img_base[y1:y2, x1:x2, :]
        # Rotate image capture
        x1, y1, x2, y2 = np.uint64(detection_rotate[object_index, 0:4].squeeze())
        img_rotate = img_rotate[y1:y2, x1:x2, :]

    return img_base, img_rotate


def histogram_gradient(img, img_rotate, mag_thres = 20, bin_num = 360):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rotate = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)

    for img in [img, img_rotate]:
        sobelx=cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0)
        sobely=cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1)
        gradient = np.arctan2(sobely, sobelx) * 180 / np.pi
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        mag_thres = magnitude.max() / 2
        gradient = (magnitude > mag_thres) * gradient
        gradient = (gradient < 0) * 360 + gradient
        hist, bins = np.histogram(gradient, bin_num)
        hist = hist[1:]
        
        plt.plot(hist)
    #plt.xlim([1, bin_num])
    #plt.ylim([0, 15000])
    plt.show()
