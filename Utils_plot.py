import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
Ploting function
"""
def plot_result_bar(name_result, result, base_value = None):
    """
    This function is ploting the SIFT, SURF, ORB results bar.
    """
    plt.figure(figsize=(12.8, 9.6))
    plt.bar(name_result, result, width=0.25)
    if base_value is not None:
        plt.plot(name_result, len(name_result) * [base_value], 'r--', linewidth=1)
    #plt.title("Average orientation estimation in {0:.3f} ".format(base_value), fontsize = 16)
    plt.title("The Results of Orientation", fontsize = 20)
    plt.xlabel('Index of keypoint', fontsize = 20)
    plt.ylabel('Orientation estimation', fontsize = 20)
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
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def cv_show_images(img, img_2 = None):
    cv2.imshow("First image", img)
    if img_2 is not None:
        cv2.imshow("Second image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()