import matplotlib.pyplot as plt
from Utils_orientation import angle_cal, create_images
from Utils_plot import plot_double_result

import time
import numpy as np
import cv2
import torch

from utils.datasets import letterbox
from utils.models import *
from Utils_orientation import *

# Image path
base_file = 'figures/' + str(71) + '.jpg'
rotate_file = 'figures/' + str(99) + '.jpg'

# Configuration file path
cfg = 'cfg/yolov3.cfg'
data_cfg = 'cfg/coco.data'
weights = 'cfg/yolov3.pt'

if False:
    # Model parameter
    img_size=640
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Initialize model and load weights
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()

img_orig = cv2.imread(base_file)
(h, w) = img_orig.shape[:2]
center = (w / 2, h / 2)
img_base = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

bins = 360
thres = 50
# The translation angles
angle_limits = 15

angle_estimation = []
for rotate_angle in range(1, 11):
   
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1)
    img_rotate = cv2.warpAffine(img_base, M, (w, h))
    start = time.time()

    hist_base = HoG_cal(img_base, mag_thres= thres, bin_num= bins)
    hist_rotate = HoG_cal(img_rotate, mag_thres= thres, bin_num= bins)

    if rotate_angle % 3 == 0 and True:
        plot_index = rotate_angle / 3
        plt.subplot(2, 3, plot_index)
        plt.imshow(img_rotate, cmap= 'Greys_r')
        plt.title("Rotate angle in " + str(rotate_angle))
        plt.subplot(2, 3, plot_index + 3) 
        plt.plot(hist_base, 'r', label = 'Angle Historgram of Original Image')
        plt.plot(hist_rotate, 'g', label = 'Angle Historgram of Rotate Image')
        plt.legend()
        plt.ylim((0, 800))
        plt.title("Histogram of Angles")
        
    # Using HoG to calculate the angel
    angle = angle_HoG(hist_base, hist_rotate, limits = angle_limits)
    print("The HoG estimation when ({0:2d}) is {1:2d} in {2:.4f}s" \
        .format(rotate_angle ,angle, time.time() - start))

plt.show()
