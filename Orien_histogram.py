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

# Model parameter
img_size=640
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize model and load weights
model = Darknet(cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()

# Detection Process Transform tensor into numpy array
if device == torch.device("cpu"):
    detection_base = object_detection(model, base_file).numpy()
    detection_rotate = object_detection(model, rotate_file).numpy()
else:
    detection_base = object_detection(model, base_file).cpu().numpy()
    detection_rotate = object_detection(model, rotate_file).cpu().numpy()
    

######
# Image Orientation Calculation
######
show_results = False
show_images = False
mode = "SIFT"
# The index of object using to calculation the capture image
detection_index = 2
# The HoG parameters
bins = 360
thres = 50

print(detection_base.shape)
print(detection_rotate.shape)

object_num = min(detection_base.shape[0], detection_rotate.shape[0])

img_base, img_rotate = object_capture(base_file, rotate_file)

start = time.time()

hist_base = HoG_cal(img_base, mag_thres= thres, bin_num= bins)
hist_rotate = HoG_cal(img_rotate, mag_thres= thres, bin_num= bins)

# Using HoG to calculate the angel
angle = angle_HoG(hist_base, hist_rotate, limits = 10)
print("The HoG result is {0:2d} in {1:.4f}".format(angle, time.time() - start))
median, mean, t = angle_cal(img_base, img_rotate, mode, show_results= show_results, show_images= show_images)
print(mode + " real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.4f}".format(median, mean, t))

for detection_index in range(object_num):
    img_base, img_rotate = object_capture(base_file, rotate_file,
                       bool_cap = True, detection_index = detection_index,
                       detection_base = detection_base, detection_rotate = detection_rotate)
    start = time.time()
    hist_base = HoG_cal(img_base, mag_thres= thres, bin_num= bins)
    hist_rotate = HoG_cal(img_rotate, mag_thres= thres, bin_num= bins)
    
    # Using HoG to calculate the angel
    angle = angle_HoG(hist_base, hist_rotate, limits = 10)
    print("The HoG result is {0:2d} in {1:.4f}".format(angle, time.time() - start))

    # Using Feature descriptor
    median, mean, t = angle_cal(img_base, img_rotate, mode, show_results= show_results, show_images= show_images)
    print(mode + " real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.4f}".format(median, mean, t))

    #plt.plot(hist_base, 'r')
    #plt.plot(hist_rotate, 'g')
    #plt.show()