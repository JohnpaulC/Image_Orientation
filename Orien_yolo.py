import time
import numpy as np
import cv2
import torch

from utils.datasets import letterbox
from utils.models import *
from Utils_orientation import object_detection, object_capture
from Utils_orientation import angle_HoG, HoG_cal, angle_cal

# Image path
base_file = "figures/71.jpg"
rotate_file = "figures/99.jpg"

# Configuration file path
cfg = 'cfg/yolov3-spp.cfg'
data_cfg = 'cfg/coco.data'
weights = 'cfg/best.pt'
data ='cfg/coco.data'

# Model parameter
img_size=640
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Initialize model and load weights
model = Darknet(cfg, img_size)
model.load_state_dict(torch.load(weights, map_location=device)['model'])
model.to(device).eval()
# This line is for model load to GPU
stupid_predict = object_detection(model, "figures/dog.jpg")

start = time.time()
# Detection Process Transform tensor into numpy array
if device == torch.device("cpu"):
    detection_base = object_detection(model, base_file).numpy()
    detection_rotate = object_detection(model, rotate_file).numpy()
else:
    detection_base = object_detection(model, base_file).cpu().numpy()
    detection_rotate = object_detection(model, rotate_file).cpu().numpy()
print("Detection time comsuming: {0:.4f}".format(time.time() - start))
# Image Orientation Calculation
show_results = False
show_images = False
mode = "SIFT"
# The index of object using to calculation the capture image
detection_index = 0

img_base, img_rotate = object_capture(base_file, rotate_file,
                   bool_cap = True, detection_index = detection_index,
                   detection_base = detection_base, detection_rotate = detection_rotate)
print("Detection time comsuming: {0:.4f}".format(time.time() - start))
hist_base = HoG_cal(img_base)
hist_rotate = HoG_cal(img_rotate)
Hog_angle = angle_HoG(hist_base, hist_rotate)
print("The HoG result is {0:2d} in {1:.4f}".format(Hog_angle, time.time() - start))

median, mean, time = angle_cal(img_base, img_rotate, mode, show_results= show_results, show_images= show_images)
print(mode + "real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))