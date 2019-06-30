import time
import numpy as np
import cv2
import torch

from utils.datasets import letterbox
from utils.models import *
from Utils_orientation import *

# Image path
base_file = "dog.jpg"
rotate_file = "dog_rotate.jpg"

# Configuration file path
cfg = 'cfg/yolov3.cfg'
data_cfg = 'cfg/coco.data'
weights = 'cfg/yolov3.pt'

# Model parameter
img_size=416
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

# Image Orientation Calculation
show_results = False
show_images = True
mode = "SIFT"
# The index of object using to calculation the capture image
detection_index = 2

if True:
    img_base, img_rotate = object_capture(base_file, rotate_file)
    histogram_gradient(img_base, img_rotate)
    #cv2.imwrite('cap.jpg', img_base)
    #cv2.imwrite('rotate.jpg', img_rotate)
    median, mean, time = angle_cal(img_base, img_rotate, mode, show_results= show_results, show_images= show_images)
    print(mode + "real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))

img_base, img_rotate = object_capture(base_file, rotate_file,
                   bool_cap = True, detection_index = detection_index,
                   detection_base = detection_base, detection_rotate = detection_rotate)
histogram_gradient(img_base, img_rotate)
median, mean, time = angle_cal(img_base, img_rotate, mode, show_results= show_results, show_images= show_images)
print(mode + "real Result: median({0:6.3f}), mean({1:6.3f}) in {2:.3f}".format(median, mean, time))