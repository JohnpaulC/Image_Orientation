import cv2 as cv
import numpy as np
from image_utils import angleCal, creat_images


result_orig, result_rotate, result_rotate_translation, result_perspective = creat_images()

mean_rotate = angleCal(result_orig, result_rotate)
print("Mean between rotate is {0:f}".format(mean_rotate))

mean_rotate_translation = angleCal(result_orig, result_rotate_translation)
print("Mean between rotate and translation is {0:f}".format(mean_rotate_translation))

mean_perspective = angleCal(result_orig, result_perspective)
print("Mean between perspective is {0:f}".format(mean_perspective))