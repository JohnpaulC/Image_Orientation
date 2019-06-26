#from image_utils import angle_cal, create_images, plot_double_result
from Utils_orientation import angle_cal, create_images, plot_double_result

rotate_angle = 15
mode_list = ["SIFT", "SURF", "ORB"]
sift_result_p = []
sift_result_c = []
surf_result_p = []
surf_result_c = []
orb_result_p = []
orb_result_c = []

for mode in mode_list:
    result_perspective = []
    result_correction = []
    for rotate_angle in range(45):
        img_orig, _, _, img_perspective, img_correction = create_images(rotate_angle)
        # For Different Feature Descriptor
        # Perspective
        mean_perspective, time = angle_cal(img_orig, img_perspective, mode)
        result_perspective.append(abs(mean_perspective - rotate_angle))
        print("Perspective: {0:6.3f} in {1:.3f}".format(mean_perspective, time))

        # Correction
        mean_correction, time = angle_cal(img_orig, img_correction, mode)
        result_correction.append(abs(mean_correction - rotate_angle))
        print("Correction: {0:6.3f} in {1:.3f}".format(mean_correction, time))

    if mode == "SIFT":
        sift_result_p = result_perspective
        sift_result_c = result_correction
    elif mode == "SURF":
        surf_result_p = result_perspective
        surf_result_c = result_correction
    else:
        orb_result_p = result_perspective
        orb_result_c = result_correction

plot_double_result(sift_result_p, sift_result_c)
plot_double_result(surf_result_p, surf_result_c)
plot_double_result(orb_result_p, orb_result_c)

