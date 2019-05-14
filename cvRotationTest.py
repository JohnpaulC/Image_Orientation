from image_utils import angle_cal, create_images, plot_result_bar

rotate_angle = 10
angles = [5, 8, 10, 20, 30]
mode_list = ["SIFT", "SURF", "ORB"]

for rotate_angle in angles:
    result_orig, result_rotate, result_rotate_translation, \
    result_perspective, result_correction = create_images(rotate_angle, show_image = True)
    print("-" * 10 + str(rotate_angle) + "-" * 10)
    name_result = []
    mean_result = []
    median_result = []
    # For Different Feature Descriptor
    for mode in mode_list:
        print("-----The Descriptor {0:s} is using-----".format(mode))
        # Rotate
        median_rotate, mean_rotate, time = angle_cal(result_orig, result_rotate, mode)
        mean_result.append(mean_rotate), name_result.append(mode + " R")
        print("Rotate: {0:6.3f} in {1:.3f}".format(mean_rotate, time))

        # Rotate and Translation
        median_rotate_translation, mean_rotate_translation, time = angle_cal(result_orig, result_rotate_translation, mode)
        mean_result.append(mean_rotate_translation), name_result.append(mode + " T+R")
        print("Rotate and translation: {0:6.3f} in {1:.3f}".format(mean_rotate_translation, time))

        # Perspective
        median_perspective, mean_perspective, time = angle_cal(result_orig, result_perspective, mode)
        mean_result.append(mean_perspective), name_result.append(mode + " P")
        print("Perspective: {0:6.3f} in {1:.3f}".format(mean_perspective, time))

        # Correction
        median_correction, mean_correction, time = angle_cal(result_orig, result_correction, mode)
        mean_result.append(mean_correction), name_result.append(mode + " C")
        print("Correction: {0:6.3f} in {1:.3f}".format(mean_correction, time))

    plot_result_bar(name_result, mean_result, rotate_angle)


