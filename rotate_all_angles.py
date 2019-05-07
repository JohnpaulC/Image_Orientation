from image_utils import angleCal, create_images, plot_result

rotate_angle = 10
mode_list = ["SIFT", "SURF", "ORB"]
sift_result = []

for mode in mode_list:
    surf_result = []
    orb_result = []
    name_result = []
    result = []
    for rotate_angle in range(20):
        result_orig, result_rotate, result_rotate_translation, \
        result_perspective, result_correction = create_images(rotate_angle)
        # For Different Feature Descriptor

        print("-----The Descriptor {0:s} is using-----".format(mode))
        # Rotate
        mean_rotate, time = angleCal(result_orig, result_rotate, mode)
        sift_result.append(mean_rotate - rotate_angle)
        print("Rotate: {0:6.3f} in {1:.3f}".format(mean_rotate, time))

        # Rotate and Translation
        mean_rotate_translation, time = angleCal(result_orig, result_rotate_translation, mode)
        sift_result.append(mean_rotate - rotate_angle)
        print("Rotate and translation: {0:6.3f} in {1:.3f}".format(mean_rotate_translation, time))

        # Perspective
        mean_perspective, time = angleCal(result_orig, result_perspective, mode)
        sift_result.append(mean_rotate - rotate_angle)
        print("Perspective: {0:6.3f} in {1:.3f}".format(mean_perspective, time))

        # Correction
        mean_correction, time = angleCal(result_orig, result_correction, mode)
        result.append(mean_correction), name_result.append(mode + " C")
        print("Correction: {0:6.3f} in {1:.3f}".format(mean_correction, time))

    plot_result(result)
