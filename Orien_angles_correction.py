import matplotlib.pyplot as plt
from Utils_orientation import angle_cal, create_images
from Utils_plot import plot_double_result

rotate_angle = 15
mode_list = ["SIFT", "ORB"]
sift_result_p = []
sift_result_c = []
surf_result_p = []
surf_result_c = []
orb_result_p = []
orb_result_c = []

for mode in mode_list:
    result_perspective = []
    result_correction = []
    for rotate_angle in range(30):
        img_orig, _, _, img_perspective, img_correction = create_images(rotate_angle)
        # For Different Feature Descriptor
        # Perspective
        _, mean_perspective, time = angle_cal(img_orig, img_perspective, mode)
        result_perspective.append(abs(mean_perspective - rotate_angle))
        print("Perspective: {0:6.3f} in {1:.3f}".format(mean_perspective, time))

        # Correction
        _, mean_correction, time = angle_cal(img_orig, img_correction, mode)
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

if False:
    plot_double_result(sift_result_p, sift_result_c)
    plot_double_result(surf_result_p, surf_result_c)
    plot_double_result(orb_result_p, orb_result_c)
else:
    plt.subplot(2, 1, 1)
    plt.plot(sift_result_p, 'r', label = 'Perspective')
    plt.plot(sift_result_c, 'g', label = 'Correction')
    plt.legend()
    plt.title('The error using SIFT matching', loc = 'left')

    # plt.subplot(3, 1, 2)
    # plt.plot(surf_result_p, 'r', label = 'Perspective')
    # plt.plot(surf_result_c, 'g', label = 'Correction')
    # plt.legend()
    # plt.title('The error using SURF matching', loc = 'left')

    plt.subplot(2, 1, 2)
    plt.plot(orb_result_p, 'r', label = 'Perspective')
    plt.plot(orb_result_c, 'g', label = 'Correction')
    plt.legend()
    plt.title('The error using ORB matching', loc = 'left')
    plt.show()


    plt.plot(sift_result_c, label = 'SIFT')
    plt.plot(orb_result_c, label = 'ORB')
    plt.legend()
    plt.title('The errors between SIFT and ORB')
    plt.ylim((0, 1))
    plt.show()

    