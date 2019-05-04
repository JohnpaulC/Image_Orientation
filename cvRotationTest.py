from image_utils import angleCal, create_images, plot_result_bar

rotate_angle = 10

result_orig, result_rotate, result_rotate_translation,\
result_perspective, result_correction = create_images(rotate_angle)

name_result = []
result = []

# Rotate
mean_rotate, time = angleCal(result_orig, result_rotate)
result.append(mean_rotate), name_result.append("Rotate")
print("Rotate: {0:6.3f} in {1:.3f}".format(mean_rotate, time))

# Rotate and Translation
mean_rotate_translation, time = angleCal(result_orig, result_rotate_translation)
result.append(mean_rotate_translation), name_result.append("Trans+Rotate")
print("Rotate and translation: {0:6.3f} in {1:.3f}".format(mean_rotate_translation, time))

# Perspective
mean_perspective, time = angleCal(result_orig, result_perspective)
result.append(mean_perspective), name_result.append("Perspective")
print("Perspective: {0:6.3f} in {1:.3f}".format(mean_perspective, time))

# Correction
mean_correction, time = angleCal(result_orig, result_correction)
result.append(mean_correction), name_result.append("Correction")
print("Correction: {0:6.3f} in {1:.3f}".format(mean_correction, time))

plot_result_bar(name_result, result, rotate_angle)


