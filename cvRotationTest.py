from image_utils import angleCal, creat_images


result_orig, result_rotate, result_rotate_translation,\
result_perspective, result_correction,\
result_rotation_per, result_rotation_per_cor= creat_images()

print("Simple Rotation:")
mean_rotate = angleCal(result_orig, result_rotate)
print("Mean between rotate is {0:f}".format(mean_rotate))
print("Simple Rotation and Translation:")
mean_rotate_translation = angleCal(result_orig, result_rotate_translation)
print("Mean between rotate and translation is {0:f}".format(mean_rotate_translation))

print("Simple Perspective:")
mean_perspective = angleCal(result_orig, result_perspective)
print("Mean between perspective is {0:f}".format(mean_perspective))
print("Correction From Perspective:")
mean_correction = angleCal(result_orig, result_correction)
print("Mean between perspective is {0:f}".format(mean_correction))

print("Rotation Perspective:")
mean_rot_perspective = angleCal(result_orig, result_rotation_per)
print("Mean between perspective is {0:f}".format(mean_rot_perspective))
print("Rotation Correction From Perspective:")
mean_rot_correction = angleCal(result_orig, result_rotation_per_cor)
print("Mean between perspective is {0:f}".format(mean_rot_correction))