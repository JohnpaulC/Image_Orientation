from image_utils import angleCal, create_images
import matplotlib.pyplot as plt

rotate_angle = 10

result_orig, result_rotate, result_rotate_translation,\
result_perspective, result_correction,\
result_rotation_per, result_rotation_per_cor= create_images(rotate_angle)

name_result = []
result = []

print("Simple Rotation:")
mean_rotate = angleCal(result_orig, result_rotate)
result.append(mean_rotate)
name_result.append("Rotate")
print("Mean between rotate is {0:f}".format(mean_rotate))
print("Simple Rotation and Translation:")
mean_rotate_translation = angleCal(result_orig, result_rotate_translation)
result.append(mean_rotate_translation)
name_result.append("Trans+Rotate")
print("Mean between rotate and translation is {0:f}".format(mean_rotate_translation))

print("Simple Perspective:")
mean_perspective = angleCal(result_orig, result_perspective)
result.append(mean_perspective)
name_result.append("Perspective")
print("Mean between perspective is {0:f}".format(mean_perspective))
print("Correction From Perspective:")
mean_correction = angleCal(result_orig, result_correction)
result.append(mean_correction)
name_result.append("Correction")
print("Mean between perspective is {0:f}".format(mean_correction))

print("Rotation Perspective:")
mean_rot_perspective = angleCal(result_orig, result_rotation_per)
result.append(mean_rot_perspective)
name_result.append("RPerspective")
print("Mean between perspective is {0:f}".format(mean_rot_perspective))
print("Rotation Correction From Perspective:")
mean_rot_correction = angleCal(result_orig, result_rotation_per_cor)
result.append(mean_rot_correction)
name_result.append("RCorrection")
print("Mean between perspective is {0:f}".format(mean_rot_correction))

plt.figure()
plt.bar(name_result, result)
plt.show()


