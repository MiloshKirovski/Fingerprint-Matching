import cv2
import numpy as np
import os


def get_descriptor_method(descriptor_method):
    if descriptor_method == 'SIFT':
        return cv2.SIFT_create()
    elif descriptor_method == 'AKAZE':
        return cv2.AKAZE_create()
    elif descriptor_method == 'BRISK':
        return cv2.BRISK_create()
    else:
        raise ValueError(f"Unsupported method: {method}")


def process_and_save(img_path, output_dir, descriptor_method):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    descriptor = get_descriptor_method(descriptor_method)

    kps, descriptors = descriptor.detectAndCompute(gray, None)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    output_desc_path = os.path.join(output_dir, f'{base_name}_{descriptor_method}_descriptors.npy')
    np.save(output_desc_path, descriptors)

    kps_array = np.array([kp.pt + (kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps])
    output_kps_path = os.path.join(output_dir, f'{base_name}_{descriptor_method}_keypoints.npy')
    np.save(output_kps_path, kps_array)


database_directory = 'SIMPLE/REAL'
output_directory = 'descriptors_output'
method = 'BRISK'

for filename in os.listdir(database_directory):
    if filename.endswith('.BMP'):
        image_path = os.path.join(database_directory, filename)
        process_and_save(image_path, output_directory, method)
