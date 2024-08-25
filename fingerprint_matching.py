import cv2
import numpy as np
import os
import argparse
import keras


def convert_to_cv2_key_points(keypoint_array):
    key_points = []
    for kp in keypoint_array:
        keypoint = cv2.KeyPoint(kp[0], kp[1], kp[2], kp[3], kp[4], int(kp[5]), int(kp[6]))
        key_points.append(keypoint)
    return tuple(key_points)


def load_data(data_dir, descriptor_method):
    fingerprint_data = []
    for file in os.listdir(data_dir):
        if file.endswith(f'_{descriptor_method}_descriptors.npy'):
            descriptor_path = os.path.join(data_dir, file)
            key_points_path = descriptor_path.replace(f'_{descriptor_method}_descriptors.npy',
                                                      f'_{descriptor_method}_keypoints.npy')
            descriptors = np.load(descriptor_path)
            key_points = np.load(key_points_path)

            if descriptor_method in ['AKAZE', 'BRISK']:
                descriptors = descriptors.astype(np.uint8)
            else:
                descriptors = descriptors.astype(np.float32)

            fingerprint_data.append((key_points, descriptors, file))
    return fingerprint_data


def get_descriptor_method(method_name):
    if method_name == 'SIFT':
        return cv2.SIFT_create()
    elif method_name == 'AKAZE':
        return cv2.AKAZE_create()
    elif method_name == 'BRISK':
        return cv2.BRISK_create()
    else:
        raise ValueError(f'Unsupported method: {method_name}')


def match_with_descriptors(query_img_path, fingerprint_data, database_dir='SIMPLE/READ', descriptor_method='SIFT'):
    query_image = cv2.imread(query_img_path)
    gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

    descriptor = get_descriptor_method(descriptor_method)
    query_key_points, query_descriptors = descriptor.detectAndCompute(gray, None)

    if descriptor_method in ['AKAZE', 'BRISK']:
        query_descriptors = query_descriptors.astype(np.uint8)
    else:
        query_descriptors = query_descriptors.astype(np.float32)

    bf = cv2.BFMatcher()
    highest_score = 0
    best_match_file = best_match_key_points = None
    best_good_points = []

    for db_key_points, db_descriptors, db_filename in fingerprint_data:
        matches = bf.knnMatch(query_descriptors, db_descriptors, k=2)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
            elif len(m_n) == 1:
                good_matches.append([m_n[0]])

        match_score = len(good_matches) / len(query_key_points) * 100
        print(f"Match score for {db_filename}: {match_score}")

        if match_score > highest_score:
            highest_score = match_score
            best_match_file = db_filename
            best_good_points = good_matches
            best_match_key_points = db_key_points

    if best_match_file is None:
        print("No matching image found.")
        return

    best_match_image_path = os.path.join(database_dir,
                                         best_match_file.replace(f'_{descriptor_method}_descriptors.npy', '.BMP'))
    best_match_image = cv2.imread(best_match_image_path)

    if best_match_image is None:
        print(f"Failed to read image from path: {best_match_image_path}")
        return

    best_match_key_points = convert_to_cv2_key_points(best_match_key_points)

    img_matches = np.empty((max(query_image.shape[0], best_match_image.shape[0]),
                            query_image.shape[1] + best_match_image.shape[1], 3), dtype=np.uint8)
    img_original = np.empty((max(query_image.shape[0], best_match_image.shape[0]),
                            query_image.shape[1] + best_match_image.shape[1], 3), dtype=np.uint8)
    cv2.drawMatchesKnn(query_image, query_key_points, best_match_image, best_match_key_points,
                       best_good_points, img_matches, flags=2)

    img_original[:query_image.shape[0], :query_image.shape[1]] = query_image
    img_original[:best_match_image.shape[0], query_image.shape[1]:] = best_match_image

    img_matches = cv2.resize(img_matches, None, fx=4, fy=4)
    img_original = cv2.resize(img_original, None, fx=4, fy=4)

    print(f'Best match: {best_match_image_path} with score: {highest_score}')    # cv2.imshow('Best Match', img_matches)
    cv2.imshow('Best Match', img_matches)
    cv2.imshow('Original', img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = np.stack([img] * 3, axis=-1)  # Converts grayscale to RGB
    img = img.astype(np.float32) / 255.0  # Normalize
    return np.expand_dims(img, axis=0)  # Adds batch dimension


def compare_fingerprints(img1_path, img2_path, model):
    img1 = preprocess_image(img1_path)
    img2 = preprocess_image(img2_path)

    prediction1 = model.predict(img1)
    prediction2 = model.predict(img2)

    similarity = np.dot(prediction1, prediction2.T) / (np.linalg.norm(prediction1) * np.linalg.norm(prediction2))
    return similarity[0][0]


def match_with_model(query_img_path, model_path, database_dir):
    model = keras.models.load_model(model_path)
    print("Model loaded!")

    best_match_file = None
    best_score = float('-inf')

    for root, _, files in os.walk(database_dir):
        for file in files:
            if file.endswith('.BMP'):
                db_img_path = os.path.join(root, file)

                score = compare_fingerprints(query_img_path, db_img_path, model)
                print(f'Similarity score for {file}: {score}')

                if score > best_score:
                    best_score = score
                    best_match_file = file

    if best_match_file is None:
        print('No matching image was found!')
    else:
        print(f'Best match: {best_match_file} with score: {best_score}')
        best_match_image_path = os.path.join(database_dir, best_match_file)
        best_match_image = cv2.imread(best_match_image_path)

        query_image_display = cv2.imread(query_img_path)
        img_matches = np.hstack((query_image_display, best_match_image))
        img_matches = cv2.resize(img_matches, None, fx=4, fy=4)

        cv2.imshow('Best Match', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Fingerprint Matching')
    parser.add_argument('query_image', type=str, help='Path to the query image')
    parser.add_argument('method', type=str, choices=['SIFT', 'AKAZE', 'BRISK', 'FEATURE_MODEL'],
                        help='Descriptor method or SIAMESE_MODEL a for neural network')
    parser.add_argument('--model', type=str, default='fingerprint_recognition_model_10.h5',
                        help='Path to the trained model (required if method is FEATURE_MODEL)')
    args = parser.parse_args()

    database_dir = 'SIMPLE/REAL'
    output_dir = 'descriptors_output'

    if args.method == 'FEATURE_MODEL':
        if not args.model:
            raise ValueError('Model path must be specified if method is SIAMESE_MODEL')
        match_with_model(args.query_image, args.model, database_dir)
    else:
        data = load_data(output_dir, args.method)
        match_with_descriptors(args.query_image, data, database_dir, args.method)


if __name__ == '__main__':
    main()

# Example usage for descriptors:
# python fingerprint_matching.py SIMPLE/ALTERED/1__M_Right_index_finger_CR.BMP SIFT
# Example usage for model:
# python fingerprint_matching.py SIMPLE/ALTERED/1__M_Right_index_finger_Obl.BMP FEATURE_MODEL --model fingerprint_recognition_model_10.h5
