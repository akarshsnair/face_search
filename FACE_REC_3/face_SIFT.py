import cv2
import numpy as np
import os


input_image = cv2.imread('eval/2023-08-21-104257.jpg', cv2.IMREAD_GRAYSCALE)

dataset_folder = 'dataset'
dataset_image_paths = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder)]

sift = cv2.SIFT_create()
keypoints_input, descriptors_input = sift.detectAndCompute(input_image, None)

bf = cv2.BFMatcher()

best_match_image = None
best_match_distance = float('inf')

for dataset_image_path in dataset_image_paths:
    dataset_image = cv2.imread(dataset_image_path, cv2.IMREAD_GRAYSCALE)
    
    keypoints_dataset, descriptors_dataset = sift.detectAndCompute(dataset_image, None)
    
    matches = bf.knnMatch(descriptors_input, descriptors_dataset, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 0:
        distance = sum([match.distance for match in good_matches]) / len(good_matches)
    else:
        distance = float('inf')

    if distance < best_match_distance:
        best_match_distance = distance
        best_match_image = dataset_image

cv2.imshow('Best Match Image', best_match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
