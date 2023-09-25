import cv2
import numpy as np
import os

# Load the input image
input_image = cv2.imread('FACE_REC/2023-08-21-104257.jpg', cv2.IMREAD_GRAYSCALE)

# Get a list of dataset image paths from the 'dataset' folder
dataset_folder = 'FACE_REC/dataset'
dataset_image_paths = [os.path.join(dataset_folder, filename) for filename in os.listdir(dataset_folder)]

# Initialize ORB detector and descriptor
orb = cv2.ORB_create()

# Find the keypoints and descriptors for the input image
keypoints_input, descriptors_input = orb.detectAndCompute(input_image, None)

# Create FLANN index with ORB descriptors
index_params = dict(algorithm=6,    # FLANN_INDEX_LSH
                    table_number=6,  # 6 hash tables
                    key_size=12,     # 12 bits for key
                    multi_probe_level=1)  # Multi-probe level 1

search_params = dict(checks=50)  # Number of checks for each query

flann = cv2.FlannBasedMatcher(index_params, search_params)

best_match_image = None
best_match_distance = float('inf')

# Iterate through the dataset images
for dataset_image_path in dataset_image_paths:
    dataset_image = cv2.imread(dataset_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Find the keypoints and descriptors for the dataset image
    keypoints_dataset, descriptors_dataset = orb.detectAndCompute(dataset_image, None)
    
    # Match descriptors using FLANN
    matches = flann.knnMatch(descriptors_input, descriptors_dataset, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for match in matches:
        if len(match) == 2 and match[0].distance < 0.75 * match[1].distance:
            good_matches.append(match[0])
    
    # Calculate the distance of good matches
    if len(good_matches) > 0:
        distance = sum([match.distance for match in good_matches]) / len(good_matches)
    else:
        distance = float('inf')
    
    # Update best match if the current image is more similar
    if distance < best_match_distance:
        best_match_distance = distance
        best_match_image = dataset_image

# Display the best match image
cv2.imshow('Best Match Image', best_match_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
