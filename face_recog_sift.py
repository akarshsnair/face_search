import cv2
import numpy as np
import os

def compute_sift_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors
query_image_path = "data/val/elton_john/httpcdncdnjustjaredcomwpcontentuploadsheadlineseltonjohnsupportsbrucejennerstransitiontowomanjpg.jpg"
query_keypoints, query_descriptors = compute_sift_features(query_image_path)


dataset_directory = "data/faces"
flann_index_kdtree = 0
index_params = dict(algorithm=flann_index_kdtree, trees=5)
search_params = dict(checks=50)  # Higher checks -> slower but more accurate

flann = cv2.FlannBasedMatcher(index_params, search_params)

best_match = None
lowest_distance = float('inf')

for filename in os.listdir(dataset_directory):
    if filename.endswith(('.jpg', '.png', '.jpeg')):  # or whatever file types you have
        image_path = os.path.join(dataset_directory, filename)
        keypoints, descriptors = compute_sift_features(image_path)
        
        matches = flann.knnMatch(query_descriptors, descriptors, k=2)
        
        # Store all the good matches as per Lowe's ratio test.
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        
        if good_matches:
            distance = sum([m.distance for m in good_matches])
            if distance < lowest_distance:
                lowest_distance = distance
                best_match = image_path

print("Best Match Found:", best_match)
if best_match:
    query_image = cv2.imread(query_image_path)
    matching_image = cv2.imread(best_match)
    
    cv2.imshow('Query Image', query_image)
    cv2.imshow('Best Match', matching_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
