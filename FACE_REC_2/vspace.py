import time

start_time = time.time()
import dlib
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import cv2

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("model/shape_predictor_5_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("model/dlib_face_recognition_resnet_model_v1.dat")

#stored_embeddings = []

def get_all_face_embeddings(image_directory):
    embeddings = []
    for image_file in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image_file)  
        embedding = get_face_embedding(image_path)
        
        if embedding is not None:
            embeddings.append(embedding)
            print(f"Processed {image_file}")
        else:
            print(f"No face detected in {image_file}")
    
    return embeddings

def get_face_embedding(image_path):
    #img = dlib.load_rgb_image(image_path)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = face_detector(img)
    if len(faces) == 0:
        print("No face detected in", image_path)
        return None

    shape = shape_predictor(img, faces[0])
    face_embedding = face_recognition_model.compute_face_descriptor(img, shape)

    return np.array(face_embedding)

def find_closest_face(query_embedding):
    distances, indices = knn.kneighbors([query_embedding])
    return indices[0][0], distances[0][0]

def get_image_path_from_directory(directory, index):
    return os.path.join(directory, os.listdir(directory)[index])

#stored_embeddings = get_all_face_embeddings('dataset')

#np.save("embeddings.npy", stored_embeddings)
stored_embeddings = np.load("embeddings.npy").tolist()

X = np.array(stored_embeddings)

knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X) 

query_image_path = 'eval/Manisha Koirala_3.jpg'
query_image = cv2.imread(query_image_path)
query_embedding = get_face_embedding(query_image_path)
cv2.imshow("Query Image", query_image)

if query_embedding is not None:
    index, dist = find_closest_face(query_embedding)
    print(f"Found similar face at index {index} with distance {dist}")
else:
    print("No face detected in the query image.")

image_path = get_image_path_from_directory("dataset", index)
img = cv2.imread(image_path)

ans = np.load('embeddings.npy')
#print(ans[index])

cv2.imshow("Matched Face", img)

end_time = time.time() 
print(f"Algorithm vectospace knn took {end_time - start_time:.4f} seconds to execute")

cv2.waitKey(0)
cv2.destroyAllWindows()