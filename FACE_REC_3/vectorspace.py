import os
import cv2
import numpy as np
import dlib
from sklearn.neighbors import NearestNeighbors
import time
COSINE_THRESHOLD = 0.5
def extract_embeddings(face_recognizer, aligned_face):
    embedding = face_recognizer.compute_face_descriptor(aligned_face)
    return np.array(embedding)

def recognize_face(image, face_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    return faces

def align_face(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape_predictor = dlib.shape_predictor("model/shape_data")
    landmarks = shape_predictor(gray, face)

    aligned_face = dlib.get_face_chip(image, landmarks)
    return aligned_face

def load_embeddings(embeddings_dir):
    embeddings = []
    user_ids = []
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            user_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(embeddings_dir, filename))
            embeddings.append(embedding)
            user_ids.append(user_id)
    return np.array(embeddings), user_ids

def main():
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings/new'
    query_image_path = 'eval/002_86e8aa58.jpg'

    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Process the dataset images and extract embeddings
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image, face_detector)

            if not faces:
                print(f"No faces found in {filename}. Skipping.")
                continue

            aligned_face = align_face(image, faces[0])
            embedding = extract_embeddings(face_recognizer, aligned_face)

            user_id = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
            np.save(embedding_path, embedding)

            print(f"Embedding saved for {filename}")

    embeddings, user_ids = load_embeddings(embeddings_dir)
    knn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(embeddings)

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detector)

    if not query_faces:
        print("No faces found in the query image.")
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_recognizer, aligned_face)

    distances, indices = knn.kneighbors([query_embedding])
    matched_user_id = user_ids[indices[0][0]]
    distance = distances[0][0]

    cv2.imshow("Query Image", query_image)
    print(f"Closest User ID: {matched_user_id}, Distance: {distance:.4f}")

    similar_image_path = os.path.join(dataset_dir, matched_user_id + '.jpg')
    similar_image = cv2.imread(similar_image_path)
    cv2.imshow("Similar Image", similar_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start=time.time()
    main()
    print(start-time.time())
