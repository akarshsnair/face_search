import time

start_time = time.time()
import os
import cv2
import numpy as np
import dlib
from sklearn.neighbors import KDTree

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
    embeddings = {}
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            user_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(embeddings_dir, filename))
            embeddings[user_id] = embedding
    return embeddings

def build_kdtree(embeddings):
    embeddings_matrix = np.array(list(embeddings.values()))
    tree = KDTree(embeddings_matrix)
    
    return tree

def match_faces_with_kdtree(tree, query_embedding, user_ids, k=5):
    k_value = min(5, tree.data.shape[0])
    distance, indices = tree.query([query_embedding], k=k_value)
    closest_users = [(user_ids[indices[0][i]], 1 - distance[0][i]) for i in range(len(indices[0]))]
    return closest_users

def main():
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/deepika-padukone-cannes-2022-main_7_202209.jpg'
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')

    embeddings = load_embeddings(embeddings_dir)
    user_ids = list(embeddings.keys())
    #print(f"Loaded {len(embeddings)} embeddings from {embeddings_dir}.")
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            if user_id in embeddings:
                continue
            
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image, face_detector)
            if not faces:
                continue

            aligned_face = align_face(image, faces[0])
            embedding = extract_embeddings(face_recognizer, aligned_face)
            embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
            np.save(embedding_path, embedding)
            print(f"Embedding saved for {filename}")

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detector)
    if not query_faces:
        print("No faces found in the query image.")
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_recognizer, aligned_face)

    # Building and querying KD-Tree
    tree = build_kdtree(embeddings)
    #print(f"Number of user_ids: {len(user_ids)}")
    #print(f"Number of data points in KD-Tree: {tree.data.shape[0]}")

    closest_users = match_faces_with_kdtree(tree, query_embedding, user_ids)

    cv2.imshow("Query Image", query_image)
    print("Similar images:")
    for user_id, similarity in closest_users:
        if similarity >= COSINE_THRESHOLD:
            print(f"User ID: {user_id}, Similarity: {similarity:.4f}")
            similar_image_path = os.path.join(dataset_dir, user_id + '.jpg')
            similar_image = cv2.imread(similar_image_path)
            cv2.imshow("Similar Image", similar_image)
            end_time = time.time()
            print(f"Algorithm kdtree took {end_time - start_time:.4f} seconds to execute")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
