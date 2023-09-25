import os
import cv2
import numpy as np
import dlib
from sklearn.metrics.pairwise import cosine_similarity
import time


COSINE_THRESHOLD = 0.5

def extract_embeddings(face_recognizer, aligned_face):
    embedding = face_recognizer.compute_face_descriptor(aligned_face)
    #print("hai")
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

def match_faces(embeddings, query_embedding):
    similarities = {}
    for user_id, reference_embedding in embeddings.items():
        similarity = cosine_similarity(
            [query_embedding], [reference_embedding])[0][0]
        similarities[user_id] = similarity
    return similarities

def main():
    start=time.time()

    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/Tony_Parker_0001.jpg'

    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')

    embeddings = load_embeddings(embeddings_dir)
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            if user_id in embeddings:
                #print(f"Embedding already exists for {filename}. Skipping.")
                continue
            
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image, face_detector)

            if not faces:
                #print(f"No faces found in {filename}. Skipping.")
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
    start = time.time()
    similarities = match_faces(embeddings, query_embedding)

    sorted_similarities = sorted(
        similarities.items(), key=lambda x: x[1], reverse=True)
    
    # cv2.imshow("Query Image", query_image)
    #print("hai")
    for user_id, similarity in sorted_similarities:
        if similarity >= COSINE_THRESHOLD:
            print(f"User ID: {user_id}, Similarity: {similarity:.4f}")

            similar_image_path = os.path.join(dataset_dir, user_id + '.jpg')
            similar_image = cv2.imread(similar_image_path)

            print(f"Similar Image Filename: {user_id}.jpg")

            cv2.imshow("Similar Image", similar_image)
            
            #cv2.resize(similar_image, (100, 100))
        end = time.time()
        print("The time of execution of above program is :",(end-start) * 10**3, "ms")    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
if __name__ == '__main__':
    main()