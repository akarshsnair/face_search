import os
import cv2
import numpy as np
import dlib
import nmslib
import time

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

def build_hnsw_index(embeddings):
    embedding_dim = len(next(iter(embeddings.values())))
    hnsw_index = nmslib.init(method='hnsw', space='cosinesimil')
    
    for i, embedding in enumerate(embeddings.values()):
        hnsw_index.addDataPoint(i, embedding)
    
    hnsw_index.createIndex({'post': 2}, print_progress=True)
    return hnsw_index

def find_most_similar_hnsw(hnsw_index, query_embedding, num_neighbors=5):
    similar_indices, _ = hnsw_index.knnQuery(query_embedding, k=num_neighbors)
    return similar_indices

def main():
    start = time.time()
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/Zoran_Djindjic_0004.jpg'
    
    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')

    embeddings = load_embeddings(embeddings_dir)
    hnsw_index_path = 'data/hnsw_index.bin'

    if os.path.exists(hnsw_index_path):
        hnsw_index = nmslib.init(method='hnsw', space='cosinesimil')
        hnsw_index.loadIndex(hnsw_index_path)
    else:
        hnsw_index = build_hnsw_index(embeddings)
        hnsw_index.saveIndex(hnsw_index_path)

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detector)

    if not query_faces:
        print("No faces found in the query image.")
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_recognizer, aligned_face)

    similar_indices = find_most_similar_hnsw(hnsw_index, query_embedding)

    if len(similar_indices) == 0:
        print("No similar image found.")
        return

    most_similar_user_ids = list(embeddings.keys())[similar_indices[0]]
    print("Most similar user:", most_similar_user_ids)

    most_similar_image_path = os.path.join(dataset_dir, most_similar_user_ids + '.jpg')
    most_similar_image = cv2.imread(most_similar_image_path)

    cv2.imshow("Most Similar Image", most_similar_image)
    end = time.time()
    print("The time of execution of the program is:", (end - start) * 10**3, "ms")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
