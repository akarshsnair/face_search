import os
import cv2
import numpy as np
import dlib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
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
    embeddings = {}
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            user_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(embeddings_dir, filename))
            embeddings[user_id] = embedding
    return embeddings

def hierarchical_clustering(embeddings):
    reference_embeddings = np.array(list(embeddings.values()))
    clustering = AgglomerativeClustering(n_clusters=None, linkage='ward', distance_threshold=0.5)
    cluster_labels = clustering.fit_predict(reference_embeddings)
    cluster_mapping = {}
    for user_id, label in zip(embeddings.keys(), cluster_labels):
        if label not in cluster_mapping:
            cluster_mapping[label] = [user_id]
        else:
            cluster_mapping[label].append(user_id)
    return cluster_mapping

def match_faces_in_cluster(cluster_embeddings, query_embedding):
    similarities = {}
    for user_id, reference_embedding in cluster_embeddings.items():
        similarity = cosine_similarity([query_embedding], [reference_embedding])[0][0]
        similarities[user_id] = similarity
    return similarities

def main():
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/Tony_Parker_0001.jpg'

    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')

    embeddings = load_embeddings(embeddings_dir)
    cluster_mapping_path = 'data/cluster_mapping.npy'
    if os.path.exists(cluster_mapping_path):
        cluster_mapping = np.load(cluster_mapping_path, allow_pickle=True).item()
    else:
        cluster_mapping = hierarchical_clustering(embeddings)
        np.save(cluster_mapping_path, cluster_mapping)

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detector)

    if not query_faces:
        print("No faces found in the query image.")
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_recognizer, aligned_face)
    start = time.time()

    cluster_similarities = {}
    for cluster_id, cluster_user_ids in cluster_mapping.items():
        cluster_embeddings = {user_id: embeddings[user_id] for user_id in cluster_user_ids if user_id in embeddings}
        if cluster_embeddings:
            similarities = match_faces_in_cluster(cluster_embeddings, query_embedding)
            cluster_similarity = sum(similarities.values()) / len(similarities)
            cluster_similarities[cluster_id] = cluster_similarity

    most_similar_cluster = max(cluster_similarities, key=cluster_similarities.get)
    most_similar_user_ids = cluster_mapping[most_similar_cluster]

    print("Most similar cluster:", most_similar_cluster)
    print("User IDs in the cluster:", most_similar_user_ids)

    cluster_embeddings = {user_id: embeddings[user_id] for user_id in most_similar_user_ids}
    similarities = match_faces_in_cluster(cluster_embeddings, query_embedding)
    most_similar_user = max(similarities, key=similarities.get)
    print("Most similar user:", most_similar_user)

    most_similar_image_path = os.path.join(dataset_dir, most_similar_user + '.jpg')
    most_similar_image = cv2.imread(most_similar_image_path)

    cv2.imshow("Most Similar Image", most_similar_image)
    end = time.time()
    print("The time of execution of the program is:", (end - start) * 10**3, "ms")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()