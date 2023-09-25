import os
import cv2
import numpy as np
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from facenet import preprocessing

COSINE_THRESHOLD = 0.5

def extract_embeddings(face_net, aligned_face):
    # Preprocess the image for FaceNet
    aligned_face = preprocessing.prewhiten(aligned_face)
    aligned_face = cv2.resize(aligned_face, (160, 160))
    
    # Expand dimensions to match FaceNet model input shape
    aligned_face = np.expand_dims(aligned_face, axis=0)
    
    # Generate embeddings
    embedding = face_net.predict(aligned_face)
    
    return embedding

def recognize_face(image, mtcnn_detector):
    faces = mtcnn_detector.detect_faces(image)
    return faces

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
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/sidhiq3.jpeg'

    mtcnn_detector = MTCNN()
    face_net = load_model('path_to_facenet_model')  # Replace with the path to your FaceNet model
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Process the dataset images and extract embeddings
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image, mtcnn_detector)

            if not faces:
                print(f"No faces found in {filename}. Skipping.")
                continue

            x, y, w, h = faces[0]['box']
            face = image[y:y+h, x:x+w]
            aligned_face = cv2.resize(face, (160, 160))  # Resize for FaceNet

            embedding = extract_embeddings(face_net, aligned_face)

            user_id = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
            np.save(embedding_path, embedding)

            print(f"Embedding saved for {filename}")

    # Load and preprocess the query image
    query_image = cv2.imread(query_image_path)
    faces = recognize_face(query_image, mtcnn_detector)

    if not faces:
        print("No faces found in the query image.")
        return

    x, y, w, h = faces[0]['box']
    query_face = query_image[y:y+h, x:x+w]
    aligned_face = cv2.resize(query_face, (160, 160))  # Resize for FaceNet
    query_embedding = extract_embeddings(face_net, aligned_face)

    # Match query embedding with dataset embeddings
    embeddings = load_embeddings(embeddings_dir)
    similarities = match_faces(embeddings, query_embedding)

    sorted_similarities = sorted(
        similarities.items(), key=lambda x: x[1], reverse=True)

    cv2.imshow("Query Image", query_image)

    print("Similar images:")
    for user_id, similarity in sorted_similarities:
        if similarity >= COSINE_THRESHOLD:
            print(f"User ID: {user_id}, Similarity: {similarity:.4f}")

            similar_image_path = os.path.join(dataset_dir, user_id + '.jpg')
            similar_image = cv2.imread(similar_image_path)
            print(f"Similar Image Filename: {user_id}.jpg")
            cv2.imshow("Similar Image", similar_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    main()
