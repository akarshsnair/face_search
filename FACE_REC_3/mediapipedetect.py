import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import cv2
import numpy as np
import dlib
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp

COSINE_THRESHOLD = 0.5

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
face_mesh = mp_face_mesh.FaceMesh()


def recognize_face(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        return []
    
    h, w, _ = image.shape
    bounding_box = results.detections[0].location_data.relative_bounding_box
    x, y, width, height = int(bounding_box.xmin * w), int(bounding_box.ymin * h), int(bounding_box.width * w), int(bounding_box.height * h)
    
    return [dlib.rectangle(left=x, top=y, right=x+width, bottom=y+height)] 

def extract_embeddings(face_recognizer, image, face_rectangle):
    shape_predictor = dlib.shape_predictor("model/shape_data")
    landmarks = shape_predictor(image, face_rectangle)
    embedding = face_recognizer.compute_face_descriptor(image, landmarks)
    return np.array(embedding)

'''

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
    '''

def get_landmarks(image):

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return []

    landmarks = results.multi_face_landmarks[0].landmark
    return [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks]

def align_face(image):

    landmarks = get_landmarks(image)
    if not landmarks:
        return None

    x_coords = [point[0] for point in landmarks]
    y_coords = [point[1] for point in landmarks]
    x, y, w, h = min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)

    cropped_face = image[y:y+h, x:x+w]
    return cropped_face


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
    query_image_path = 'eval/mamm.jpg'

    face_detector = dlib.get_frontal_face_detector()
    face_recognizer = dlib.face_recognition_model_v1('model/data')
    
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image)

    if not query_faces:
        #print("No faces found in the query image.")
        return
    
    query_embedding = extract_embeddings(face_recognizer, query_image, query_faces[0])  

    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")

            if os.path.exists(embedding_path):
                #print(f"Embedding for {filename} already exists. Skipping.")
                continue

            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)
            faces = recognize_face(image)

            if not faces:
                #print(f"No faces found in {filename}. Skipping.")
                continue

            embedding = extract_embeddings(face_recognizer, image, faces[0])  
            np.save(embedding_path, embedding)
            print(f"Embedding saved for {filename}")


    embeddings = load_embeddings(embeddings_dir)
    similarities = match_faces(embeddings, query_embedding)

    sorted_similarities = sorted(
        similarities.items(), key=lambda x: x[1], reverse=True)
    

    #aligned_face = align_face(query_image)
    #query_embedding = extract_embeddings(face_recognizer, aligned_face)

    # Match query embedding with dataset embeddings
    

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
