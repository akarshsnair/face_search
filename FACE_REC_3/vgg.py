import os
import cv2
import numpy as np
import mediapipe as mp
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import faiss

THRESHOLD = 0.5

def extract_embeddings(face_mesh, image, face):
    # Extract face landmarks using MediaPipe Face Mesh
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks]

        # Draw landmarks on the face (optional)
        for landmark in landmarks:
            cv2.circle(image, landmark, 2, (0, 255, 0), -1)

        # Extract embeddings (you may need to customize this part)
        embedding = np.array([landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks])

        return embedding
    else:
        return None
    
def align_face(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape_predictor = dlib.shape_predictor("model/shape_data")
    landmarks = shape_predictor(gray, face)

    aligned_face = dlib.get_face_chip(image, landmarks)
    return aligned_face

def recognize_face(image, face_detector):
    # Use MediaPipe Face Detection for face detection
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append((x, y, x+w, y+h))
    return faces

def load_embeddings(embeddings_dir):
    embeddings = {}
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npy'):
            user_id = os.path.splitext(filename)[0]
            embedding = np.load(os.path.join(embeddings_dir, filename))
            embeddings[user_id] = embedding
    return embeddings

def main():
    dataset_dir = 'dataset'
    embeddings_dir = 'data/embeddings'
    query_image_path = 'eval/1137877401.jpg'

    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize MediaPipe Face Detection and Face Mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
    face_mesh = mp_face_mesh.FaceMesh()

    embeddings = load_embeddings(embeddings_dir)

    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Load the VGGFace model
    vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    # Create an index for Faiss
    d = 2048  # Dimension of the embeddings from VGGFace
    index = faiss.IndexFlatIP(d)  # Use Inner product (dot product) as the similarity measure

    # Add reference embeddings to Faiss index
    for filename in os.listdir(dataset_dir):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            user_id = os.path.splitext(filename)[0]
            if user_id in embeddings:
                continue

            image_path = os.path.join(dataset_dir, filename)
            image = cv2.imread(image_path)

            faces = recognize_face(image, face_detection)

            if not faces:
                continue

            aligned_face = align_face(image, faces[0])
            
            # Preprocess the image for VGGFace
            aligned_face = cv2.resize(aligned_face, (224, 224))
            aligned_face = np.expand_dims(aligned_face, axis=0)
            aligned_face = preprocess_input(aligned_face, version=2)

            # Extract embeddings using VGGFace
            embedding = vggface.predict(aligned_face)

            if embedding is not None:
                # Add the embedding to Faiss index
                index.add(embedding)

                embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
                np.save(embedding_path, embedding)

                print(f"Embedding saved for {filename}")

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detection)

    if not query_faces:
        return

    aligned_face = align_face(query_image, query_faces[0])

    # Preprocess the query image for VGGFace
    aligned_face = cv2.resize(aligned_face, (224, 224))
    aligned_face = np.expand_dims(aligned_face, axis=0)
    aligned_face = preprocess_input(aligned_face, version=2)

    # Extract query embeddings using VGGFace
    query_embedding = vggface.predict(aligned_face)

    # Perform similarity search using Faiss
    query_embedding = np.array([query_embedding])
    D, I = index.search(query_embedding, len(embeddings))

    print("Similar images:")
    for i, idx in enumerate(I[0]):
        similarity = D[0][i]  # Faiss Inner product (dot product) similarity
        if similarity >= THRESHOLD:
            user_id = os.path.splitext(os.listdir(dataset_dir)[idx])[0]
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
