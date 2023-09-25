import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

COSINE_THRESHOLD = 0.5

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

def recognize_face(image, face_detector):
    # Use MediaPipe Face Detection for face detection
    results = face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append(dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h))
    return faces

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
            embedding = extract_embeddings(face_mesh, aligned_face, faces[0])

            if embedding is not None:
                embedding_path = os.path.join(embeddings_dir, f"{user_id}.npy")
                np.save(embedding_path, embedding)

                print(f"Embedding saved for {filename}")

    query_image = cv2.imread(query_image_path)
    query_faces = recognize_face(query_image, face_detection)

    if not query_faces:
        return

    aligned_face = align_face(query_image, query_faces[0])
    query_embedding = extract_embeddings(face_mesh, aligned_face, query_faces[0])

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
