import os
import cv2
from PIL import Image
from dotenv import load_dotenv
from numpy import asarray, expand_dims
from joblib import load
from keras.models import load_model
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer

from src.features.face_detection import extract_face
from src.features.face_embedding import create_embeddings
from src.ui.employee_recognition import unrecognized_face, recognized_face, undetected_face

load_dotenv()

face_detector = MTCNN()
embedding_model = load_model('../models/facenet_keras.h5')
classification_model = load("../models/classification-model.joblib")
in_encoder = Normalizer(norm='l2')
out_encoder = load("../data/intermediate/encoder.joblib")

print("Loading complete")

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('s'):
        image = Image.fromarray(frame, 'RGB')
        pixels = asarray(image)
        cv2.imwrite("../data/raw/emp.jpg", pixels)

        faces = list()

        extracted_face = extract_face("../data/raw/emp.jpg", face_detector)

        if extracted_face is None:
            undetected_face()
            continue

        faces.append(extracted_face)
        faces = asarray(faces)

        print(faces.shape)

        embedding_face = create_embeddings(embedding_model, faces)

        print(embedding_face.shape)

        embedding_face = in_encoder.transform(embedding_face)
        embedding_face = expand_dims(embedding_face[0], axis=0)

        result_class = classification_model.predict(embedding_face)
        result_prob = classification_model.predict_proba(embedding_face)

        result_class_name = out_encoder.inverse_transform(result_class)

        print(result_class[0], result_prob, result_class_name[0])

        if result_prob[0, result_class[0]] < 0.65:
            unrecognized_face()

        else:
            recognized_face(result_class_name[0])

        os.remove("../data/raw/emp.jpg")

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
