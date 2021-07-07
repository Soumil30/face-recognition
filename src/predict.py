import os
import cv2
from PIL import Image
from dotenv import load_dotenv
from numpy import asarray, expand_dims
from joblib import load
from keras.models import load_model
from mtcnn import MTCNN
from sklearn.preprocessing import Normalizer

from src.face_detection import extract_face
from src.face_embedding import create_embeddings
from src.tk_face import unrecognized_face, recognized_face, undetected_face

load_dotenv()

face_detector = MTCNN()
embedding_model = load_model('../models/facenet_keras.h5')
classification_model = load("../models/celebrity-classification-model.joblib")
in_encoder = Normalizer(norm='l2')
out_encoder = load("../models/encoder.joblib")

print("Loading complete")

vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        image = Image.fromarray(frame, 'RGB')
        pixels = asarray(image)
        cv2.imwrite("../data/emp.jpg", pixels)

        faces = list()

        extracted_face = extract_face("../data/emp.jpg", face_detector)

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

        result_class = classification_model.predict(embedding_face)[0]
        result_prob = classification_model.predict_proba(embedding_face)

        result_class_name = out_encoder.inverse_transform(result_class)[0]

        print(result_class, result_prob, result_class_name)

        if result_prob[0, result_class] < 0.6:
            unrecognized_face()

        else:
            recognized_face(result_class_name)

        os.remove("../data/emp.jpg")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
