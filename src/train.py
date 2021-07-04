from joblib import dump
from keras.models import load_model

from src.face_classification import train_model
from src.face_detection import load_dataset
from src.face_embedding import create_embeddings

directory = "../data/train-images/"

train_X, train_y = load_dataset(directory)
print(train_X.shape, train_y.shape)


embedding_model = load_model('../models/facenet_keras.h5')
print('Loaded Model')

embedding_train_X = create_embeddings(embedding_model, train_X)

print(embedding_train_X.shape)

classification_model = train_model(embedding_train_X, train_y)

dump(classification_model, '../models/classification-model.joblib')
