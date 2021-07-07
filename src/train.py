import os
import sys

import numpy as np
from joblib import dump
from keras.models import load_model
from numpy import savez_compressed, load

from src.features.face_classification import train_model
from src.features.face_detection import load_dataset
from src.features.face_embedding import create_embeddings

directory = "../data/raw/"

train_X, train_y = load_dataset(directory)
print(train_X.shape, train_y.shape)


embedding_model = load_model('../models/facenet_keras.h5')
print('Loaded Model')

embedding_train_X = create_embeddings(embedding_model, train_X)

print(embedding_train_X.shape)

if embedding_train_X.shape[0] == 0:
    sys.exit("No new data to train")

if os.path.exists('../data/intermediate/image-embeddings.npz'):
    data = load('../data/intermediate/image-embeddings.npz')
    embedding_train_X_past, train_y_past = data['arr_0'], data['arr_1']
    print(embedding_train_X_past.shape)

    embedding_train_X = np.concatenate((embedding_train_X, embedding_train_X_past), axis=0)
    train_y = np.concatenate((train_y, train_y_past), axis=0)

savez_compressed('../data/intermediate/image-embeddings.npz', embedding_train_X, train_y)

print(embedding_train_X.shape)

classification_model, out_encoder = train_model(embedding_train_X, train_y)

dump(classification_model, '../models/classification-model.joblib')
dump(out_encoder, '../data/intermediate/encoder.joblib')
