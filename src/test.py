from random import choice

import numpy
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
import mtcnn
from mtcnn import MTCNN
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
from os import listdir
from os.path import isdir
# from PIL import Image
import tensorflow as tf
from datetime import datetime
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


# def extract_face(filename, required_size=(160, 160)):
#     # load image from file
#     image = Image.open(filename)
#     # convert to RGB, if needed
#     image = image.convert('RGB')
#     # convert to array
#     pixels = asarray(image)
#     # create the detector, using default weights
#     detector = MTCNN()
#     # detect faces in the image
#     results = detector.detect_faces(pixels)
#     # extract the bounding box from the first face
#     x1, y1, width, height = results[0]['box']
#     # bug fix
#     x1, y1 = abs(x1), abs(y1)
#     x2, y2 = x1 + width, y1 + height
#     # extract the face
#     face = pixels[y1:y2, x1:x2]
#     # resize pixels to the model size
#     image = Image.fromarray(face)
#     image = image.resize(required_size)
#     face_array = asarray(image)
#     return face_array
#
#
# def load_faces(directory):
#     faces = list()
#     # enumerate files
#     for filename in listdir(directory):
#         # path
#         path = directory + filename
#         # get face
#         face = extract_face(path)
#         # store
#         faces.append(face)
#     return faces
#
#
# # load a dataset that contains one subdir for each class that in turn contains images
# def load_dataset(directory):
#     X, y = list(), list()
#     # enumerate folders, on per class
#     for subdir in listdir(directory):
#         # path
#         path = directory + subdir + '/'
#         # skip any files that might be in the dir
#         if not isdir(path):
#             continue
#         # load all faces in the subdirectory
#         faces = load_faces(path)
#         # create labels
#         labels = [subdir for _ in range(len(faces))]
#         # summarize progress
#         print('>loaded %d examples for class: %s' % (len(faces), subdir))
#         # store
#         X.extend(faces)
#         y.extend(labels)
#     return asarray(X), asarray(y)


# load the face dataset
# data = load('../data/image-vectors-dataset.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
# print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
#
#
# # load the facenet model
# model = load_model('../data/facenet_keras.h5')
# print('Loaded Model')
#
# # startTime = datetime.now()
#
#
# # get the face embedding for one face
# def get_embedding(_model, _face_pixels):
#     print("Inside !!!!!!!!")
#     # scale pixel values
#     _face_pixels = _face_pixels.astype('float32')
#     # standardize pixel values across channels (global)
#     mean, std = _face_pixels.mean(), _face_pixels.std()
#     _face_pixels = (_face_pixels - mean) / std
#     # transform face into one sample
#     samples = expand_dims(_face_pixels, axis=0)
#     # make prediction to get embedding
#     print("Before embedding!!!!!!")
#     y_hat = _model.predict(samples)
#     print("DOne!!!!!!!!")
#     return y_hat[0]
#
#
# startTime = None
# x = False
# newTrainX = list()
# # embedding = get_embedding(model, trainX[0])
# # newTrainX.append(embedding)
# # newTrainX = asarray(newTrainX)
# # print(newTrainX.shape)
# for face_pixels in trainX:
#     if x:
#         startTime = datetime.now()
#     embedding = get_embedding(model, face_pixels)
#     newTrainX.append(embedding)
#     x = True
# newTrainX = asarray(newTrainX)
# print(newTrainX.shape)
# # convert each face in the test set to an embedding
#
# for i in range(20):
#     newTestX = list()
#     for face_pixels in testX:
#         embedding = get_embedding(model, face_pixels)
#         newTestX.append(embedding)
#
#     newTestX = asarray(newTestX)
#     print(newTestX.shape)
# # save arrays to one file in compressed format
# # savez_compressed('image-embeddings-gpu.npz', newTrainX, trainy, newTestX, testy)
# print("Time taken:", datetime.now() - startTime)


data = load('../data/celebrity-dataset/image-vectors-dataset.npz')
testX_faces = data['arr_2']

# load dataset
data = load('../data/celebrity-dataset/image-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))


# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)


# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)


# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)


# predict
# yhat_train = model.predict(trainX)
# yhat_test = model.predict(testX)
# # score
# score_train = accuracy_score(trainy, yhat_train)
# score_test = accuracy_score(testy, yhat_test)
# # summarize
# print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))


selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
plt.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
plt.title(title)
plt.show()



