import shutil
from os import listdir
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray


def extract_face(filename, face_detector, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = face_detector.detect_faces(pixels)
    if len(results) > 0:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    else:
        return None


def load_faces(directory, face_detector):
    faces = list()

    for filename in listdir(directory):

        path = directory + filename

        face = extract_face(path, face_detector)
        if face is not None:
            faces.append(face)

    shutil.rmtree(directory)

    return faces


def load_dataset(directory):
    x, y = list(), list()

    face_detector = MTCNN()
    for subdir in listdir(directory):

        path = directory + subdir + '/'
        if not isdir(path):
            continue

        faces = load_faces(path, face_detector)

        labels = [subdir for _ in range(len(faces))]

        print('>loaded %d examples for class: %s' % (len(faces), subdir))

        x.extend(faces)
        y.extend(labels)
    return asarray(x), asarray(y)
