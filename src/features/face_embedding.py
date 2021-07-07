from numpy import expand_dims, asarray


def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    y_hat = model.predict(samples)
    return y_hat[0]


def create_embeddings(model, train_x):
    embedding_train_x = list()

    for face_pixels in train_x:
        embedding = get_embedding(model, face_pixels)
        embedding_train_x.append(embedding)

    return asarray(embedding_train_x)
