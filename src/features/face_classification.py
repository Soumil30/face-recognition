from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def train_model(train_x, train_y):
    in_encoder = Normalizer(norm='l2')
    train_x = in_encoder.transform(train_x)

    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)

    train_y = out_encoder.transform(train_y)

    model = SVC(kernel='linear', probability=True)

    model.fit(train_x, train_y)

    return model, out_encoder
