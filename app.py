import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

from functions import get_database_for_training

batch_size = 8
img_width = 128
img_height = 128
channel = 3

img_size = (img_width, img_height)
img_shape = (img_width, img_height, channel)

folder_samples = 'data/small_samples'
folder_preprocess = 'data/preprocess'

train_data, test_data = get_database_for_training(folder_samples, folder_preprocess, img_size)

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()

        self.encoder = Sequential([
            Input(shape=img_shape),

            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            Conv2D(16, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            Conv2D(8, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same")
        ])

        self.decoder = Sequential([
            Conv2D(8, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Conv2D(16, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Conv2D(32, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Conv2D(1, (3, 3), activation="sigmoid", padding="same")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss="binary_crossentropy")

history = autoencoder.fit(train_data)

