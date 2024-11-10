from src.constants import img_shape, channel, n_bottleneck

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU

class AnomalyConv2DDetector(Model):
    def __init__(self):
        super(AnomalyConv2DDetector, self).__init__()

        self.encoder = Sequential([
            Input(shape=img_shape),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),

            Conv2D(16, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same")
        ])

        self.decoder = Sequential([
            Conv2D(16, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Conv2D(32, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Conv2D(64, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),

            Dense(channel, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyDenseDetector(Model):
    def __init__(self):
        super(AnomalyDenseDetector, self).__init__()

        self.encoder = Sequential([
            Input(shape=img_shape),

            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
        ])

        self.decoder = Sequential([
            Dense(32, activation="relu"),
            Dense(64, activation="relu"),
            Dense(channel, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AnomalyLeakyDetector(Model):
    def __init__(self):
        super(AnomalyLeakyDetector, self).__init__()

        self.encoder = Sequential([
            Input(shape=img_shape),

            Dense(img_shape[0] * 2),
            BatchNormalization(),
            LeakyReLU(),

            Dense(img_shape[0]),
            BatchNormalization(),
            LeakyReLU(),

            Dense(n_bottleneck)
        ])

        self.decoder = Sequential([
            Dense(img_shape[0]),
            BatchNormalization(),
            LeakyReLU(),

            Dense(img_shape[0] * 2),
            BatchNormalization(),
            LeakyReLU(),

            Dense(channel, activation='linear')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded