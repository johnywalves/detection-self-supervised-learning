import os
from src.compensator import from_file_to_array

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adamax

import warnings
warnings.filterwarnings('ignore')

def get_fitness(model, name, x_train, x_test):
    early_stopping = EarlyStopping(
        patience=10,
        min_delta=0,
        mode=min,
        monitor='val_loss',
        verbose=0,
        restore_best_weights=True,
        baseline=None
    )

    plateau = ReduceLROnPlateau(
        patience=4,
        mode=min,
        monitor='val_loss',
        factor=.2,
        verbose=0
    )

    model.compile(Adamax(learning_rate=.0005), loss='mse')

    history = model.fit(
        x_train,
        x_train,
        verbose=1,
        validation_data=(x_test, x_test),
        epochs=1000,
        callbacks=[early_stopping, plateau]
    )

    if not os.path.exists('model'):
        os.makedirs('model')

    model.save(os.path.join('model', f'{name}_model.keras'))

    return history
