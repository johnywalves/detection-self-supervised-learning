import os
import pandas as pd
import numpy as np
from PIL import Image
from src.compensator import from_file_to_array, from_array_to_image, from_decoded_to_image

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.utils import plot_model

def generate_report(prefix, history, model, x_test, y_test, f_test):
    f_series = pd.Series(f_test, name='filepaths')
    l_series = pd.Series(y_test, name='labels')
    df_teste = pd.concat([f_series, l_series], axis=1)
    df_teste['parasitized'] = df_teste['labels'] == 'Parasitized'

    if not os.path.exists('figs'):
        os.makedirs('figs')

    if not os.path.exists('results'):
        os.makedirs('results')
    f = open(os.path.join('results', f'{prefix}.txt'), "w")

    # ==========================================================
    # Loss Report
    # ==========================================================
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    index_loss = np.argmin(valid_loss)
    lowest_loss = valid_loss[index_loss]
    label_loss = f'best epoch= {str(index_loss + 1)}'

    epochs = [i+1 for i in range(len(train_loss))]

    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, valid_loss, label='Validation Loss')
    plt.scatter(index_loss + 1, lowest_loss, s=75, c='red', label=label_loss)

    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figs', f'{prefix}_loss.jpg'))
    plt.clf()
    plt.cla()

    # ==========================================================
    # Composition Encoder
    # ==========================================================
    plot_model(model.encoder, os.path.join('figs', f'{prefix}_encoder.jpg'), show_shapes=True)

    # ==========================================================
    # Composition Decoder
    # ==========================================================
    plot_model(model.decoder, os.path.join('figs', f'{prefix}_decoder.jpg'), show_shapes=True)

    # ==========================================================
    # Visualize dataset 
    # ==========================================================
    num_images = 6
    num_classes = df_teste['labels'].unique()
    num_lines = len(num_classes) * 3

    plt.figure(figsize=(20, 30))

    for index_label, label in enumerate(num_classes):
        df_choose = df_teste[df_teste['labels'] == label].sample(num_images, random_state=98)
        
        for index_file, file in enumerate(df_choose['filepaths'].values):
            img_array = from_file_to_array(file)
            img_encoded = model.predict(img_array)

            index_image = (index_label * 3) * num_images + index_file + 1
            plt.subplot(num_lines, num_images, index_image)
            img = Image.open(file)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label)

            index_image = (index_label * 3) * num_images + index_file + 1 + (num_images * 1)
            plt.subplot(num_lines, num_images, index_image)
            img = from_array_to_image(img_array)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label)

            index_image = (index_label * 3) * num_images + index_file + 1 + (num_images * 2)
            plt.subplot(num_lines, num_images, index_image)
            img = from_decoded_to_image(img_encoded)
            plt.imshow(img)
            plt.axis('off')
            plt.title(label)

    plt.tight_layout()
    plt.savefig(os.path.join('figs', f'{prefix}_visualization.jpg'))
    plt.clf()
    plt.cla()

    # ==========================================================
    # Calculate reconstruction errors and threshold
    # ==========================================================
    reconstruction_errors = []
    for img_test in x_test:
        reconstructed = model.predict(img_test)
        mse = np.mean(np.square(img_test - reconstructed), axis=(1, 2, 3))
        reconstruction_errors.append(mse[0])

    threshold = np.percentile(reconstruction_errors, 50)
    f.write(f'Limiar para anomalias: {threshold}\n')

    anomalies = reconstruction_errors > threshold
    f.write(f'Número de anomalias detectadas: {len(anomalies)}\n')

    anomalies_series = pd.Series(anomalies, name='anomalies')
    error_series = pd.Series(reconstruction_errors, name='errors')
    df_teste = pd.concat([df_teste, anomalies_series, error_series], axis=1)

    plt.hist(reconstruction_errors, bins=10)
    plt.xlabel("Erro de Reconstrução")
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('figs', f'{prefix}_errors.jpg'))
    plt.clf()
    plt.cla()

    # ==========================================================
    # Visualize anomalies
    # ==========================================================
    plt.figure(figsize=(20, 10))

    num_anomalies = df_teste['anomalies'].unique()
    for index_anomaly, anomaly in enumerate(num_anomalies):
        df_choose = df_teste[df_teste['anomalies'] == anomaly].sample(num_images, random_state=98)

        index_file = 0
        for index, element in df_choose.iterrows():
            filepath = element['filepaths']
            label = element['labels']

            plt.subplot(len(num_anomalies), num_images, index_anomaly * num_images + index_file + 1)
            img = Image.open(filepath)
            plt.imshow(img)
            plt.axis('off')
            plt.title(('Anomaly' if anomaly else 'Normal') + ' (' + label + ')' )

            index_file += 1

    plt.tight_layout()
    plt.savefig(os.path.join('figs', f'{prefix}_anomalies.jpg'))
    plt.clf()
    plt.cla()

    parasitized = df_teste['parasitized']
    anomalies = df_teste['anomalies']

    f.write(classification_report(parasitized, anomalies))
    f.close()
