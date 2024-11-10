import numpy as np
from src.constants import img_size

from tensorflow import expand_dims
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

def from_file_to_array(f_path):
    img = load_img(
        f_path,
        target_size=img_size,
        color_mode='rgb',
        interpolation='nearest'
    )
    img_data = img_to_array(img) / 255.0
    img_array = expand_dims(img_data, axis=0)
    return img_array

def from_array_to_image(img_array):
    rgb_array = np.array(img_array.numpy()[0] * 255.0)
    img = array_to_img(rgb_array)
    return img

def from_decoded_to_image(img_array):
    rgb_array = np.array(img_array)[0] * 255.0
    img = array_to_img(rgb_array)
    return img
