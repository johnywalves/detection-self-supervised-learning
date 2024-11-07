import os
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_database_for_training(data_in, data_out, img_size):
    filepaths, label_list = [], []

    folders = os.listdir(data_in)
    for fold in folders:
        fold_path = os.path.join(data_in, fold)
        fold_save = os.path.join(data_out, fold)

        if not os.path.exists(fold_save):
            os.makedirs(fold_save)

        f_list = os.listdir(fold_path)
        for f in f_list:
            f_path = os.path.join(fold_path, f)
            f_save = os.path.join(fold_save, f)

            img = Image.open(f_path)
            img.resize(img_size).save(f_save)

            filepaths.append(f_save)
            label_list.append(fold)

    f_series = pd.Series(filepaths, name='filepaths')
    l_series = pd.Series(label_list, name='labels')
    dataframe = pd.concat([f_series, l_series], axis=1)

    print(dataframe)

    stratify = dataframe['labels']
    train_df, test_df = train_test_split(
            dataframe,
            test_size=.3,
            shuffle=True,
            stratify=stratify,
            random_state=42
        )

    train_gen = ImageDataGenerator()
    test_gen = ImageDataGenerator()

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical', 
        color_mode='rgb',
        shuffle= True, 
        batch_size=8
    )

    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col='filepaths',
        y_col='labels',
        target_size=img_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False,
        batch_size=8
    )

    return train_data, test_data
