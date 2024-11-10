import os
from src.compensator import from_file_to_array

from sklearn.model_selection import train_test_split

def get_database_for_training(data_in):
    image_list, file_list, label_list = [], [], []

    folders = os.listdir(data_in)
    for fold in folders:
        fold_path = os.path.join(data_in, fold)

        f_list = os.listdir(fold_path)
        for f in f_list:
            f_path = os.path.join(fold_path, f)
            img_array = from_file_to_array(f_path)

            image_list.append(img_array)
            label_list.append(fold)
            file_list.append(f_path)

    x_train, x_test, y_train, y_test, f_train, f_test = train_test_split(
        image_list,
        label_list,
        file_list,
        test_size=.3,
        shuffle=True,
        stratify=label_list,
        random_state=123
    )

    return x_train, x_test, y_train, y_test, f_train, f_test
