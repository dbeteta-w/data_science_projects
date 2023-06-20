import os
import cv2
import shutil

import numpy as np
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Dense
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score
from Enum import GenderLabels, EthnicityGroup
from face_recognition_main import generate_embedding


def create_directories_per_demographic_group(data_path: str,
                                             output_path: str,
                                             create=True) -> List[str]:
    demographic_groups = list()

    for _, dirs, _ in os.walk(data_path, topdown=True):
        for dir in dirs:
            if create:
                os.mkdir(output_path + dir)
            demographic_groups.append(dir)
        break
    return demographic_groups


def get_given_amount_of_subdirectories(data_path: str,
                                       demographic_groups: List[str],
                                       amount: int) -> Dict[str, List[str]]:
    subdirectories_dict = dict()

    for demographic_group in demographic_groups:
        for _, dirs, _ in os.walk(data_path + demographic_group, topdown=True):
            if dirs:
                subdirectories_dict[demographic_group] = dirs[:amount]
    return subdirectories_dict


def copy_images_per_demographic_group(data_path: str,
                                      output_path: str,
                                      demographic_groups: List[str],
                                      subdirectories_dict: Dict[
                                          str, List[str]],
                                      copy=True):
    for demographic_group in demographic_groups:
        for subdirectory in subdirectories_dict[demographic_group]:
            subdirectory_path = data_path + demographic_group + "/" + subdirectory
            for _, _, files in os.walk(subdirectory_path, topdown=True):
                complete_data_path = subdirectory_path + "/" + files[0]
                complete_output_path = output_path + demographic_group + "/" + \
                                       files[0]
                if copy:
                    shutil.copyfile(complete_data_path, complete_output_path)


def get_embedding_from_image(path_to_image: str) -> np.ndarray:
    image = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
    embedding = generate_embedding(image)
    embedding = np.asarray(embedding)
    return np.squeeze(embedding)


def plot_tsne_graphic(tsne_result: np.ndarray):
    plt.title("T-SNE Graphic")
    plt.xlabel("tsne_1")
    plt.ylabel("tsne_2")

    ha = plt.scatter(tsne_result[:500, 0], tsne_result[:500, 1],
                     color="navy")
    hb = plt.scatter(tsne_result[500: 1000, 0], tsne_result[500: 1000, 1],
                     color="darkgreen")
    mb = plt.scatter(tsne_result[1000: 1500, 0], tsne_result[1000: 1500, 1],
                     color="springgreen")
    hn = plt.scatter(tsne_result[1500: 2000, 0], tsne_result[1500: 2000, 1],
                     color="darkred")
    mn = plt.scatter(tsne_result[2000: 2500, 0], tsne_result[2000: 2500, 1],
                     color="lightcoral")
    ma = plt.scatter(tsne_result[2500: 3000, 0], tsne_result[2500: 3000, 1],
                     color="cornflowerblue")

    plt.legend((ha, ma, hb, mb, hn, mn),
               ("Chinese Men", "Chinese Women", "White Men",
                "White Women", "Black Men", "Black Women"),
               bbox_to_anchor=(1.02, 0.5), loc="upper left")

    plt.show()


def baseline_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def get_gender_and_ethnicity(path_to_image: str) -> Tuple[str, str]:
    dict_dir = {
        "HA4K_120": [GenderLabels.Male, EthnicityGroup.Chinese],
        "HB4K_120": [GenderLabels.Male, EthnicityGroup.White],
        "HN4K_120": [GenderLabels.Male, EthnicityGroup.Black],
        "MA4K_120": [GenderLabels.Female, EthnicityGroup.Chinese],
        "MB4K_120": [GenderLabels.Female, EthnicityGroup.White],
        "MN4K_120": [GenderLabels.Female, EthnicityGroup.Black],
    }

    for key in dict_dir.keys():
        if key in path_to_image:
            gender, ethnicity = dict_dir[key]
            return gender, ethnicity


def get_list_of_embeddings_by_ethnicity(dict_data, ethnicity) -> np.ndarray:
    list_of_embeddings = list()

    list_of_embeddings.extend(dict_data[GenderLabels.Male][ethnicity])
    list_of_embeddings.extend(dict_data[GenderLabels.Female][ethnicity])

    return np.array(list_of_embeddings)


def get_list_of_labels(amount: int) -> np.ndarray:
    list_of_labels = list()

    for i in range(amount):
        if i <= amount / 2:
            list_of_labels.append(GenderLabels.Male.value)
        else:
            list_of_labels.append(GenderLabels.Female.value)

    return np.array(list_of_labels)


def get_accuracy_from_model(model,
                            test_data: np.ndarray,
                            test_labels: np.ndarray) -> float:
    predictions = np.array(model.predict(test_data, verbose=0))
    predictions = np.where(predictions < 0.5, 0, 1)

    accuracy = round(accuracy_score(predictions, test_labels), 3)

    return accuracy