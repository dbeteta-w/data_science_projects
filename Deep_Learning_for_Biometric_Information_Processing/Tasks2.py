import os

import Utils as ut
import numpy as np

from sklearn.model_selection import train_test_split
from Enum import EthnicityGroup
from tabulate import tabulate


################ TASK 1 #########################

# Train 3 different Gender Classifiers (previous Task 1.4)
# using images from same ethnic group: Model A (only Asian),
# Model B (only Black), Model C (only Caucasian)

################ TASK 2 #########################

# Evaluate the 3 Gender Classifiers (previous Task 2.1) using images
# from each of the three ethnic groups.

DATA_PATH = "imagenes_task2/"
ORIGINAL_RAR = "4K_120/"
AMOUNT_OF_IMAGES = 750  # 500 (train) + 250 (test)

# Rearrange and get the proper amount of data
demographic_groups = ut.create_directories_per_demographic_group(
    data_path=DATA_PATH + ORIGINAL_RAR,
    output_path=DATA_PATH,
    create=False,
)
subdirectories_dict = ut.get_given_amount_of_subdirectories(
    data_path=DATA_PATH + ORIGINAL_RAR,
    demographic_groups=demographic_groups,
    amount=AMOUNT_OF_IMAGES
)
ut.copy_images_per_demographic_group(
    data_path=DATA_PATH + ORIGINAL_RAR,
    output_path=DATA_PATH,
    demographic_groups=demographic_groups,
    subdirectories_dict=subdirectories_dict,
    copy=False
)

dict_data = dict()

# Get the embeddings from the images in an ordered manner
for root, dirs, files in os.walk(DATA_PATH, topdown=True):
    gender, ethnicity = "", ""
    list_of_embeddings = list()
    for name in files:
        gender, ethnicity = ut.get_gender_and_ethnicity(
            path_to_image=os.path.join(root, name)
        )
        embedding = ut.get_embedding_from_image(
            path_to_image=os.path.join(root, name)
        )
        list_of_embeddings.append(embedding)

    if gender and gender not in dict_data:
        dict_data[gender] = dict()

    if ethnicity and ethnicity not in dict_data[gender]:
        dict_data[gender][ethnicity] = list_of_embeddings

# Get the processed data and its labels
list_of_embeddings_chinese = ut.get_list_of_embeddings_by_ethnicity(
    dict_data=dict_data, ethnicity=EthnicityGroup.Chinese
)
list_of_embeddings_black = ut.get_list_of_embeddings_by_ethnicity(
    dict_data=dict_data, ethnicity=EthnicityGroup.Black
)
list_of_embeddings_white = ut.get_list_of_embeddings_by_ethnicity(
    dict_data=dict_data, ethnicity=EthnicityGroup.White
)

list_of_labels = ut.get_list_of_labels(
    amount=len(list_of_embeddings_chinese)
)

# Split the data in train and test
X_train_chinese, X_test_chinese, \
    y_train_chinese, y_test_chinese = train_test_split(
    list_of_embeddings_chinese,
    list_of_labels,
    test_size=0.33, random_state=123, shuffle=True
)
X_train_black, X_test_black, \
    y_train_black, y_test_black = train_test_split(
    list_of_embeddings_black,
    list_of_labels,
    test_size=0.33, random_state=123, shuffle=True
)
X_train_white, X_test_white, \
    y_train_white, y_test_white = train_test_split(
    list_of_embeddings_white,
    list_of_labels,
    test_size=0.33, random_state=123, shuffle=True
)

# Train 3 Gender Classifiers differentiated by ethnicity
model_chinese = ut.baseline_model()
model_chinese.fit(X_train_chinese, y_train_chinese, epochs=100, verbose=0)

model_black = ut.baseline_model()
model_black.fit(X_train_black, y_train_black, epochs=100, verbose=0)

model_white = ut.baseline_model()
model_white.fit(X_train_white, y_train_white, epochs=100, verbose=0)

# Test the 3 Classifiers with the 3 ethnic test data
accuracies_chinese_model = [
    EthnicityGroup.Chinese.name + " Model",
    ut.get_accuracy_from_model(model_chinese, X_test_chinese, y_test_chinese),
    ut.get_accuracy_from_model(model_chinese, X_test_black, y_test_black),
    ut.get_accuracy_from_model(model_chinese, X_test_white, y_test_white),
]

accuracies_black_model = [
    EthnicityGroup.Black.name + " Model",
    ut.get_accuracy_from_model(model_black, X_test_chinese, y_test_chinese),
    ut.get_accuracy_from_model(model_black, X_test_black, y_test_black),
    ut.get_accuracy_from_model(model_black, X_test_white, y_test_white),
]

accuracies_white_model = [
    EthnicityGroup.White.name + " Model",
    ut.get_accuracy_from_model(model_white, X_test_chinese, y_test_chinese),
    ut.get_accuracy_from_model(model_white, X_test_black, y_test_black),
    ut.get_accuracy_from_model(model_white, X_test_white, y_test_white),
]

# Plot the matrix of results
accuracies = [
    accuracies_chinese_model,
    accuracies_black_model,
    accuracies_white_model
]
headers = [
    EthnicityGroup.Chinese.name + " Data",
    EthnicityGroup.Black.name + " Data",
    EthnicityGroup.White.name + " Data",
]

print("-------------------Task 1-------------------")
print(tabulate(accuracies, headers=headers, tablefmt='fancy_grid'))

################ TASK 3 #########################

# Train one Gender Classifiers (previous Task 1.4) using
# images from all three ethnic groups.

################ TASK 4 #########################

# Evaluate the Gender Classifier (previous Task 2.3) using
# images from each of the three ethnic groups.

# Bring together the data
X_train = list(X_train_chinese) + list(X_train_black) + list(X_train_white)
y_train = list(y_train_chinese) + list(y_train_black) + list(y_train_white)

# Train a general Gender Classifier
model = ut.baseline_model()
model.fit(np.array(X_train), np.array(y_train), epochs=100, verbose=0)

# Test the Classifier with the 3 ethnic test data
accuracies_model = [
    "Model",
    ut.get_accuracy_from_model(model, X_test_chinese, y_test_chinese),
    ut.get_accuracy_from_model(model, X_test_black, y_test_black),
    ut.get_accuracy_from_model(model, X_test_white, y_test_white),
]

print("-------------------Task 2-------------------")
print(tabulate([accuracies_model], headers=headers, tablefmt='fancy_grid'))