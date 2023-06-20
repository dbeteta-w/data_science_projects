import os

import numpy as np
import Utils as ut

from sklearn.manifold import TSNE

################ TASK 1 #########################

# Read the DiveFace database and obtain the embeddings of 50 face images
# (1 image per subject) from the 6 demographic groups (500*6=3000 embeddings in total).

# DiveFace contains face images from 3 demographic groups (3 ethnicity and 2 gender).

DATA_PATH = "imagenes_task1-2/"
ORIGINAL_RAR = "4K_120/"
AMOUNT_OF_IMAGES = 500

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

################ TASK 2 #########################

# Using t-SNE, represent the embeddings and its
# demographic group. Can you differentiate the different demographic groups?

list_of_emmbedings = list()

for root, _, files in os.walk("imagenes_task1-2", topdown=True):
    for name in files:
        embedding = ut.get_embedding_from_image(
            path_to_image=os.path.join(root, name)
        )
        list_of_emmbedings.append(embedding)

tsne = TSNE(n_components=2)
tsne_result = tsne.fit_transform(np.array(list_of_emmbedings))

ut.plot_tsne_graphic(tsne_result)

################ TASK 3 #########################

# Using the ResNet-50 embedding (freeze the model),
# train your own attribute classifiers (ethnicity and gender).

# Recommendation: use a simple dense layer with a softmax output.
# Divide DiveFace into train and test.

ut.baseline_model()
