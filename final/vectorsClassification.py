import os
import numpy as np
from netBuilding import sub_net, siam_net
from baseFunctions import define_class, get_data_set
from specrogramPreprocessing import spectrogramm_size_preprocessing, contrast_streching
import display
from classificators import allClassifiers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model1 = sub_net()
model2 = sub_net()

data_set, files, srs = get_data_set("dataset")
test_len = round(len(data_set))
first_vecs = []
scnd_vecs = []
pairs = []

for i in range(test_len):
    for j in range(i + 1, test_len):
        mel1 = display.audio_to_mel(data_set[i], srs[i])
        mel2 = display.audio_to_mel(data_set[j], srs[j])

        mel1 = mel1[0:127]
        mel2 = mel2[0:127]

        class_ = define_class(files[i], files[j])

        first_transformed = contrast_streching(spectrogramm_size_preprocessing(mel1))
        second_transformed = contrast_streching(spectrogramm_size_preprocessing(mel2))

        first_vect, second_vect = siam_net(model1, model2, first_transformed, second_transformed)

        first_vecs.append(first_vect[0])
        scnd_vecs.append(second_vect[0])
        pairs.append(class_)


lenth = len(pairs)

data1 = first_vecs
data2 = scnd_vecs

labels = pairs
data = []
tmp = []
for i,j in zip(data1, data2):
    tmp = np.concatenate((i, j), axis=0)
    data.append(tmp)

data.reverse()

allClassifiers(data, labels)
