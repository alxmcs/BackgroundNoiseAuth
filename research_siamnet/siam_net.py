from display import get_data_set
from scipy.signal import spectrogram
import skimage.color.colorconv as conv
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def spectrogramm_size_preprocessing(initial_spectrogramm, freq, time):
    """
    Required to reshape the size of the spectrogram to the 224x224
    Initial size is 1968x129
    """
    initial_spectrogramm = np.array(initial_spectrogramm)
    lenth = len(freq)*len(time)
    in_one_line = initial_spectrogramm.reshape((1, lenth))
    resized_spectrogramm = np.resize(in_one_line[0, 50000:120000], (1, 50176)).reshape((224, 224))  # 224x224 = 50176

    return resized_spectrogramm

def spectrogramm_color_preprocessing(spectrogramm):
    """
    Required to convert the gray spectrogram to RGB
    """
    rgb_image = conv.gray2rgb(spectrogramm)
    return rgb_image

def euclidean_dist(first_vect, second_vect):
    return np.linalg.norm(first_vect - second_vect)

def siam_net(first_net, second_net, first_audio, second_audio):

    x = preprocess_input(first_audio)
    x = np.expand_dims(x, axis=0)
    first_vect = first_net.predict(x)

    y = preprocess_input(second_audio)
    y = np.expand_dims(y, axis=0)
    second_vect = second_net.predict(y)

    return first_vect, second_vect


data_set, files = get_data_set("dataset")
test_len = round(len(data_set))

first_model_full = VGG16()
second_model_full = VGG16()

first_model = Model(inputs=first_model_full.input, outputs=first_model_full.get_layer('fc2').output)
second_model = Model(inputs=second_model_full.input, outputs=second_model_full.get_layer('fc2').output)

euclideans = []

for i in range(test_len):
    for j in range(i+1, test_len):

        f1, t1, s1 = spectrogram(data_set[i])
        f2, t2, s2 = spectrogram(data_set[j])

        first_transformed = spectrogramm_color_preprocessing(spectrogramm_size_preprocessing(s1, f1, t1))
        second_transformed = spectrogramm_color_preprocessing(spectrogramm_size_preprocessing(s2, f2, t2))

        first_vect, second_vect = siam_net(first_model, second_model, first_transformed, second_transformed)

        r = euclidean_dist(first_vect, second_vect)
        euclideans.append(r)

print(euclideans)
