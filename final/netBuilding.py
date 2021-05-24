import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.layers import AveragePooling2D

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import preprocess_input


def siam_net(first_net, second_net, first_audio, second_audio):

    x = preprocess_input(first_audio)
    x = np.expand_dims(x, axis=0)
    first_vect = first_net.predict(x)

    y = preprocess_input(second_audio)
    y = np.expand_dims(y, axis=0)
    second_vect = second_net.predict(y)

    return first_vect, second_vect


def sub_net():
    baseModel = VGG16(weights='imagenet', input_tensor=Input(shape=(128, 128, 3)), include_top=False)

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(4096, activation="relu")(headModel)
    # headModel = Dropout(0.5)(headModel)
    headModel = Dense(4096, activation="relu")(headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    model = Model(inputs=baseModel.input, outputs=headModel)

    return model

