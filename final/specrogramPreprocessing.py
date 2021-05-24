import numpy as np
import cv2

def spectrogramm_size_preprocessing(initial_spectrogramm):
    """
    Required to reshape the size of the spectrogram to the 357x357
    Initial size is 986x129 = 1x127194~357^2(127449)
    """
    initial_spectrogramm = np.array(initial_spectrogramm)

    resized_spectrogramm = np.resize(initial_spectrogramm, (128, 128))
    return resized_spectrogramm

def contrast_streching(image):
    maximum = max(map(max, image))
    minimum = min(map(min, image))

    contrasted = 255*((image-minimum)/(maximum-minimum))

    r_channel = contrasted
    g_channel = contrasted
    b_channel = contrasted
    return cv2.merge((r_channel, g_channel, b_channel))