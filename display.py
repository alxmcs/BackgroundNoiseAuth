import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import os


def read_file(file_name):
    y, sr = librosa.load(file_name)
    return y, sr

def get_data_set(dyrectory):

    files_name = os.listdir("dataset")
    y = []

    for file in files_name:
        path = dyrectory + "/" + file
        [a, b] = read_file(path)
        y.append(a)

    return y

def audio_to_mel(y,sr):
    return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax = 8000)


def mel_to_audio(S):
    return librosa.feature.inverse.mel_to_audio(S)


def forward_fft(y, sr):
    freqs = np.fft.fftfreq(y.size, sr)
    vals = np.fft.fft(y)
    return [vals, freqs]


def inverse_fft(vals):
    return np.fft.ifft(vals)


def display_melspectrogram(y, sr,title):
    S = audio_to_mel(y,sr)
    plt.figure(figsize=(10, 4))
    display.specshow(librosa.power_to_db(S,ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()


def display_track(y,sr,title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.title(title)


def display_frequency(y, sr, title):
    [vals, freqs] = forward_fft(y, sr)
    [f,arr] = plt.subplots(2, 1,figsize=(10, 4))
    arr[0].plot(freqs, np.real(vals))
    arr[0].title.set_text(title+' real')
    arr[1].plot(freqs, np.imag(vals))
    arr[1].title.set_text(title+' imaginary')
    plt.tight_layout()

def save_file(path, y, sr):
    sf.write(path, y, sr)


[y,sr]=read_file('test.wav')
display_track(y,sr,'Amplitude')
display_melspectrogram(y,sr,'Mel Spectrogram')
display_frequency(y,sr,'FFT Spectrum')
plt.show()
