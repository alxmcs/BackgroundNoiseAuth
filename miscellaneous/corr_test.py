from scipy import signal, stats
import numpy as np
from matplotlib import pyplot as plt
import librosa
from librosa import display
from math import sqrt

def corr (x,y):
  mx = sum(sum(x))/(x.shape[0]*x.shape[1])
  my = sum(sum(y))/(y.shape[0]*y.shape[1])
  return sum(sum((x-mx) * (y-my)))/sqrt(sum(sum((x-mx)**2))*sum(sum((y-my)**2)))

a, samplerate_a = librosa.load('1_1.wav')
b, samplerate_b = librosa.load('1_2.wav')
c, samplerate_c = librosa.load('2.wav')

Sa = librosa.feature.melspectrogram(y=a, sr=samplerate_a, n_mels=128,fmax = 8000)
Sb = librosa.feature.melspectrogram(y=b, sr=samplerate_b, n_mels=128,fmax = 8000)
Sc = librosa.feature.melspectrogram(y=c, sr=samplerate_c, n_mels=128,fmax = 8000)

fig=plt.figure(figsize=(30, 5))
fig.add_subplot(1,3,1)
display.specshow(librosa.power_to_db(Sa,ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
plt.colorbar(format='%+2.0f dB')
fig.add_subplot(1,3,2)
display.specshow(librosa.power_to_db(Sb,ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
plt.colorbar(format='%+2.0f dB')
fig.add_subplot(1,3,3)
display.specshow(librosa.power_to_db(Sc,ref = np.max), y_axis = 'mel', fmax = 8000, x_axis = 'time')
plt.colorbar(format='%+2.0f dB')
plt.show()

ab = corr(Sb[0:127,0:127], Sa[0:127,0:127])
ac = corr(Sa[0:127,0:127], Sc[0:127,0:127])
bc = corr(Sc[0:127,0:127], Sb[0:127,0:127])

print('<should be close to 1> Corr(Sa,Sb)=', ab)
print('<should be close to 0> Corr(Sa,Sc)=', ac)
print('<should be close to 0> Corr(Sb,Sc)=', bc)