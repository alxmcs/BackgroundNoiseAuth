from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import librosa


a, samplerate_a = librosa.load('1_1.wav')
b, samplerate_b = librosa.load('1_2.wav')
c, samplerate_c = librosa.load('2.wav')

freq, A = signal.welch(a)
freq, B = signal.welch(b)
freq, C = signal.welch(c)

#Itakura-Saito 
def itakuro_saito(p,q):
  if q ==0:
    q = 1e-17
  return p/q - np.log(p/q)-1

#Kullback-Leibner
def kullback_leibner(p,q):
  if q ==0:
    q = 1e-17
  return p*np.log(p/q) + q-p


is_vect = np.vectorize(itakuro_saito)
kl_vect = np.vectorize(kullback_leibner)

print('<should be small> Dis(a,b)=',sum(is_vect(A,B)))
print('<should be small> Dis(b,a)=',sum(is_vect(B,A)))
print('<should be huge> Dis(a,c)=', sum(is_vect(A,C)))
print('<should be huge> Dis(c,a)=', sum(is_vect(C,A)))
print('<should be huge> Dis(b,c)=', sum(is_vect(B,C)))
print('<should be huge> Dis(c,b)=', sum(is_vect(C,B)))
print('=============================================')
print('<should be small> Dkl(a,b)=',sum(kl_vect(A,B)))
print('<should be small> Dkl(b,a)=',sum(kl_vect(B,A)))
print('<should be huge> Dkl(a,c)=', sum(kl_vect(A,C)))
print('<should be huge> Dkl(c,a)=', sum(kl_vect(C,A)))
print('<should be huge> Dkl(b,c)=', sum(kl_vect(B,C)))
print('<should be huge> Dkl(c,b)=', sum(kl_vect(C,B)))