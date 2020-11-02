import display
from scipy.signal import spectrogram
import statistics
import numpy as np
np.seterr(divide='ignore')
np.seterr(invalid='ignore')


[y1, sr1] = display.read_file("se-2.wav")
[y2, sr2] = display.read_file("se.wav")

f1, t1, s1 = spectrogram(x=y1, fs=sr1)
f2, t2, s2 = spectrogram(x=y2, fs=sr2)

common_freqs = list(set(f1) & set(f2))

chi = 0
umann = 0
student = 0
indexes = list(range(0, len(f1), 2))

for i in indexes:
    [hist1, edges1] = np.histogram(a=s1[i], bins=492)
    [hist2, edges2] = np.histogram(a=s2[i], bins=492)

    decision_chi = statistics.chi_sqauare(data=hist1, data_expected=hist2)
    decision_umann = statistics.u_mannwhitney(data=hist1, data_expected=hist2)
    decision_student = statistics.t_stjudent(data=s1[i], data_expected=s2[i])

    chi += decision_chi
    umann += decision_umann
    student += decision_student

print(chi/len(indexes), umann/len(indexes), student/len(indexes))
