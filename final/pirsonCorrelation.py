from math import sqrt
import librosa
from baseFunctions import get_data_set, define_class
from sklearn.naive_bayes import BaseNB as classifier
from sklearn.model_selection import train_test_split
from specrogramPreprocessing import contrast_streching, spectrogramm_size_preprocessing
from sklearn.metrics import classification_report
import numpy as np


def corr (x,y):
  mx = sum(sum(x))/(x.shape[0]*x.shape[1])
  my = sum(sum(y))/(y.shape[0]*y.shape[1])
  return sum(sum((x-mx) * (y-my)))/sqrt(sum(sum((x-mx)**2))*sum(sum((y-my)**2)))


data_set, files, srs = get_data_set("dataset")
test_len = round(len(data_set))

correlations = []
labels = []
for i in range(test_len):
    for j in range(i + 1, test_len):

        mel1 = librosa.feature.melspectrogram(y=data_set[i], sr=srs[i], n_mels=128,fmax = 8000)
        mel2 = librosa.feature.melspectrogram(y=data_set[j], sr=srs[j], n_mels=128,fmax = 8000)

        mel1 = mel1[0:127]
        mel2 = mel2[0:127]

        class_ = define_class(files[i], files[j])

        first_transformed = np.array(contrast_streching(spectrogramm_size_preprocessing(mel1)))
        second_transformed = np.array(contrast_streching(spectrogramm_size_preprocessing(mel2)))

        corr_ = corr(first_transformed, first_transformed)

        correlations.append(corr_)
        labels.append(class_)


model = classifier()
correlations = np.array(correlations)
correlations = correlations.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(correlations, labels, test_size=0.30)

model.fit(x_train, y_train)
predicted = model.predict(x_test)

print('For all:\n', classification_report(y_test, predicted))

