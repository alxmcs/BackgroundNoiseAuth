import pandas as pd
from sklearn.naive_bayes import BaseNB as classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
result_df = pd.read_csv('mel_metrics.csv')

data = result_df['euclidean']
labels = result_df['is a pair']
data2 = result_df['manhetten']
data3 = result_df['cosinuses']
data4 = result_df['chebysheves']

data = data.to_numpy()
data = data.reshape(-1, 1)
data2 = data2.to_numpy()
data2 = data2.reshape(-1, 1)
data3 = data3.to_numpy()
data3 = data3.reshape(-1, 1)
data4 = data4.to_numpy()
data4 = data4.reshape(-1, 1)

chars_data = np.column_stack((data, data2, data3, data4))

model_euclidean = classifier()
model_manhetten = classifier()
model_cosinus = classifier()
model_chebyshev = classifier()
model_all = classifier()

x_train_euclidean, x_test_euclidean, y_train_euclidean, y_test_euclidean = train_test_split(data, labels.values.ravel(), test_size=0.30)
x_train_manhetten, x_test_manhetten, y_train_manhetten, y_test_manhetten = train_test_split(data2, labels.values.ravel(), test_size=0.30)
x_train_cosinus, x_test_cosinus, y_train_cosinus, y_test_cosinus = train_test_split(data3, labels.values.ravel(), test_size=0.30)
x_train_chebyshev, x_test_chebyshev, y_train_chebyshev, y_test_chebyshev = train_test_split(data4, labels.values.ravel(), test_size=0.30)
x_train_all, x_test_all, y_train_all, y_test_all = train_test_split(chars_data, labels.values.ravel(), test_size=0.30)

model_euclidean.fit(x_train_euclidean, y_train_euclidean)
model_manhetten.fit(x_train_manhetten, y_train_manhetten)
model_cosinus.fit(x_train_cosinus, y_train_cosinus)
model_chebyshev.fit(x_train_chebyshev, y_train_chebyshev)
model_all.fit(x_train_all, y_train_all)

predicted_euclidean = model_euclidean.predict(x_test_euclidean)
predicted_manhetten = model_manhetten.predict(x_test_manhetten)
predicted_cosinus = model_cosinus.predict(x_test_cosinus)
predicted_chebyshev = model_chebyshev.predict(x_test_chebyshev)
predicted_all = model_all.predict(x_test_all)

euclidean_report = classification_report(y_test_euclidean, predicted_euclidean)
manhetten_report = classification_report(y_test_manhetten, predicted_manhetten)
cosinus_report = classification_report(y_test_cosinus, predicted_cosinus)
chebyshev_report = classification_report(y_test_chebyshev, predicted_chebyshev)
all_report = classification_report(y_test_all, predicted_all)

print("Euclidean:\n", euclidean_report)
print("Manhetten:\n", manhetten_report)
print("Cosinuses:\n", cosinus_report)
print("Chebyshev:\n", chebyshev_report)
print("All:\n", all_report)