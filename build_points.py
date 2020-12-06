import statistics
from display import get_data_set
import matplotlib.pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

data_set = get_data_set("dataset")

chi_points = []
umann_points = []
student_points = []

test_len = round(len(data_set)*0.8)

test_split = data_set[0:test_len]
test_split_2 = data_set[0:test_len]

for y1 in test_split:
    for y2 in test_split_2:
        [a, b, c] = statistics.use_stat_for_spectr(y1, y2)
        chi_points.append(a)
        umann_points.append(b)
        student_points.append(c)

fig = plt.figure()


ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('chi')
ax.set_ylabel('mann-whitney')
ax.set_zlabel('student')
ax.scatter(chi_points, umann_points, student_points)
plt.show()
