import statistics
from display import get_data_set
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

inside_points_chi = []
inside_points_student = []
inside_points_umann = []

outside_points_chi = []
outside_points_student = []
outside_points_umann = []

for i in range(len(chi_points)):
    if statistics.regression(chi=chi_points[i], student=student_points[i], mann=umann_points[i]) == 1:
        inside_points_chi.append(chi_points[i])
        inside_points_student.append(student_points[i])
        inside_points_umann.append(umann_points[i])
    else:
        outside_points_chi.append(chi_points[i])
        outside_points_student.append(student_points[i])
        outside_points_umann.append(umann_points[i])


fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('chi')
ax.set_ylabel('mann-whitney')
ax.set_zlabel('student')
ax.scatter(inside_points_chi, inside_points_umann, inside_points_student, c='g', marker='o')
ax.scatter(outside_points_chi, outside_points_umann, outside_points_student, c='r', marker='x')
plt.show()
