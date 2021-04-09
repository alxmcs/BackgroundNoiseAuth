from scipy import stats
import display
import numpy as np

def chi_sqauare(data, data_expected):

    [chi, p] = stats.chisquare(f_obs=data, f_exp=data_expected)
    decision = 0
    if p <= 0.05:
        decision = 1

    return decision

def t_stjudent(data, data_expected):

    [t, p] = stats.ttest_ind(a=data, b=data_expected, equal_var=False)
    decision = 0
    if p <= 0.05:
        decision = 1

    return decision

def u_mannwhitney(data, data_expected):

    [u, p] = stats.mannwhitneyu(x=data, y=data_expected)
    decicion = 0
    if p<= 0.05:
        decicion = 1

    return decicion

def filter_zeroes(s1, s2):
    zero_index_1 = []
    zero_index_2 = []

    for j in range(len(s1)):
        if s1[j] != 0 & s1[j] != np.NaN:
            zero_index_1.append(j)
        if s2[j] != 0 & s1[j] != np.NaN:
            zero_index_2.append(j)

    common_indexes = []

    for k in range(min(len(zero_index_1), len(zero_index_2))):
        if zero_index_1[k] == zero_index_2[k]:
            common_indexes.append(k)

    filtered_s_1 = []
    filtered_s_2 = []

    for i in common_indexes:
        filtered_s_1.append(s1[i])
        filtered_s_2.append(s2[i])

    return filtered_s_1, filtered_s_2

def use_stat_for_spectr(y1, y2):

    f1, t1, s1 = spectrogram(x=y1)
    f2, t2, s2 = spectrogram(x=y2)

    chi = 0
    umann = 0
    student = 0

    indexes = list(range(0, len(f1), 2))

    for i in indexes:
        [hist1, edges1] = np.histogram(a=s1[i], bins=492)
        [hist2, edges2] = np.histogram(a=s2[i], bins=492)
        filtered_s1, filtered_s2 = filter_zeroes(hist1, hist2)

        decision_chi = chi_sqauare(data=filtered_s1, data_expected=filtered_s2)
        decision_umann = u_mannwhitney(data=s1[i], data_expected=s2[i])
        decision_student = t_stjudent(data=s1[i], data_expected=s2[i])

        chi += decision_chi
        umann += decision_umann
        student += decision_student

    chi_point = chi/len(indexes)
    umann_point = umann/len(indexes)
    student_point = student/len(indexes)

    return chi_point, umann_point, student_point

def weighted_vote(decisions, weights):
    decisions = np.array(decisions)
    weights = np.array(weights)
    f = sum(decisions*weights)
    return f > 0.7

def regression(chi, student, mann):

    inside = 0
    if (chi > 0.85) and (student > 0.7) and (mann > 0.8):
        inside = 1

    return inside
