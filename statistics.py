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

        decision_chi = chi_sqauare(data=hist1, data_expected=hist2)
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
