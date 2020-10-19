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

    [t, p] = stats.ttest_ind(a=data, b=data_expected)
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


[y1, sr1] = display.read_file("se-2.wav")
[y2, sr2] = display.read_file("se.wav")

[hist1, edges1] = np.histogram(a=y1, bins=50)
[hist2, edges2] = np.histogram(a=y2, bins=50)

[result_chi, p_chi] = chi_sqauare(data=hist1, data_expected=hist2)
[result_t, p_t] = t_stjudent(data=y1, data_expected=y2)
[result_u, p_u] = u_mannwhitney(data=hist1, data_expected=hist2)



print("chi result: ", result_chi, " with p-value = ", p_chi, "\nstud result: ", result_t, " with p-value = ", p_t, "\nmann result: ", result_u, "with p-value = ", p_u)
