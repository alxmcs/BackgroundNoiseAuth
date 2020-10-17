from scipy import stats

def chi_sqauare(data, data_expected):

    [chi, p] = stats.chisquare(f_obs=data, f_exp=data_expected)
    decision = 0
    if p <= 0.05:
        decision = 1

    return decision, p

def t_stjudent(data, data_expected):

    [t, p] = stats.ttest_ind(a=data, b=data_expected)
    decision = 0
    if p <= 0.05:
        decision = 1

    return decision, p

def u_mannwhitney(data, data_expected):

    [u, p] = stats.mannwhitneyu(x=data, y=data_expected)
    decicion = 0
    if p<= 0.05:
        decicion = 1

    return decicion, p
