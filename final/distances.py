import numpy as np


def euclidean_dist(first_vect, second_vect):
    return np.linalg.norm(first_vect - second_vect)

def dist_cos(first_vect, second_vect):
    return (sum(a*b for a,b in zip(first_vect[0],second_vect[0])))/(((sum(first_vect[0]**2))**0.5) *((sum(second_vect[0]**2))**0.5))

def dist_chebyshev(first_vect,second_vect):
    return max(abs(first_vect[0]-second_vect[0]))

def dist_manhetten(first_vect,second_vect):
    return np.sum(np.abs(first_vect-second_vect))

