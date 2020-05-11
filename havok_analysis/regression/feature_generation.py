'''

      Project: HAVOK analysis
         File: feature_generation.py
 File Created: 03.03.2020
       Author: Paul Wilhelm (paul.wilhelm@stud.uni-goettingen.de)
-----
Last Modified: 11.05.2020 21:50:11
  Modified By: Jan Zetzmann (yangntze+github_havok@gmail.com)
-----
    Copyright: 2020 havokanalysisteam

    All function need to take an array x (x.shape=(n_steps, n_variables)) as 
    first argument and return an array f (x.shape = (n_steps, n_features))
    
'''


import numpy as np
from itertools import combinations_with_replacement
from scipy.special import comb

# --- feature generation ---


def poly(x, k):
    """
    creates all polynomial features from variables upto order k
    """
    if type(k) == int and k > 0:
        n_steps, n_variables = x.shape
        # number of features for each order upto order k
        n_features_per_order = \
            [comb(n_variables, i+1, exact=True, repetition=True)
             for i in range(k)]
        features = np.zeros((n_steps, 1 + sum(n_features_per_order)))
        features[:, 0] = 1  # constant feature
        # index offset for each order -> length=k
        offset = [1, *np.cumsum(n_features_per_order[:-1])+1]
        for i in range(k):
            # list containing tuples of length i+1 with ind. of the variables
            # -> all combinations of the variables of order i+1
            ind_tuple = combinations_with_replacement(range(n_variables), i+1)
            for j, ind in enumerate(ind_tuple):
                # evaluate each index tuple
                features[:, offset[i]+j] = np.prod(x[:, ind], axis=-1)
        return features
    else:
        raise Exception('Error: invalid polynomial order')


# --- names of features ---


def poly_exp(n, k):
    """
    construct description of the features produced by poly(..., k)
    with n variables
    return list of lists of length n with the exponents of each variable
    """
    exponent_list = [[0, ]*n]  # first feature is constant
    for i in range(k):
        ind_tuple = combinations_with_replacement(range(n), i+1)
        for ind in ind_tuple:
            exponent_list.append([ind.count(i) for i in range(n)])
    return exponent_list


def poly_str(n, offset, k):
    """
    calls poly_exp(n, k) and converts exponent list to strings
    labeling of variables starts with v_{offset+1}
    """
    exp_list = poly_exp(n, k)
    str_list = []
    for exp in exp_list:
        ind_list = np.nonzero(exp)[0]
        if len(ind_list) != 0:
            term = []
            for i in ind_list:
                current_term = 'v_{' + '{:d}'.format(i + 1 + offset) + '}'
                if exp[i] > 1:
                    term.append(current_term + '^{:d}'.format(exp[i]))
                else:
                    term.append(current_term)
            str_list.append('$' + ' '.join(term) + '$')
        else:
            str_list.append('$1$')
    return str_list


# --- tests ---


def test_poly():
    # generate random test data
    x = np.random.rand(100, 2)
    features = poly(x, 2)
    assert features.shape == (100, 6)
    assert np.all(np.isclose(features[:, 0], 1))
    assert np.all(np.isclose(features[:, 1:3], x))
    assert np.all(np.isclose(features[:, 3], x[:, 0]**2))
    assert np.all(np.isclose(features[:, 4], x[:, 1]*x[:, 0]))
    assert np.all(np.isclose(features[:, 5], x[:, 1]**2))
    assert poly_exp(2, 2) == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1],
                              [0, 2]]
