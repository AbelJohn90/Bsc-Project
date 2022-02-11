# import pdb
import numpy as np
import sympy as sp
import time

from joblib import Parallel, delayed
from itertools import combinations, chain
from scipy.sparse import coo_matrix


def _certificate(variables, g, deg, **kwargs):
    # var_dict = {m: i for i, m in enumerate(sp.itermonomials(variables, deg))}
    # g = g
    h_poly_ = delayed(h_poly)
    fun = Parallel(**kwargs)
    iter_ = enumerate(alpha_beta(len(g), deg))
    result = fun(
            h_poly_(col, idx, coefs) for col, (idx, coefs) in iter_)
    result = list(chain.from_iterable(result))
    result = [x for x in zip(*result)]

    data, row, column = result
    data = np.array(data, dtype=float)
    row = np.array(row, dtype=int)
    column = np.array(column, dtype=int)

    return coo_matrix(
            (data, (row, column)), shape=(max(row) + 1, column[-1] + 1))


def lp_matrix_certificate(variables, g, deg, **kwargs):
    pass


def h_poly(col, idx, coefs):
    """
    Polynomials composing the certificate

    Args:
        g (tuple): polynomial constraints defining the semialgebraic set
            K:={x: 0 <= g_i(x) <= 1, i=1,...,m}, as sympy expressions.
        idx (np.array): indices of nonzero exponents (alpha_i, beta_i)
        coefs (np.array): coefficients alpha_i, beta_i corresponding to
            the nonzero elements of the partition
    """
    poly = sp.sympify(1)
    for i in range(len(idx)):
        alpha, beta = coefs[i]
        poly *= g[idx[i]] ** alpha * (1 - g[idx[i]]) ** beta

    coeffs = poly.expand().as_coefficients_dict()
    coeffs = [(v, var_dict[k], col) for k, v in coeffs.items()]
    return coeffs


def alpha_beta(len_g, deg):
    for x in combinations(
            range(1, 2 * len_g + deg + 1), 2 * len_g):
        x = np.array(x)
        x[1:] -= x[:-1]
        x -= 1
        x = x.reshape(len_g, 2)
        colsums = x.sum(axis=1)
        nonzero = np.nonzero(colsums)[0]
        yield nonzero, x[nonzero]


if __name__ == '__main__':
    n = 200
    deg = 2
    g = sp.symbols('x_0:{}'.format(n))
    variables = sp.symbols('x_0:{}'.format(n))
    var_dict = {m: i for i, m in enumerate(sp.itermonomials(variables, deg))}
    h_poly_ = delayed(h_poly)
    multiproc = Parallel(n_jobs=-1, backend='multiprocessing', verbose=1)
    iter_ = enumerate(alpha_beta(len(g), deg))
    result = multiproc(
            h_poly_(col, idx, coefs) for col, (idx, coefs) in iter_)

    start_mat = time.process_time()
    result = list(chain.from_iterable(result))
    result = [x for x in zip(*result)]
    data, row, column = result
    data = np.array(data, dtype=float)
    row = np.array(row, dtype=int)
    column = np.array(column, dtype=int)
    a_mat = coo_matrix(
            (data, (row, column)), shape=(max(row) + 1, column[-1] + 1))
    mat_time = time.process_time() - start_mat
    print('building coo matrix time: ' + str(mat_time))

    start_csr = time.process_time()
    a_mat = a_mat.tocsr()
    csr_time = time.process_time() - start_csr
    print('building csr from coo matrix time: ' + str(csr_time))

#    print('*** SEQUENTIAL ***')
#    # start_seq = time.process_time()
#    a_mat = _certificate(
#        g, g, deg=2, n_jobs=1, backend='multiprocessing', verbose=1)
#    # seq_time = time.process_time() - start_seq
#    # print('sequential time ' + str(seq_time))
#
#    print('*** THREADING ***')
#    g = sp.symbols('x_0:{}'.format(n))
#    # start_mp = time.process_time()
#    a_mat = _certificate(
#        g, g, deg=2, n_jobs=-1, backend='threading', verbose=1)
#    # mp_time = time.process_time() - start_mp
#    # print('multiprocessing time ' + str(mp_time))
#
#    print('*** MULTIPROCESSING ***')
#    g = sp.symbols('x_0:{}'.format(n))
#    # start_mp = time.process_time()
#    a_mat = _certificate(
#        g, g, deg=2, n_jobs=-1, backend='threading', verbose=1)
#    # mp_time = time.process_time() - start_mp
#    # print('multiprocessing time ' + str(mp_time))
#

