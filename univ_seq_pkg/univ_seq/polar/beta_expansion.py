#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# beta expansion
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

#import warnings
#warnings.filterwarnings("error", category=RuntimeWarning)
#import traceback

def bin2int(in_vec, n):
    axis = in_vec.ndim - 1
    exp_vec = np.arange(n)[::-1]
    return np.sum(in_vec * 2**exp_vec, axis=axis)

def int2bin(in_vec, n):
    # n: bitwidth
    bin_mat = ((in_vec.reshape(-1, 1) & (2 ** np.arange(n))) != 0).astype(int)
    bin_mat = bin_mat[:, ::-1]
    return bin_mat

def get_beta_exp(N):
    n = np.log2(N).astype(int)
    indices = np.arange(N)
    bin_mat = int2bin(indices,n)

    beta = 1.1892
    pvec = np.arange(n)[::-1]
    beta_pvec = np.power(beta, pvec)
    beta_exp_mat = bin_mat * beta_pvec
    beta_exp_vec = np.sum(beta_exp_mat, 1)
    beta_seq = np.argsort(beta_exp_vec)
    return beta_seq

if __name__ == "__main__":
#    import argparse
#    parser = argparse.ArgumentParser(description="")
#    parser.add_argument('N', type=int, help='')
    #parser.add_argument("--option", help="Optional argument", default="default_value")

#    args = parser.parse_args()

    def filter_c_vec(c_vec, n):
        if np.sum(c_vec == 1) == 0 or np.sum(c_vec == -1) == 0:
            return False
        if np.sum(c_vec) == 0:
            return False
        return True

    def filter_roots(roots):
        idx = np.logical_and( roots.imag == 0, roots.real > (1.0+1e-9) )
        return roots[idx]

    print(f'2^(1/4) = {2**(1/4):.4f}')
    beta0 = 2**(1/4)

    n_vec = [3,4,5,6,7,8,9,10,11]

    for n in n_vec:
        N = 2**n
        indices = np.arange(N)
        bin_mat = int2bin(indices, n)
        c_list = []
        for i in range(2, N):
            for j in range(i+1, N):
                c_vec = bin_mat[j] - bin_mat[i]
                c_list.append(c_vec)
        c_mat_init = np.array(c_list)
        c_mat_uniq = np.unique(c_mat_init, axis=0)
        c_mat_filt = np.array([c_vec for c_vec in c_mat_uniq if filter_c_vec(c_vec, n)])
        c_mat = c_mat_filt

        x_vec = []
        for c_vec in c_mat:
#            print('c_vec', c_vec)
            roots = np.roots(c_vec)
#            print('roots', roots)
            roots = filter_roots(roots)
#            print('roots_filtered', roots)
            x = np.real(roots)
            if x.size == 0:
                x = np.nan
                x_vec.append(x)
            elif x.size == 1:
                x = x.item()
                x_vec.append(x)
            else:
                # more than 2 roots
                for x_ in x:
                    x_vec.append(x_)

        x_vec = np.array(x_vec)

#        indices = np.argsort(x_vec)
#        print(f'n = {n}, N = {N}')
#        for i in indices:
#            print(f'x = {x_vec[i]:.3f}, c = {c_mat[i]}')

        print(f'n = {n}, N = {N}')
        x_vec = x_vec[~np.isnan(x_vec)]
        x_vec = np.unique(x_vec)
        x_vec = np.insert(x_vec, 0, 1)
        # print beta0 neighborhood
        if x_vec.size > 6:
            hi = np.where(x_vec > beta0)[0][:3]
            lo = np.where(x_vec < beta0)[0][-3:]
            indices = np.concatenate([lo, hi])
            x_vec = x_vec[indices]
        for x in x_vec:
            if n >= 8:
                print(f'x = {x:.5f}')
            else:
                print(f'x = {x:.3f}')

        print(f'n = {n}, N = {N}')
        # get interval
        hi = np.where(x_vec > beta0)[0][0]
        lo = np.where(x_vec < beta0)[0][-1]
        x_intv = np.array((x_vec[lo], x_vec[hi]))
        print(f'beta \in ({x_intv[0]:.5f}, {x_intv[1]:.5f})')

    exit()

    ################################################################################
    # old stuff
    ################################################################################

    c_table = []

    # (beta+1, beta^2)
    c_list = []
    c_list.append([1, -1, -1])
    c_mat = np.array(c_list)
    c_table.append(c_mat)

    # (beta+1, beta^2)
    # (beta^2+beta, beta^3)
    # (beta+1, beta^3)
    # (beta^2+1, beta^3)
    # (beta^2+beta+1, beta^3)
    c_list = []
    c_list.append([0,  1, -1, -1])
    c_list.append([1, -1, -1, -1])
    c_list.append([1, -1,  0, -1])
    c_list.append([1,  0, -1, -1])
    c_mat = np.array(c_list)
    c_table.append(c_mat)


