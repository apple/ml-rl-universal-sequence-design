#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# implement DE/GA algorithm from
# [1] Wu, Daolong, Ying Li, and Yue Sun. "Construction and block error rate analysis of polar codes over AWGN channel based on Gaussian approximation." IEEE Communications Letters 18.7 (2014): 1099-1102.
# [2] Trifonov, Peter. "Efficient design and decoding of polar codes." IEEE transactions on communications 60.11 (2012): 3221-3227.
# [3] Chung, Sae-Young, Thomas J. Richardson, and Rüdiger L. Urbanke. "Analysis of sum-product decoding of low-density parity-check codes using a Gaussian approximation." IEEE Transactions on Information theory 47.2 (2001): 657-670.
# [4] Dai, Jincheng, Kai Niu, Zhongwei Si, Chao Dong, and Jiaru Lin. "Does Gaussian approximation work well for the long-length polar code construction?." IEEE Access 5 (2017): 7950-7963.
import numpy as np
from scipy.special import erfc
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

from sequences import get_beta_exp, get_beta_exp_seq, get_nr_seq


#import warnings
#warnings.filterwarnings("error", category=RuntimeWarning)
#import traceback

################################################################################
# helper functions
################################################################################
import time
import functools

def timer(func):
    """A decorator that prints the execution time of the function it decorates."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # More precise than time.time()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Time {run_time:.4f} secs")
        return value
    return wrapper_timer

def get_capacity_limit(K,N):
    # compute target SNR
    R = K/N
    capacityLimitdB = 10*np.log10(2**R - 1)
    SNRdB = capacityLimitdB
    return SNRdB

def qfunc(x):
    return 0.5 * erfc( x / np.sqrt(2))

# eqn (9) in [1] , defn 1 in [3]
def _phi_fn_exact(x):
    if x == 0: return 1
    def integrand(u, x):
        return np.tanh( u/2 ) * np.exp( -(u-x)**2/(4*x) )
    def integral(x):
        return quad(integrand, -np.inf, np.inf, args=(x))[0]
    out = 1 - ( 1 / np.sqrt(4*np.pi*x) * integral(x) )
    return out

def _phi_fn_approx2(x):
    if x == 0: return 1
    if x < 10:
        def integrand(u, x):
            return np.tanh( u/2 ) * np.exp( -(u-x)**2/(4*x) )
        def integral(x):
            return quad(integrand, -np.inf, np.inf, args=(x))[0]
        out = 1 - ( 1 / np.sqrt(4*np.pi*x) * integral(x) )
        return out
    else:
        return np.sqrt(np.pi/x) * np.exp(-x/4) * ( 1 - 10/(7*x) )

# eqn (11) in [1]
def _phi_fn_approx(x):
    if x == 0: return 1
    if x < 10:
        return np.exp(-0.4527*(x**0.86) + 0.0218)
    else:
        return np.sqrt(np.pi/x) * np.exp(-x/4) * ( 1 - 10/(7*x) )

# AGA-2 in [4]
def _phi_fn_aga2(x):
    if x == 0: return 1
    if x > 7.0633:
        out = np.exp(-0.2944*x - 0.3169)
    else:
        out = np.exp(0.0116*x**2 - 0.4212*x)
    return out

# AGA-3 in [4]
def _phi_fn_aga3(x):
    if x == 0: return 1
    if x > 9.2254:
        out = np.exp(-0.2832*x - 0.4254)
    elif x > 0.6357:
        out = np.exp(-0.4527*x**0.86 + 0.0218)
    else:
        out = np.exp(0.06725*x**2 - 0.4908*x)
    return out

# AGA-4 in [4]
def _phi_fn_aga4(x):
    if x == 0: return 1
    if x > 9.2254:
        out = np.exp(-0.2832*x - 0.4254)
    elif x > 0.7420:
        out = np.exp(-0.4527*x**0.86 + 0.0218)
    elif x > 0.1910:
        out = 0.9981*np.exp(0.05315*x**2 - 0.4795*x)
    else:
        out = np.exp(0.1047*x**2 - 0.4992*x)
    return out

#def _inv_phi_fn(y):
#    # find the root of f(x) - y = 0
#    def target(x, y):
#        return _phi_fn(x) - y
#    x0=0.01
#    return fsolve(target, x0, args=(y))[0]

def _inv_phi_fn_aga4(y):
    if y == 1: return 0
    a = 0.1910
    b = 0.7420
    c = 9.2254
    if y < np.exp(-0.4527*c**0.86 + 0.0218):
        c1 = -0.2832
        c0 = -0.4254 - np.log(y)
        coeffs = [c1, c0]
        roots = np.roots(coeffs)
        return roots.item()
    elif y < 0.9981*np.exp(0.05315*b**2 - 0.4795*b):
        c1 = -0.4527
        c0 = 0.0218
        alpha = 0.86
        r = np.log(y)
        return ((r - c0)/c1)**(1/alpha)
    elif y < np.exp(0.1047*a**2 - 0.4992*a):
        c2 =  0.05315
        c1 = -0.4795
        gamma = 0.9981
        c0 = -np.log(y/gamma)
        coeffs = [c2, c1, c0]
        roots = np.roots(coeffs)
        print(f'a {a} b {b} roots {roots}')
        return roots[np.logical_and(a < roots, roots <= b)].item()
    else:
        c2 =  0.1047
        c1 = -0.4992
        c0 = -np.log(y)
        coeffs = [c2, c1, c0]
        roots = np.roots(coeffs)
        return roots[roots <= a].item()

def _inv_phi_fn_aga3(y):
    if y == 1: return 0
    a = 0.6357
    b = 9.2254
    if y < np.exp(-0.4527*b**0.86 + 0.0218):
        c1 = -0.2832
        c0 = -0.4254 - np.log(y)
        coeffs = [c1, c0]
        roots = np.roots(coeffs)
        return roots.item()
    elif y < np.exp(0.06725*a**2 - 0.4908*a):
        c1 = -0.4527
        c0 = 0.0218
        alpha = 0.86
        r = np.log(y)
        return ((r - c0)/c1)**(1/alpha)
    else:
        c2 =  0.06725
        c1 = -0.4908
        c0 = -np.log(y)
        coeffs = [c2, c1, c0]
        roots = np.roots(coeffs)
        return roots[roots <= a].item()

def _inv_phi_fn_aga2(y):
    if y == 1: return 0
    a = 7.0633
    if y < np.exp(0.0116*a**2 - 0.4212*a):
        c1 = -0.2944
        c0 = -0.3169 - np.log(y)
        coeffs = [c1, c0]
        roots = np.roots(coeffs)
        return roots.item()
    else:
        c2 =  0.0116
        c1 = -0.4212
        c0 = -np.log(y)
        coeffs = [c2, c1, c0]
        roots = np.roots(coeffs)
        return roots[roots <= a].item()

def _inv_phi_fn_v1(y):
    # minimize |f(x) - y|^2
    # subject to x >= 0
    def target(x, y):
        return (_phi_fn(x) - y)**2
    x0=0.01
    tol=1e-12
    constr = LinearConstraint(1, lb=0)
    method = 'trust-constr'
    return minimize(target, x0, args=(y), constraints=(constr), method=method, tol=tol).x.item()

def _inv_phi_fn_v2(y):
    # search in log domain
    # minimize |f(x) - y|^2
    # subject to x >= 0
    def target(x, y):
        return (np.log(_phi_fn(x)) - np.log(y))**2
    x0=0.01
    tol=1e-6
    constr = LinearConstraint(1, lb=0)
    method = 'trust-constr'
    method = None
    x_hat = minimize(target, x0, args=(y), constraints=(constr), method=method, tol=tol).x
    return np.exp(x_hat)

def _inv_phi_fn_v3(y):
    # minimize |f(x) - y|
    # subject to x >= 0
    def target(x, y):
        return abs(_phi_fn(x) - y)
    x0=0.01
    tol=1e-12
    constr = LinearConstraint(1, lb=0)
    method = 'trust-constr'
    return minimize(target, x0, args=(y), constraints=(constr), method=method, tol=tol).x

def _fc_fn_vec(t):
    if t.size > 100:
        out = inv_phi_fn_mp( 1 - ( 1 - phi_fn_mp(t) )**2 )
    else:
        out = inv_phi_fn( 1 - (1 - phi_fn(t) )**2 )
    return out

def _fc_fn_default(t):
    return _inv_phi_fn( 1 - ( 1 - _phi_fn(t) )**2 )

def _fc_fn_aga2(t):
    tau = 9.4177
    if t > tau:
        return t - 2.3544
    else:
        return _inv_phi_fn_aga2( 1 - ( 1 - _phi_fn_aga2(t) )**2 )

def _fc_fn_aga3(t):
    tau = 11.673
    if t > tau:
        return t - 2.4476
    else:
        return _inv_phi_fn_aga3( 1 - ( 1 - _phi_fn_aga3(t) )**2 )

def _fc_fn_aga4(t):
    tau = 11.673
    if t > tau:
        return t - 2.4476
    else:
        return _inv_phi_fn_aga4( 1 - ( 1 - _phi_fn_aga4(t) )**2 )

################################################################################
# function selection
################################################################################

#_phi_fn = _phi_fn_approx
#_phi_fn = _phi_fn_exact
_phi_fn = _phi_fn_approx2

_inv_phi_fn = _inv_phi_fn_v1

_fc_fn = _fc_fn_vec

################################################################################
# phi and inv_phi vectorized functions
################################################################################
def phi_fn(x):
    vec_phi_fn = np.vectorize(_phi_fn)
    out = np.ones_like(x)
    ind = x > 0
    out[ind] = vec_phi_fn(x[ind])
    return out

#@timer
def inv_phi_fn(y):
    vec_inverse = np.vectorize(_inv_phi_fn)
    out = np.zeros_like(y)
    ind = y < 1
    out[ind] = vec_inverse(y[ind])
    return out

from multiprocessing import Pool
num_cpus = 8
def phi_fn_mp(x):
    out = np.ones_like(x)
    ind = x > 0
    with Pool(num_cpus) as pool:
        out[ind] = np.array(pool.map(_phi_fn, x[ind]))
    return out

#@timer
def inv_phi_fn_mp(y):
    out = np.zeros_like(y)
    ind = y < 1
    with Pool(num_cpus) as pool:
        out[ind] = np.array(pool.map(_inv_phi_fn, y[ind]))
    return out

# vectorize
if 'vec' in _fc_fn.__name__:
    fc_fn = _fc_fn
else:
    fc_fn = np.vectorize(_fc_fn)

################################################################################
# main DE/GA algorithm
################################################################################
def get_DE_GA_weights(snr_db, N):
    '''
    see eqns (9) and (10) in [1]
        eqns (5) and  (6) in [2] <-- typo
        eqns (6) - (9) in [4]

    {1,-1} BPSK signal in AWGN noise with variance sigma^2
    LLR distribution is normal with
    L ~ N( 2/sigma^2, 4/sigma^2 )
    '''
    n = int(np.log2(N))
    n_var = 10 ** ( - snr_db / 10 )

    llr_mean = np.array(2/n_var)
    #llr_var  = 4/n_var
    llr_mean_vec = []
    llr_mean_vec.append(llr_mean)

    for n_i in range(1,n+1):
        N_i = 2**n_i
        j_vec = np.arange(N_i//2)
        i_odd  = 2*j_vec
        i_even = 2*j_vec + 1
#        print('n_i', n_i, 'N_i', N_i)
        llr_mean = llr_mean_vec[n_i - 1]
        llr_mean_new = np.zeros((N_i,))
#        print('llr_mean', llr_mean)
#        inv_phi_args = 1 - ( 1 - phi_fn(llr_mean) )**2
#        print('inv_phi_args', inv_phi_args)
        llr_mean_new[i_odd]  = fc_fn(llr_mean)
        llr_mean_new[i_even] = 2 * llr_mean
        llr_mean_vec.append(llr_mean_new)

    return llr_mean_vec[-1]

def get_DE_GA_sequence(snr_db, N, verbose=False):
    llr_mean = get_DE_GA_weights(snr_db, N)
    print('llr_mean', llr_mean.tolist()) if verbose else None
    seq = np.argsort(llr_mean)
    return seq

def get_snr_adjusted_DE_GA_sequence(N, snr_offset_db, verbose=False):
    print(f'offset {snr_offset_db} dB')
    seq = []
    for k in range(1,N+1): # 1 to N
        print(f'k {k} of N {N}')
        # get close to the operating SNR
        snr_db = get_capacity_limit(k,N) + snr_offset_db
        seq_fixed_snr = get_DE_GA_sequence(snr_db, N, verbose)
        index = seq_fixed_snr[-k]
        # check consistency
        if index in seq:
            print(f'index {index} already in seq!!!')
            n = (k-1) # check last k-1
            print('seq1', seq)
            print('seq2', seq_fixed_snr[-n:].tolist())
            for i in range(1,5):
                s_index = k-i
                new_index = seq_fixed_snr[-s_index]
                if not new_index in seq:
                    print(f'found new index {new_index} at offset {i}')
                    index = new_index
                    break
        seq.insert(0, index)
    seq = np.array(seq)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="")
    #parser.add_argument('N', type=int, help='')
    parser.add_argument("--N", type=int, help="", default=128)
    #parser.add_argument("--option", help="Optional argument", default="default_value")

    args = parser.parse_args()

    N = int(args.N)
    one_shot = False
    test = False
    full_sweep = True

    snr_db_cr1_3 = get_capacity_limit(1,3) + 6.0
    print(f'snr_db @ CR 1/3 {snr_db_cr1_3:.2f}')

    if one_shot:
        snr_db = 0
        snr_db = -3.5
        print(f'N {N} snr_db {snr_db:.2f}')
        seq = get_DE_GA_sequence(snr_db, N)
        print('DEGA_seq', seq.tolist())

        beta_seq = get_beta_exp(N)
        print('beta_seq', beta_seq.tolist())

    if test:
        snr_offset_db = 6.0
        snr_db = get_capacity_limit(63,64) + snr_offset_db
        print(f'N {N} snr_db {snr_db:.2f} offset {snr_offset_db}')
#        seq = get_DE_GA_sequence(snr_db, N)
#        print('DEGA_seq', seq)

        beta_seq = get_beta_exp(N)
#        beta_seq = get_beta_exp_seq()
#        beta_seq = beta_seq[beta_seq < N]
        print('beta_seq', beta_seq.tolist())
        nr_seq = get_nr_seq()
        nr_seq = nr_seq[nr_seq < N]
        print('nr_seq', nr_seq.tolist())

    if full_sweep:
#        snr_offset_db = 6.0
#        offsets = [6.0, 5.0, 4.0]
        offsets = [7.0, 6.0, 5.0]
        for snr_offset_db in offsets:
            seq = get_snr_adjusted_DE_GA_sequence(N, snr_offset_db)
            print(f'offset {snr_offset_db} dB')
            print('DEGA_seq', seq.tolist())
#            beta_seq = get_beta_exp(N)
#            print('beta_seq', beta_seq.tolist())

    exit()

    import matplotlib.pyplot as plt
    x = np.linspace(0.0, 20.0, 1000)
#    x = np.linspace(0.0, 10.0, 1000)
    fig, ax = plt.subplots()
    ax.semilogy(x, phi_fn(x), label='phi')
    ax.grid(True, which='both')
    # save file
    pFile = f'plot.pdf'
    fig.savefig(pFile)
    print('saving plot to:', pFile)

    x = np.linspace(0.0, 10.0, 11)
    u = 1
    out = phi_fn(x)
    print(x)
    print(out)
    inv_out = inv_phi_fn(out)
    print(inv_out)


'''
snr_db @ CR 1/3 0.15 dB

N 16 snr_db 0.00
llr_mean [2.48735913e-04 3.39764960e-02 5.93752817e-02 8.04135103e-01
 1.21660889e-01 1.21818218e+00 1.71980587e+00 6.57873768e+00
 2.76057120e-01 1.99653535e+00 2.72114131e+00 9.09516031e+00
 3.78550384e+00 1.15800441e+01 1.35078436e+01 3.20000000e+01]
DEGA_seq [ 0  1  2  4  8  3  5  6  9 10 12  7 11 13 14 15]
beta_seq [ 0  1  2  4  8  3  5  6  9 10 12  7 11 13 14 15]

N 32 snr_db 0.00
llr_mean [1.65405073e-04 4.97471825e-04 5.95267768e-04 6.79529921e-02
 1.66819386e-03 1.18750563e-01 1.93733741e-01 1.60827021e+00
 6.62734755e-03 2.43321778e-01 3.80129012e-01 2.43636436e+00
 6.52624166e-01 3.43961174e+00 4.48787422e+00 1.31574754e+01
 3.03513284e-02 5.52114241e-01 8.20176989e-01 3.99307071e+00
 1.30407872e+00 5.44228263e+00 6.81655119e+00 1.81903206e+01
 2.10311164e+00 7.57100769e+00 9.09717635e+00 2.31600881e+01
 1.10587540e+01 2.70156872e+01 2.93816546e+01 6.40000000e+01]
DEGA_seq [ 0  1  2  4  8 16  3  5  6  9 10 17 12 18 20  7 24 11 13 19 14 21 22 25
 26 28 15 23 27 29 30 31]
beta_seq [ 0  1  2  4  8 16  3  5  6  9 10 17 12 18 20  7 24 11 13 19 14 21 22 25
 26 28 15 23 27 29 30 31]

N 64 snr_db 0.00
llr_mean [1.65396448e-04 3.30810146e-04 1.65451462e-04 9.94943651e-04
 1.65478168e-04 1.19053554e-03 2.16676229e-03 1.35905984e-01
 1.66085427e-04 3.33638773e-03 6.32970968e-03 2.37501127e-01
 1.58641975e-02 3.87467482e-01 5.88309536e-01 3.21654041e+00
 1.76597953e-04 1.32546951e-02 2.41272533e-02 4.86643557e-01
 5.37860017e-02 7.60258024e-01 1.10693965e+00 4.87272872e+00
 1.36725328e-01 1.30524833e+00 1.83406625e+00 6.87922348e+00
 2.67161822e+00 8.97574843e+00 1.07154944e+01 2.63149507e+01
 4.93466944e-04 6.07026568e-02 1.02919952e-01 1.10422848e+00
 2.00160850e-01 1.64035398e+00 2.26823349e+00 7.98614141e+00
 4.23547098e-01 2.60815743e+00 3.48177188e+00 1.08845653e+01
 4.70282658e+00 1.36331024e+01 1.56690253e+01 3.63806412e+01
 8.87498345e-01 4.20622328e+00 5.39287130e+00 1.51420154e+01
 6.81845541e+00 1.81943527e+01 2.05917606e+01 4.63201762e+01
 8.59777645e+00 2.21175080e+01 2.44219364e+01 5.40313743e+01
 2.67753121e+01 5.87633092e+01 4.86480931e+01 1.28000000e+02]
DEGA_seq [ 0  2  4  8 16  1 32  3  5  6  9 10 17 12 18 20 33 34  7 24 36 11 13 40
 19 14 21 48 35 22 25 37 26 38 41 28 15 42 49 44 23 50 52 27 39 56 29 30
 43 45 51 46 53 54 57 58 31 60 47 55 62 59 61 63]
beta_seq [ 0  1  2  4  8 16  3 32  5  6  9 10 17 12 18 33 20 34  7 24 36 11 40 13
 19 14 48 21 35 22 25 37 26 38 41 28 42 15 49 44 50 23 52 27 39 56 29 43
 30 45 51 46 53 54 57 58 31 60 47 55 59 61 62 63]

'''

