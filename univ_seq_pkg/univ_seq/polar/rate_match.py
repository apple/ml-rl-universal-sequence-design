#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

# NR control channel processing
import numpy as np
from numpy import ceil, log2
from univ_seq.polar.sequences import get_nr_seq

# TS 38.212 - 5G NR Multiplexing and channel coding
# Bioglio, Valerio, Carlo Condo, and Ingmar Land. "Design of polar codes in 5G new radio." IEEE Communications Surveys & Tutorials 23.1 (2020): 29-40.
# Pandia Physical Layer NR Uplink System Design Document, Apple

# 1. Message Segmentation (UCI only)
# 2. Mother code length selection
# 3. CRC encoding
# 4. Input bits interleaving (PDCCH/PBCH only)
# 5. Subchannel allocation
# 6. Polar encoding
# 7. Sub-block interleaving
# 8. Rate matching
# 9. Channel interleaving (UCI only)


# section 5.3.1, TS 38.212
def get_n(K, E, DL=True):
    n_min = 5
    n_max = 9 if DL else 10
    cond = E <= (9/8)*2**(ceil(log2(E))-1) and K/E < 9/16
    n1 = ceil(log2(E))
    n1 = n1 - 1 if cond else n1
    n2 = ceil(log2(8*K))
    n = np.min([n1, n2, n_max])
    n = np.max([n, n_min])
    return int(n)


# section 5.1, TS 38.212
# CRC polynomial used by BCH, DCI, UCI
#         24  20  16  12   8   4   0
#          |   |   |   |   |   |   |
# bin      1101100101011000100010111
# hex      1   B   2   B   1   1   7

CRC24C_bin = '1101100101011000100010111'

def bin2list(bin_rep):
    crc = [int(c) for c in list(bin_rep)]
    return crc

# Table 5.3.1.1-1, TS 38.212
bit_intlv_seq = np.array([
   0,   2,   4,   7,   9,  14,  19,  20,  24,  25,  26,  28,
  31,  34,  42,  45,  49,  50,  51,  53,  54,  56,  58,  59,
  61,  62,  65,  66,  67,  69,  70,  71,  72,  76,  77,  81,
  82,  83,  87,  88,  89,  91,  93,  95,  98, 101, 104, 106,
 108, 110, 111, 113, 115, 118, 119, 120, 122, 123, 126, 127,
 129, 132, 134, 138, 139, 140,   1,   3,   5,   8,  10,  15,
  21,  27,  29,  32,  35,  43,  46,  52,  55,  57,  60,  63,
  68,  73,  78,  84,  90,  92,  94,  96,  99, 102, 105, 107,
 109, 112, 114, 116, 121, 124, 128, 130, 133, 135, 141,   6,
  11,  16,  22,  30,  33,  36,  44,  47,  64,  74,  79,  85,
  97, 100, 103, 117, 125, 131, 136, 142,  12,  17,  23,  37,
  48,  75,  80,  86, 137, 143,  13,  18,  38, 144,  39, 145,
  40, 146,  41, 147, 148, 149, 150, 151, 152, 153, 154, 155,
 156, 157, 158, 159, 160, 161, 162, 163,
 ])

# section 5.3.1.1
def get_ib_indices(K):
    K_max_IL = 164
    h = K_max_IL - K

    indices = np.zeros((K,), dtype=int)
    k = 0
    for m in range(K_max_IL):
        index_m = bit_intlv_seq[m]
        if index_m >= h:
            indices[k] = index_m - h
            k += 1
    return indices

def bit_interleave(in_seq, K):
    assert in_seq.size == K
    indices = get_ib_indices(K)
    out_seq = in_seq[indices]
    return out_seq

def bit_deinterleave(in_seq, K):
    assert in_seq.size == K
    out_seq = np.zeros((K,), dtype=int)
    indices = get_ib_indices(K)
    out_seq[indices] = in_seq
    return out_seq


# Table 5.4.1.1-1, TS 38.212
sub_block_intlv_seq = np.array([
  0,  1,  2,  4,  3,  5,  6,  7,
  8, 16,  9, 17, 10, 18, 11, 19,
 12, 20, 13, 21, 14, 22, 15, 23,
 24, 25, 26, 28, 27, 29, 30, 31,
 ])

# section 5.4.1.1
def get_sb_indices(N):
    blk_len = N//32
    P = sub_block_intlv_seq
    J = np.zeros((N,), dtype=int)
    for n in range(N):
        i = 32*n // N
        J[n] = P[i] * blk_len + (n % blk_len)
    return J

def sub_block_interleave(in_seq, N):
    assert in_seq.size == N
    indices = get_sb_indices(N)
    out_seq = in_seq[indices]
    return out_seq

def sub_block_deinterleave(in_seq, N):
    out_seq = np.zeros((N,))
    indices = get_sb_indices(N)
    out_seq[indices] = in_seq
    return out_seq

# section 5.4.1.2
def rate_match(in_seq, K, E, N, debug=False):
    # K = A + L = n_info + n_crc
    assert in_seq.size == N
    out_seq = np.zeros((E,), dtype=int)
    if E >= N: # repetition
        print('repetition') if debug else None
        n_exp = int(ceil(E/N))
        indices = np.tile(np.arange(N), n_exp)[:E]
        out_seq = in_seq[indices]
    else: # E < N
        if K/E <= 7/16: # puncturing
            print('puncturing') if debug else None
            U = N-E
            out_seq = in_seq[U:]
        else: # shortening
            print('shortening') if debug else None
            out_seq = in_seq[:E]
    return out_seq

def rate_recovery(in_seq, K, E, N):
    assert in_seq.size == E
    LLR_max = 1000
    out_seq = np.zeros((N,))
    if E >= N: # repetition
        n_exp = int(ceil(E/N))
        padded_seq = np.zeros((N*n_exp,))
        padded_seq[:E] = in_seq
        out_seq = padded_seq.reshape(-1,N).sum(axis=0)
    else: # E < N
        if K/E <= 7/16: # puncturing
            U = N-E
            out_seq[U:] = in_seq
        else:           # shortening
            out_seq[:E] = in_seq
            out_seq[E:] = LLR_max
    return out_seq

# channel interleaver (used in UCI only)
# section 5.4.1.3
def get_T(E):
    # smallest integer such that T(T+1)/2 >= E
    # closed form
    T = np.ceil((np.sqrt(8*E+1)-1)/2).astype(int)
    return T

def channel_interleave(in_seq, E):
    # max_T = 128
    assert E <= 8192
    T = get_T(E)
    out_seq = np.zeros((E,), dtype=int)
    v_matrix = np.zeros((T,T), dtype=int)

    # fill
    k = 0
    for i in range(T):
        for j in range(T-i):
            if k < E:
                v_matrix[i,j] = in_seq[k]
            else:
                v_matrix[i,j] = -1
            k += 1

#    print(v_matrix)
    # read
    k = 0
    for j in range(T):
        for i in range(T-j):
            if v_matrix[i,j] != -1:
                out_seq[k] = v_matrix[i,j]
                k += 1

    return out_seq

# int num_zeros_accum[] =
# [   0,    1,    3,    6,   10,   15,   21,   28,   36,   45,   55,
#    66,   78,   91,  105,  120,  136,  153,  171,  190,  210,  231,
#   253,  276,  300,  325,  351,  378,  406,  435,  465,  496,  528,
#   561,  595,  630,  666,  703,  741,  780,  820,  861,  903,  946,
#   990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485,
#  1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
#  2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926,
#  3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828,
#  3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560, 4656, 4753, 4851,
#  4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995,
#  6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260,
#  7381, 7503, 7626, 7750, 7875, 8001, 8128];

def channel_deinterleave(in_seq, E):
    assert E <= 8192
    T = get_T(E)
    out_seq = np.zeros((E,), dtype=int)
    v_matrix = np.zeros((T,T), dtype=int)

    num_zeros = np.arange(T)
    num_zeros_accum = np.cumsum(num_zeros)

    # fill
    k = 0
    for j in range(T):
        for i in range(T-j):
            index = i*T + j
            index = index - num_zeros_accum[i-1] if i > 0 else index
            if index < E:
                v_matrix[i,j] = in_seq[k]
                k += 1
            else:
                v_matrix[i,j] = -1

#    print(v_matrix)
    # read
    k = 0
    for i in range(T):
        for j in range(T-i):
            if k < E:
                out_seq[k] = v_matrix[i,j]
            k += 1

    return out_seq



# Section 5.4.1.1, TS 38.212
def get_info_frozen_set(K, E, N):

    Q_all = get_nr_seq()
    Q_N = Q_all[Q_all < N]
#    print('Q_N', Q_N)

    Q_F_tmp = None
    if E < N:
        J = get_sb_indices(N)
        if K/E <= 7/16: # puncturing
            print('puncturing')
            U = N-E
            Q_F_tmp = J[:U]
            print('Q_F_tmp', Q_F_tmp.size, Q_F_tmp)
            if E >= 3*N/4:
                T = ceil(3*N/4 - E/2)
            else:
                T = ceil(9*N/16 - E/4)
            T_set = np.arange(int(T))
            Q_F_tmp = np.union1d(Q_F_tmp, T_set)
            print('Q_F_tmp', Q_F_tmp.size, Q_F_tmp)
        else:           # shortening
            print('shortening')
            Q_F_tmp = J[E:]
            print('Q_F_tmp', Q_F_tmp.size, Q_F_tmp)
    # set assume_unique=True to prevent sorting of Q_N
    Q_I_tmp = np.setdiff1d(Q_N, Q_F_tmp, assume_unique=True)
#    print('Q_I_tmp', Q_I_tmp)

    Q_I = Q_I_tmp[-K:] # K most reliable bit indices
#    print('Q_I_unsorted', Q_I)
    Q_I = np.sort(Q_I)
    Q_F = np.setdiff1d(Q_N, Q_I)
    # frozen_bitmap
    frozen_bitmap = np.zeros((N,), dtype=int)
    frozen_bitmap[Q_F] = 1
#    print('frozen_bitmap', frozen_bitmap.tolist())
    return Q_I, Q_F

if __name__ == "__main__":

    K = 300
    E = 600
#    K = 80
#    E = 160
#    K = 200
#    E = 400
    DL = False
    n = get_n(K, E, DL)
    N = 2**n
    print(f'K {K} E {E} N {N}')

    in_vec = np.arange(N)
    debug = True
    sb_vec = sub_block_interleave(in_vec, N)
    rm_vec = rate_match(sb_vec, K, E, N, debug)
    print(np.sort(rm_vec)[::-1])

    # plot shortened bits
#    import matplotlib.pyplot as plt
#    x = np.arange(N)
#    y = np.zeros((N,))
#    y[rm_vec] = 1
#    plt.stem(x,y)
#    plt.grid()
#    plt.show()

    Q_I, Q_F = get_info_frozen_set(K, E, N)
    print('info set')
    print(np.sort(Q_I)[::-1])

    exit()

    # channel interleaver
    E = 23
#    E = 28
    in_vec = np.arange(E)
    out_vec = channel_interleave(in_vec, E)
    print('in_vec', in_vec)
    print('out_vec', out_vec)
    de_vec = channel_deinterleave(out_vec, E)
    print('de_vec', de_vec)

    exit()

    AL_vec = np.array([1,2,4,8,16])
    E_vec = 108*AL_vec

    n_vec = []
    K = 40
    for E in E_vec:
        n_vec.append( get_n(K, E) )
    print(n_vec)

    print(bin2list(CRC24C_bin))

    in_vec = np.arange(K)
    intlv_vec = bit_interleave(in_vec, K)
    print(intlv_vec)
    dintlv_vec = bit_deinterleave(intlv_vec, K)
    print(dintlv_vec)

    N = 64
    in_vec = np.arange(N)
    sb_intlv_vec = sub_block_interleave(in_vec, N)
    print(sb_intlv_vec)
    sb_dintlv_vec = sub_block_deinterleave(sb_intlv_vec, N)
    print(sb_dintlv_vec.astype(int))

    print('RM')

    A = 12; E = 432
    K = A + 24
    n = get_n(K, E)
    N = 2**n
    print('N', N)

    exit()


    N = 64; K = 20; E = 70
    in_vec = np.arange(N)
    rm_vec = rate_match(in_vec, K, E, N)
    print(rm_vec)

    N = 64; K = 20; E = 60
    in_vec = np.arange(N)
    rm_vec = rate_match(in_vec, K, E, N)
    print(rm_vec)

    N = 64; K = 40; E = 60
    in_vec = np.arange(N)
    rm_vec = rate_match(in_vec, K, E, N)
    print(rm_vec)

    N = 64; K = 20; E = 70
    in_vec = np.ones((E,))
    rr_vec = rate_recovery(in_vec, K, E, N)
    print(rr_vec)

    N = 64; K = 20; E = 60
    in_vec = np.ones((E,))
    rr_vec = rate_recovery(in_vec, K, E, N)
    print(rr_vec)

    N = 64; K = 40; E = 60
    in_vec = np.ones((E,))
    rr_vec = rate_recovery(in_vec, K, E, N)
    print(rr_vec)

    N = 64; K = 18; E = 44
    Q_I, Q_F = get_info_frozen_set(K, E, N)
    print('Q_I', Q_I)
    print('Q_F', Q_F)

    # test whole encoding chain
    A = 12; K = 18; E = 44
    N = 2**get_n(K, E)
    Q_I, Q_F = get_info_frozen_set(K, E, N)
    info_crc_vec = np.ones((K,), dtype=int)
    info_carrier = np.zeros((N,), dtype=int)
    info_carrier[Q_I] = info_crc_vec
    print('info_carrier', np.sum(info_carrier), info_carrier)
    encoded_vec = info_carrier # hack
    sb_intlv_vec = sub_block_interleave(encoded_vec, N)
    print('sb_intlv_vec', np.sum(sb_intlv_vec), sb_intlv_vec)
    rm_vec = rate_match(sb_intlv_vec, K, E, N)
    print('rm_vec', rm_vec.size, np.sum(rm_vec), rm_vec)

    N = 128; K = 44; E = 108
    print(f'N = {N}, K = {K}, E = {E}')
    Q_I, Q_F = get_info_frozen_set(K, E, N)
    print('Q_I', Q_I)
    print('Q_F', Q_F)

