#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import time
import numpy as np
from numpy.random import randint
import polar
from univ_seq.polar.sequences import get_nr_seq
from univ_seq.polar.rate_match import get_n

#from sequences import get_nr_seq

#N = 64
#E = 44
#A = 12
#N = 128
#E = 108*1
#A = 40 # 20
#N = 512
#E = 108*8
#A = 100
#A = 430
#E = 574
#A = 70
#E = 560
#A = 340
#E = 510

Q = 2
A = 200
#R = 1/4; snr_db = -1.5
#R = 1/2; snr_db = 3.0
#R = 2/3; snr_db = 3.0
A = 100; R = 1/4; snr_db = 30.0
A = 200; R = 1/2; snr_db = 30.0
A = 200; R = 1/2; snr_db = 3.0
A = 300; R = 1/2; snr_db = 30.0
A = 300; R = 1/2; snr_db = 2.5
#A = 80; R = 1/2; snr_db = 30.0
#A = 400; R = 2/3; snr_db = 30.0

def get_E(A,R):
    return (np.ceil(A/(Q*R)) * Q).astype(int)
E = get_E(A,R)
print(f'R {R:.3f} A {A} E {E}')

pac_code = True

L = 8
max_err = 1
max_runs = 1
max_err = 10
max_runs = 10
max_err = 10000
max_runs = 10000
#crc_coeffs = np.array([])
#crc_coeffs = np.array([1,0,0,1,1])
#crc_coeffs = np.array([1,1,0,0,0,0,1]) # CRC6
#crc_coeffs = np.array([1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1]) # CRC24C
crc_coeffs = np.array([1,1,1,0,0,0,1,0,0,0,0,1]) # CRC11 0xE21
crc_coeffs = np.array([])


if pac_code:
    print('PAC code')
    conv_coeffs = np.array([1,0,1,1,0,1,1]) # octal 133
else:
    print('Polar code')
    conv_coeffs = np.array([])

genie_aided = False
channel_interleave = True
downlink = False

n_crc = crc_coeffs.size - 1 if crc_coeffs.size > 1 else 0
K = A + n_crc
n = get_n(K, E, downlink)
N = 2**n

if E >= N:
    print('Repetition')
elif K/E <= 7/16:
    print('Puncturing')
else:
    print('Shortening')


print(f'A {A} K {K} E {E} N {N}')

nr_seq = get_nr_seq()
sequence = nr_seq[nr_seq < N]

print('seq')
print(sequence)

#N = 128
#K = 76
#indices = sequence[-K:]
#info_bitmap = np.zeros((N,),dtype=int)
#info_bitmap[indices] = 1
#frozen_bits = 1 - info_bitmap

#K = 64
#frozen_bits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

#K = 44
#frozen_bits = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#print('frozen_bits', frozen_bits)
#
##polar_sim = polar.PolarSimulator()
#polar_sim = polar.PolarSimulator(1)
#start = time.perf_counter()
#bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, False)
#end = time.perf_counter()
#print('bler',bler)
#print(f'time {end-start:.2f} secs')
#
#exit()

#polar_sim = polar.PolarSimulator()
polar_sim = polar.PolarSimulator(1)
start = time.perf_counter()
bler = polar_sim.get_bler_rm(A, E, L, snr_db, max_err, max_runs, sequence, crc_coeffs, genie_aided, channel_interleave, downlink, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

exit()

polar_sim = polar.MultiThreadPolarSimulator(2,1000,8) # seed, runs, num_threads
start = time.perf_counter()
bler = polar_sim.get_bler_rm(A, E, L, snr_db, max_err, max_runs, sequence, crc_coeffs, genie_aided, channel_interleave, downlink, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')
