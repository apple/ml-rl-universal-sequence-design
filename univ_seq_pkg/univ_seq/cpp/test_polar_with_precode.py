#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import time
import numpy as np
from numpy.random import randint
import polar
import pac

from univ_seq.polar.sequences import get_nr_seq
from univ_seq.polar.rate_match import get_n


N = 64
L = 8
snr_db = 2.0
max_err = 100000
max_runs = 100000
#crc_coeffs = np.array([1,0,0,1,1])
#crc_coeffs = np.array([1,1,1,0,0,0,1,0,0,0,0,1]) # CRC11 0xE21
#crc_coeffs = np.array([1,1,0,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,1,1]) # CRC24C
crc_coeffs = np.array([])
conv_coeffs = np.array([1,0,1,1,0,1,1]) # octal 133
#conv_coeffs = np.array([])
genie_aided = False
#frozen_bits = randint(2, size=N)
frozen_bits = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#N = 1024
#snr_db = -5.0
#fbm = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
#frozen_bits = 1 - fbm
#print('len', fbm.size)

print('using NR seq')
nr_seq = get_nr_seq()
N = 256
K = 80
snr_db = 0.0
N = 512
K = 200
snr_db = 2.0
seq_N = nr_seq[nr_seq < N]
indices = seq_N[-K:]
fbm = np.zeros((N,), dtype=int)
fbm[indices] = 1
frozen_bits = 1 - fbm
print(f'N {N} K {K}')
print('fbm\n', fbm)


if crc_coeffs.size == 0:
    print('NO CRC!')
else:
    print('crc coeffs', crc_coeffs)
if conv_coeffs.size == 0:
    print('polar mode (no precoding')
else:
    print('PAC mode (with precoding)')
    print('conv coeffs', conv_coeffs)

print(f'Polar SIM')
polar_sim = polar.PolarSimulator(1)
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, genie_aided, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

polar_sim = polar.MultiThreadPolarSimulator(2,1000,8) # seed, runs, num_threads
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, genie_aided, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

print(f'PAC SIM')
pac_sim = pac.PacSimulator(1)
start = time.perf_counter()
bler = pac_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

pac_sim = pac.MultiThreadPacSimulator(2,1000,8) # seed, runs, num_threads
start = time.perf_counter()
bler = pac_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, conv_coeffs)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

exit()


#polar_sim = polar.PolarSimulator()
polar_sim = polar.PolarSimulator(1)
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, False)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

polar_sim = polar.MultiThreadPolarSimulator(2,1000,8) # seed, runs, num_threads
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, crc_coeffs, False)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

exit()

################################################################################
# genie aided
################################################################################
polar_sim = polar.PolarSimulator(1)
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, np.array([]), True)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

polar_sim = polar.MultiThreadPolarSimulator(2,1000,8) # seed, runs, num_threads
start = time.perf_counter()
bler = polar_sim.get_bler(N, L, snr_db, max_err, max_runs, frozen_bits, np.array([]), True)
end = time.perf_counter()
print('bler',bler)
print(f'time {end-start:.2f} secs')

