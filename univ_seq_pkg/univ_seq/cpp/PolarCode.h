//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

//
// Updated by David KW Ho on 10/23/2025.
// Created by Saurabh Tavildar on 5/17/16.
//

#ifndef POLARC_POLARCODE_H
#define POLARC_POLARCODE_H


#include <iostream>
#include <cstdint>
#include <vector>
#include <math.h>
#include <stack>          // std::stack
#include <random>
#include <cmath>
#include <algorithm>

// type definitions
typedef std::vector<uint8_t>  bvector_t;
typedef std::vector<uint16_t> uvector_t;
typedef std::vector<double>   dvector_t;
using std::vector;


// CRC class
class CRC {

public:
  CRC(bvector_t& crc_coeffs) :
    divisor(crc_coeffs),
    crc_size(crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0)
  {}

  void add_crc(bvector_t& info_bits_padded, uint16_t info_length)
  {
    if (crc_size == 0)
      return;

    uint16_t eff_length = info_length + crc_size;
    bvector_t remainder = info_bits_padded;

    for (uint16_t i = 0; i < info_length; i++) {
      if (remainder.at(i) == 1) {
        for (uint16_t j = 0; j < divisor.size(); j++)
          remainder.at(i+j) ^= divisor.at(j);
      }
    }

    for (uint16_t i = info_length; i < eff_length; i++)
      info_bits_padded.at(i) = remainder.at(i);

  }

  bool crc_check(bvector_t& info_bits_padded, uint16_t info_length)
  {
    if (crc_size == 0)
      return true;

    uint16_t eff_length = info_length + crc_size;
    bvector_t remainder = info_bits_padded;

    for (uint16_t i = 0; i < info_length; i++)
      if (remainder.at(i) == 1)
        for (uint16_t j = 0; j < divisor.size(); j++)
          remainder.at(i+j) ^= divisor.at(j);

    uint16_t sum = 0;
    for (uint16_t i = info_length; i < eff_length; i++)
      sum += remainder.at(i);

    return (sum == 0);

  }


private:
  bvector_t divisor;
  uint16_t crc_size; // coeffs length - 1

}; // CRC

// Convolutional transform
class Conv {

public:

  Conv(bvector_t conv_coeffs) : conv_coeffs(conv_coeffs), conv_size(conv_coeffs.size()) {}

  void push(uint8_t v, bvector_t& state)
  {
    for (int16_t i = conv_size-2; i >= 0; i--) // count down to 0 (i can be negative)
      state.at(i+1) = state.at(i);
    state.at(0) = v;
  }

  uint8_t conv1bTrans(uint8_t v, bvector_t& state)
  {
    uint8_t u = 0;
    push(v, state);
    for(uint16_t i = 0; i < conv_size; i++)
      u ^= conv_coeffs.at(i) & state.at(i);
    return u;
  }

  void convTrans(const bvector_t& v_vec, bvector_t& u_vec, bvector_t& state, int max_index, const bvector_t& shortening_mask)
  {
    if (shortening_mask.size() == 0) {
      for (uint16_t i = 0; i <= max_index; i++)
        u_vec.at(i) = conv1bTrans(v_vec.at(i), state);
    } else {
      for (uint16_t i = 0; i <= max_index; i++) {
        if (shortening_mask.at(i) == 0) {
          u_vec.at(i) = conv1bTrans(v_vec.at(i), state);
        } else {
          u_vec.at(i) = 0;                          // force output to zero
//          std::fill(state.begin(), state.end(), 0); // reset state
        }
      }
    }
  }

private:
  bvector_t conv_coeffs;
  uint8_t conv_size;
};

using std::ceil;
using std::log2;
using std::pow;
using std::min;
using std::max;
using std::sqrt;

// NRStandard class
class NRStandard {

public:
    NRStandard() {}

    NRStandard(uvector_t seq, uint16_t K, uint16_t E, bool DL) : seq(seq), K(K), E(E), J(N_max) {
      n = get_n(K,E,DL);
      N = 1 << n;
      set_seq_N();
      get_sb_indices();
    }

    void set_seq_N() {
      seq_N.resize(N, -1);
      uint16_t len = seq.size();
      for (uint16_t i=0, j=0; i<len && j<N; i++) {
        uint16_t index = seq.at(i);
        if (index < N) {
          seq_N.at(j) = index;
          j++;
        }
      }
    }

    // section 5.3.1, TS 38.212
    uint16_t get_n(double K, double E, bool DL = true) {

      uint16_t n_min = 5;
      uint16_t n_max = DL ? 9 : 10;
      bool cond = (E <= (9./8.) * pow(2., ceil(log2(E))-1)) && (K/E < 9./16.);
      uint16_t n1 = ceil(log2(E));
               n1 = cond ? n1 - 1 : n1;
      uint16_t n2 = ceil(log2(8.*K));
      uint16_t n = max( min({n1, n2, n_max}), n_min );

      return n;
    }

    // Table 5.4.1.1-1, TS 38.212
    uint16_t sub_block_intlv_seq[32] = {
      0,  1,  2,  4,  3,  5,  6,  7,
      8, 16,  9, 17, 10, 18, 11, 19,
     12, 20, 13, 21, 14, 22, 15, 23,
     24, 25, 26, 28, 27, 29, 30, 31,
    };

    // section 5.4.1.1
    void get_sb_indices() {
      uint16_t blk_len = N/32;
      uint16_t* P = sub_block_intlv_seq;
      for (uint16_t n = 0; n<N; n++) {
        uint16_t i = (uint32_t)32*n/N;
        J[n] = P[i] * blk_len + (n % blk_len);
      }
    }

    bvector_t sub_block_interleave(bvector_t in_seq) {
      bvector_t out_seq(N);
      for (uint16_t i = 0; i<N; i++) {
        out_seq.at(i) = in_seq.at(J[i]);
      }
      return out_seq;
    }

    dvector_t sub_block_deinterleave(dvector_t in_seq) {
      dvector_t out_seq(N);
      for (uint16_t i = 0; i<N; i++) {
        out_seq.at(J[i]) = in_seq.at(i);
      }
      return out_seq;
    }

    // section 5.4.1.2
    bvector_t rate_match(bvector_t in_seq) {
      bvector_t out_seq(E);
      if (E >= N) { // repetition
        for (uint16_t i = 0; i<E; i++) {
          out_seq[i] = in_seq[i % N];
        }
      } else { // E < N
        if ((double)K/E <= 7./16.) { // puncturing
          uint16_t U = N-E;
          for (uint16_t i = 0; i<E; i++) {
            out_seq[i] = in_seq[U+i];
          }
        } else { // shortening
          for (uint16_t i = 0; i<E; i++) {
            out_seq[i] = in_seq[i];
          }
        }
      }
      return out_seq;
    }

    dvector_t rate_recovery(dvector_t in_seq) {
      dvector_t out_seq(N);
      std::fill(out_seq.begin(), out_seq.end(), 0.);
      if (E >= N) { // repetition
        for (uint16_t i = 0; i<E; i++) {
          out_seq[i % N] += in_seq[i];
        }
      } else {
        if ((double)K/E <= 7./16.) { // puncturing
          uint16_t U = N-E;
          for (uint16_t i = 0; i<E; i++) {
            out_seq[U+i] = in_seq[i];
          }
        } else { // shortening
          for (uint16_t i = 0; i<E; i++) {
            out_seq[i] = in_seq[i];
          }
          for (uint16_t i = E; i<N; i++) {
            out_seq[i] = LLR_max;
          }
        }
      }
      return out_seq;
    }

    // channel interleaver (used in UCI only)
    // section 5.4.1.3
    int get_T(int E) {
        return uint16_t(ceil((sqrt(8.*E+1)-1)/2));
    }

    bvector_t channel_interleave(bvector_t in_seq) {
        uint16_t T = get_T(E);
        bvector_t out_seq(E);
        vector<vector<int>> v_matrix(T, vector<int>(T,0));
        // fill
        uint16_t k = 0;
        for (uint16_t i = 0; i<T; i++) {
            for (uint16_t j = 0; j<(T-i); j++) {
                if (k < E) {
                    v_matrix[i][j] = in_seq[k];
                } else {
                    v_matrix[i][j] = -1;
                }
                k++;
            }
        }
        // read
        k = 0;
        for (uint16_t j = 0; j<T; j++) {
            for (uint16_t i = 0; i<(T-j); i++) {
                if (v_matrix[i][j] != -1) {
                    out_seq[k] = v_matrix[i][j];
                    k++;
                }
            }
        }
        return out_seq;
    }

    int num_zeros_accum[128] = {
         0,    1,    3,    6,   10,   15,   21,   28,   36,   45,   55,
        66,   78,   91,  105,  120,  136,  153,  171,  190,  210,  231,
       253,  276,  300,  325,  351,  378,  406,  435,  465,  496,  528,
       561,  595,  630,  666,  703,  741,  780,  820,  861,  903,  946,
       990, 1035, 1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485,
      1540, 1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
      2211, 2278, 2346, 2415, 2485, 2556, 2628, 2701, 2775, 2850, 2926,
      3003, 3081, 3160, 3240, 3321, 3403, 3486, 3570, 3655, 3741, 3828,
      3916, 4005, 4095, 4186, 4278, 4371, 4465, 4560, 4656, 4753, 4851,
      4950, 5050, 5151, 5253, 5356, 5460, 5565, 5671, 5778, 5886, 5995,
      6105, 6216, 6328, 6441, 6555, 6670, 6786, 6903, 7021, 7140, 7260,
      7381, 7503, 7626, 7750, 7875, 8001, 8128
    };

    dvector_t channel_deinterleave(dvector_t in_seq) {
        uint16_t T = get_T(E);
        dvector_t out_seq(E);
        vector<vector<double>> v_matrix(T, vector<double>(T,0.));

        // fill
        uint16_t k = 0;
        for (uint16_t j = 0; j<T; j++) {
            for (uint16_t i = 0; i<(T-j); i++) {
                uint16_t index = i*T + j;
                index = i > 0 ? index - num_zeros_accum[i-1] : index;
                if (index < E) {
                    v_matrix[i][j] = in_seq[k];
                    k++;
                } else {
                    v_matrix[i][j] = -1;
                }
            }
        }
        // read
        k = 0;
        for (uint16_t i = 0; i<T; i++) {
            for (uint16_t j = 0; j<(T-i); j++) {
                if (k < E) {
                    out_seq[k] = v_matrix[i][j];
                }
                k++;
            }
        }
        return out_seq;
    }

    bvector_t get_frozen_bitmap(bvector_t& shortening_mask) {
      //uvector_t &Q_N = seq;
      //uvector_t Q_F_tmp;
      uint16_t N = 1 << n;
      bvector_t frozen_bitmap(N, 0);

      if (E < N) {
        uint16_t U = N-E;
        if ((double)K/E <= 7./16.) { // puncturing
          for (uint16_t i=0; i<U; i++) {
            frozen_bitmap.at(J[i]) = 1; // Q_F_tmp = J[:U]
          }
          uint16_t T;
          if ((double)E >= 3.*N/4.) {
            T = uint16_t(ceil(3.*N/4. - E/2.));
          } else {
            T = uint16_t(ceil(9.*N/16. - E/4.));
          }
          for (uint16_t i=0; i<T; i++) {
            frozen_bitmap.at(i) = 1; // Q_F_tmp = union(Q_F_tmp, T_set)
          }
        } else { // shortening
          for (uint16_t i=0; i<U; i++) {
            frozen_bitmap.at(J[E+i]) = 1; // Q_F_tmp = J[E:]
          }
          // return shortening mask
          shortening_mask = frozen_bitmap;
        }
      }
      uint16_t n_frozen = std::accumulate(frozen_bitmap.begin(), frozen_bitmap.end(), 0);
      uint16_t n_f = N - K - n_frozen;
      for (uint16_t i=0, cnt=0; i<N && cnt<n_f; i++) {
        if (frozen_bitmap.at(seq_N.at(i)) == 0) {
          frozen_bitmap.at(seq_N.at(i)) = 1;
          cnt++;
        }
      }
      return frozen_bitmap;
    }

public:
    uint16_t n;

private:
    static constexpr uint16_t N_max = 1024;
    static constexpr double LLR_max = 1000;
    std::vector<uint16_t> seq;
    std::vector<uint16_t> seq_N;
    uint16_t K;
    uint16_t E;
    uint16_t N;
    std::vector<uint16_t> J;

}; // NRStandard class



// PolarCode class
class PolarCode {


public:

    PolarCode(uint8_t num_layers, uint16_t info_length, double epsilon, bvector_t crc_coeffs, bvector_t conv_ceoffs) :
            _n(num_layers), _info_length(info_length), _design_epsilon(epsilon),
            _crc_size(crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0),
            _crc(crc_coeffs),
            _conv(conv_ceoffs),
            _conv_size(conv_ceoffs.size()),
            _pac_code(false),
            _use_external_frozen_bits(false),
            _genie_aided(false),
            _llr_based_computation(true)
    {
        _block_length = (uint16_t) (1 << _n);
        _frozen_bits.resize(_block_length);
        _bit_rev_order.resize(_block_length);
        create_bit_rev_order();
        initialize_frozen_bits();
    }

    PolarCode(uint8_t num_layers, uint16_t info_length,
              bvector_t frozen_bits, bvector_t crc_coeffs, bool genie_aided, bvector_t conv_coeffs) :
            _n(num_layers), _info_length(info_length),
            _crc_size(crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0),
            _crc(crc_coeffs),
            _conv(conv_coeffs),
            _conv_size(conv_coeffs.size()),
            _pac_code(conv_coeffs.size()>0),
            _use_external_frozen_bits(true),
            _genie_aided(genie_aided),
            _rate_matching(false),
            _channel_interleave(false),
            _frozen_bits(frozen_bits),
            _shortening_mask(0),
            _llr_based_computation(true)
    {
        _block_length = (uint16_t) (1 << _n);
        //_frozen_bits.resize(_block_length);
        _bit_rev_order.resize(_block_length);
        create_bit_rev_order();
        initialize_frozen_bits();
    }

    PolarCode(uint16_t info_length, uint16_t codeword_length, // A, E
              uvector_t sequence, bvector_t crc_coeffs, bool genie_aided,
              bool channel_interleave, bool downlink, bvector_t conv_coeffs) :
            _info_length(info_length),
            _codeword_length(codeword_length),
            _crc_size(crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0),
            _k(info_length + _crc_size),
            _crc(crc_coeffs),
            _conv(conv_coeffs),
            _conv_size(conv_coeffs.size()),
            _pac_code(conv_coeffs.size()>0),
            _use_external_frozen_bits(true),
            _genie_aided(genie_aided),
            _nr_std(sequence, _k, codeword_length, downlink),
            _rate_matching(true),
            _channel_interleave(channel_interleave),
            _shortening_mask(0),
            _llr_based_computation(true)
    {
        _n = _nr_std.n;
        _block_length = (uint16_t) (1 << _n);
        _frozen_bits = _nr_std.get_frozen_bitmap(_shortening_mask);
        _bit_rev_order.resize(_block_length);
        create_bit_rev_order();
        initialize_frozen_bits();
    }


    std::vector<uint8_t> encode(std::vector<uint8_t> info_bits);
    std::vector<uint8_t> decode_scl_p1(std::vector<double> p1, std::vector<double> p0, uint16_t list_size);
    std::vector<uint8_t> decode_scl_llr(std::vector<double> llr, uint16_t list_size);

    double get_bler(double snr_db, uint8_t list_size, int max_err, int max_runs, std::mt19937& generator);
    double get_errors(double snr_db, uint8_t list_size, int max_runs, std::mt19937& generator);


private:

    uint16_t _n;
    uint16_t _info_length;
    uint16_t _block_length;
    uint16_t _codeword_length;
    double _design_epsilon;
    uint16_t _crc_size;
    uint16_t _k;

    // DKWHO
    CRC _crc;
    Conv _conv;
    uint8_t _conv_size;
    bool _pac_code;
    bool _use_external_frozen_bits;
    bool _genie_aided;
    NRStandard _nr_std;
    bool _rate_matching;
    bool _channel_interleave;

    std::vector<uint8_t> _info_bits;
    std::vector<uint8_t> _frozen_bits;
    std::vector<uint8_t> _shortening_mask;
    std::vector<uint16_t> _channel_order_descending;
    std::vector<uint16_t> _bit_rev_order;

    void initialize_frozen_bits();
    void create_bit_rev_order();

    std::vector<uint8_t> decode_scl();
    bool _llr_based_computation;

    std::vector<std::vector<double *>> _arrayPointer_LLR;
    std::vector<double> _pathMetric_LLR;

    uint16_t _list_size;

    std::stack<uint16_t> _inactivePathIndices;
    std::vector<uint16_t > _activePath;
    std::vector<std::vector<double *>> _arrayPointer_P;
    std::vector<std::vector<uint8_t *>> _arrayPointer_C;
    std::vector<uint8_t *> _arrayPointer_Info;
    std::vector<uint8_t *> _arrayPointer_Polar;
    std::vector<std::vector<uint8_t>> _stateVectors;
    std::vector<std::vector<uint16_t>> _pathIndexToArrayIndex;
    std::vector<std::stack<uint16_t>> _inactiveArrayIndices;
    std::vector<std::vector<uint16_t>> _arrayReferenceCount;

    void initializeDataStructures();
    uint16_t assignInitialPath();
    uint16_t clonePath(uint16_t);
    void killPath(uint16_t l);

    double * getArrayPointer_P(uint16_t lambda, uint16_t  l);
    double * getArrayPointer_LLR(uint16_t lambda, uint16_t  l);
    uint8_t * getArrayPointer_C(uint16_t lambda, uint16_t  l);

    void recursivelyCalcP(uint16_t lambda, uint16_t phi);
    void recursivelyCalcLLR(uint16_t lambda, uint16_t phi);
    void recursivelyUpdateC(uint16_t lambda, uint16_t phi);

    void continuePaths_FrozenBit(uint16_t phi);
    void continuePaths_UnfrozenBit(uint16_t phi);

    uint16_t findMostProbablePath(bool check_crc);

    bool genie_check(uint8_t * info_bits_padded);
    bool crc_check(uint8_t * info_bits_padded);

};


#endif //POLARC_POLARCODE_H
