//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

//
// Updated by David KW Ho on 10/23/2025.
// Created by Saurabh Tavildar on 5/17/16.
//

#include "PolarCode.h"
#include <iostream>
#include <cmath>       /* log */
#include <sstream>      // std::stringstream
#include <fstream>
#include <iomanip>      // std::setprecision
#include <random>
#include <algorithm>
#include <chrono>

void PolarCode::initialize_frozen_bits() {

    if (not _use_external_frozen_bits) {
      std::vector<double> channel_vec(_block_length);

      for (uint16_t i = 0; i < _block_length; ++i) {
          channel_vec.at(i) = _design_epsilon;
      }
      for (uint8_t iteration = 0; iteration < _n; ++iteration) {
          uint16_t  increment = 1 << iteration;
          for (uint16_t j = 0; j < increment; j +=  1) {
              for (uint16_t i = 0; i < _block_length; i += 2 * increment) {
                  double c1 = channel_vec.at(i + j);
                  double c2 = channel_vec.at(i + j + increment);
                  channel_vec.at(i + j) = c1 + c2 - c1*c2;
                  channel_vec.at(i + j + increment) = c1*c2;
              }
          }
      }

      _channel_order_descending.resize(_block_length);
      std::size_t n_t(0);
      std::generate(std::begin(_channel_order_descending), std::end(_channel_order_descending), [&]{ return n_t++; });
      std::sort(  std::begin(_channel_order_descending),
                  std::end(_channel_order_descending),
                  [&](int i1, int i2) { return channel_vec[_bit_rev_order.at(i1)] < channel_vec[_bit_rev_order.at(i2)]; } );


      uint16_t  effective_info_length = _info_length + _crc_size;

      for (uint16_t i = 0; i < effective_info_length; ++i) {
          _frozen_bits.at(_channel_order_descending.at(i)) = 0;
      }
      for (uint16_t i = effective_info_length; i < _block_length; ++i) {
          _frozen_bits.at(_channel_order_descending.at((i))) = 1;
      }

    } else { // _use_external_frozen_bits

      // generate channel_order_descending from bitmap
      _channel_order_descending.resize(_block_length);
      for (int16_t i = _block_length - 1, j = 0; i >= 0 ; i--) {
          if (_frozen_bits.at(i) == 0) {
            _channel_order_descending.at(j) = i;
            j++;
          }
      }

    }

#if 0
    std::cout << "frozen_bits: ";
    for (uint16_t i = 0; i < _block_length; ++i) {
        std::cout << int(_frozen_bits.at(i)) << ",";
    }
    std::cout << std::endl;
#endif

}

std::vector<uint8_t> PolarCode::encode(std::vector<uint8_t> info_bits) {

    if (_genie_aided)
        _info_bits = info_bits;

    std::vector<uint8_t> info_bits_padded(_block_length, 0);
    std::vector<uint8_t> coded_bits(_block_length);

    for (uint16_t i = 0; i < _info_length; ++i) {
        info_bits_padded.at(_channel_order_descending.at(i)) = info_bits.at(i);
    }

    // add CRC bits
    bvector_t info_crc_bits = info_bits;
    info_crc_bits.resize(_info_length + _crc_size);
    _crc.add_crc(info_crc_bits, _info_length);
    for (uint16_t i = _info_length; i < _info_length + _crc_size; ++i) {
        info_bits_padded.at(_channel_order_descending.at(i)) = info_crc_bits.at(i);
    }

#if 0
    std::cout << "channel_order_descending: " << std::endl;
    for (uint16_t i = 0; i < _info_length + _crc_size; ++i) {
        std::cout << int(_channel_order_descending.at(i)) << ",";
    }
    std::cout << std::endl;
#endif

#if 0
    std::cout << "input_carrier_bits: " << std::endl;
    for (uint16_t i = 0; i < _block_length; ++i) {
        std::cout << int(info_bits_padded.at(i)) << ",";
    }
    std::cout << std::endl;
#endif

    // convolutional encoding
    bvector_t state(_conv_size, 0);
    bvector_t conv_encoded_bits(_block_length);
    if (_pac_code) {
        int max_index = _channel_order_descending.at(0);
        _conv.convTrans(info_bits_padded, conv_encoded_bits, state, max_index, _shortening_mask);
    } else {
        conv_encoded_bits = info_bits_padded;
    }

#if 0
    if (_pac_code) {
        std::cout << "conv_bits: " << std::endl;
        for (uint16_t i = 0; i < _block_length; ++i) {
            std::cout << int(conv_encoded_bits.at(i)) << ",";
        }
        std::cout << std::endl;
    }
#endif

    // polar transform
    for (uint8_t iteration = 0; iteration < _n; ++iteration) {
        uint16_t  increment = (uint16_t) (1 << iteration);
        for (uint16_t j = 0; j < increment; j +=  1) {
            for (uint16_t i = 0; i < _block_length; i += 2 * increment) {
                conv_encoded_bits.at(i + j) = (uint8_t)((conv_encoded_bits.at(i + j) + conv_encoded_bits.at(i + j + increment)) % 2);
            }
        }
    }

    for (uint16_t i = 0; i < _block_length; ++i) {
        coded_bits.at(i) = conv_encoded_bits.at(i);
    }

    return coded_bits;

}

bool PolarCode::genie_check(uint8_t * info_bit_padded) {

    uint16_t eff_length = _info_length + _crc_size;
    bvector_t info_crc_bits(eff_length);

    bool pass = true;
    for (uint16_t i = 0; i < eff_length; ++i) {
      pass &= _info_bits.at(i) == info_bit_padded[_channel_order_descending.at(i)];
    }

    return pass;
}

bool PolarCode::crc_check(uint8_t * info_bit_padded) {

    uint16_t eff_length = _info_length + _crc_size;
    bvector_t info_crc_bits(eff_length);

    for (uint16_t i = 0; i < _info_length + _crc_size; ++i) {
      info_crc_bits.at(i) = info_bit_padded[_channel_order_descending.at(i)];
    }

    return _crc.crc_check(info_crc_bits, _info_length);

}

std::vector<uint8_t> PolarCode::decode_scl_p1(std::vector<double> p1, std::vector<double> p0, uint16_t list_size) {

    _list_size = list_size;
    _llr_based_computation = false;

    initializeDataStructures();

    uint16_t  l = assignInitialPath();

    double * p_0 = getArrayPointer_P(0, l);

    for (uint16_t beta = 0; beta < _block_length; ++beta ) {
        p_0[2*beta] = (double) p0.at(beta);
        p_0[2*beta + 1] = (double) p1.at(beta);
    }

    return decode_scl();

}

std::vector<uint8_t> PolarCode::decode_scl_llr(std::vector<double> llr, uint16_t list_size) {

    _list_size = list_size;

    _llr_based_computation = true;

    initializeDataStructures();

    uint16_t  l = assignInitialPath();

    double * llr_0 = getArrayPointer_LLR(0, l);

    for (uint16_t beta = 0; beta < _block_length; ++beta ) {
        llr_0[beta] = llr.at(beta);
    }

    return decode_scl();

}

std::vector<uint8_t> PolarCode::decode_scl() {

//    for (uint16_t phi = 0; phi < _block_length; ++phi ){
    for (uint16_t phi = 0; phi <= _channel_order_descending.at(0); ++phi ){
//        std::cout << "phi: " << int(phi) << std::endl;

        if (_llr_based_computation )
            recursivelyCalcLLR(_n, phi);
        else
            recursivelyCalcP(_n, phi);


        if (_frozen_bits.at(phi) == 1)
            continuePaths_FrozenBit(phi);
        else
            continuePaths_UnfrozenBit(phi);

        if ((phi%2) == 1)
            recursivelyUpdateC(_n, phi);

    }
    bool check = (_crc_size > 0) || _genie_aided;
    uint16_t l = findMostProbablePath(check);

    uint8_t * c_0 = _arrayPointer_Info.at(l);
    std::vector<uint8_t> decoded_info_bits(_info_length);
    for (uint16_t beta = 0; beta < _info_length; ++beta )
        decoded_info_bits.at(beta) = c_0[_channel_order_descending.at(beta)];

#if 0
    if (_pac_code) {
        uint8_t * d_0 = _arrayPointer_Polar.at(l);
        std::cout << "decoded_conv_bits: " << std::endl;
        for (uint16_t i = 0; i < _block_length; ++i) {
            std::cout << int(d_0[i]) << ",";
        }
        std::cout << std::endl;
    }
#endif

#if 0
    std::cout << "decoded_input_bits: " << std::endl;
    for (uint16_t i = 0; i < _block_length; ++i) {
        std::cout << int(c_0[i]) << ",";
    }
    std::cout << std::endl;
#endif

    for (uint16_t s = 0; s < _list_size; ++s) {
        delete[] _arrayPointer_Info.at(s);
        delete[] _arrayPointer_Polar.at(s);
        for (uint16_t lambda = 0; lambda < _n + 1; ++lambda) {

            if (_llr_based_computation )
                delete[] _arrayPointer_LLR.at(lambda).at(s);
            else
                delete[] _arrayPointer_P.at(lambda).at(s);
            delete[] _arrayPointer_C.at(lambda).at(s);
        }
    }

    return decoded_info_bits;

}




void PolarCode::initializeDataStructures() {

    while (_inactivePathIndices.size()) {
        _inactivePathIndices.pop();
    };
    _activePath.resize(_list_size);

    if (_llr_based_computation) {
        _pathMetric_LLR.resize(_list_size);
        _arrayPointer_LLR.resize(_n + 1);
        for (int i = 0; i < _n + 1; ++i)
            _arrayPointer_LLR.at(i).resize(_list_size);
    }
    else {
        _arrayPointer_P.resize(_n + 1);
        for (int i = 0; i < _n + 1; ++i)
            _arrayPointer_P.at(i).resize(_list_size);
    }

    _arrayPointer_C.resize(_n + 1);
    for (int i = 0; i < _n + 1; ++i)
        _arrayPointer_C.at(i).resize(_list_size);

    _arrayPointer_Info.resize(_list_size);
    _arrayPointer_Polar.resize(_list_size);
    _stateVectors.resize(_list_size);

    _pathIndexToArrayIndex.resize(_n + 1);
    for (int i = 0; i < _n + 1; ++i)
        _pathIndexToArrayIndex.at(i).resize(_list_size);

    _inactiveArrayIndices.resize(_n + 1);
    for (int i = 0; i < _n + 1; ++i) {
        while (_inactiveArrayIndices.at(i).size()) {
            _inactiveArrayIndices.at(i).pop();
        };
    }

    _arrayReferenceCount.resize(_n + 1);
    for (int i = 0; i < _n + 1; ++i)
        _arrayReferenceCount.at(i).resize(_list_size);

    for (uint16_t s = 0; s < _list_size; ++s) {
        _arrayPointer_Info.at(s) = new uint8_t[_block_length]();
        _arrayPointer_Polar.at(s) = new uint8_t[_block_length]();
        _stateVectors.at(s).clear();
        _stateVectors.at(s).resize(_conv_size);

        for (uint16_t lambda = 0; lambda < _n + 1; ++lambda) {
            if (_llr_based_computation) {
                _arrayPointer_LLR.at(lambda).at(s) = new double[(1 << (_n - lambda))]();
            }
            else {
                _arrayPointer_P.at(lambda).at(s) = new double[2 * (1 << (_n - lambda))]();
            }
            _arrayPointer_C.at(lambda).at(s) = new uint8_t[2 * (1 << (_n - lambda))]();
            _arrayReferenceCount.at(lambda).at(s) = 0;
            _inactiveArrayIndices.at(lambda).push(s);
        }
    }

    for (uint16_t l = 0; l < _list_size; ++l) {
        _activePath.at(l) = 0;
        _inactivePathIndices.push(l);
        if (_llr_based_computation) {
            _pathMetric_LLR.at(l) = 0;
        }
    }
}

uint16_t PolarCode::assignInitialPath() {

    uint16_t  l = _inactivePathIndices.top();
    _inactivePathIndices.pop();
    _activePath.at(l) = 1;
    // Associate arrays with path index
    for (uint16_t lambda = 0; lambda < _n + 1; ++lambda) {
        uint16_t  s = _inactiveArrayIndices.at(lambda).top();
        _inactiveArrayIndices.at(lambda).pop();
        _pathIndexToArrayIndex.at(lambda).at(l) = s;
        _arrayReferenceCount.at(lambda).at(s) = 1;
    }
    return l;
}

uint16_t PolarCode::clonePath(uint16_t l) {
    uint16_t l_p = _inactivePathIndices.top();
    _inactivePathIndices.pop();
    _activePath.at(l_p) = 1;

    if (_llr_based_computation)
        _pathMetric_LLR.at(l_p) = _pathMetric_LLR.at(l);

    for (uint16_t lambda = 0; lambda < _n + 1; ++lambda ) {
        uint16_t s = _pathIndexToArrayIndex.at(lambda).at(l);
        _pathIndexToArrayIndex.at(lambda).at(l_p) = s;
        _arrayReferenceCount.at(lambda).at(s)++;
    }
    return l_p;
}

void PolarCode::killPath(uint16_t l) {
    _activePath.at(l) = 0;
    _inactivePathIndices.push(l);
    if (_llr_based_computation )
        _pathMetric_LLR.at(l) = 0;

    for (uint16_t lambda = 0; lambda < _n + 1; ++lambda ) {
        uint16_t s = _pathIndexToArrayIndex.at(lambda).at(l);
        _arrayReferenceCount.at(lambda).at(s)--;
        if (_arrayReferenceCount.at(lambda).at(s) == 0 ) {
            _inactiveArrayIndices.at(lambda).push(s);
        }
    }
}

double * PolarCode::getArrayPointer_P(uint16_t lambda, uint16_t  l) {
    uint16_t  s = _pathIndexToArrayIndex.at(lambda).at(l);
    uint16_t s_p;
    if (_arrayReferenceCount.at(lambda).at(s) == 1) {
        s_p = s;
    }
    else {
        s_p = _inactiveArrayIndices.at(lambda).top();
        _inactiveArrayIndices.at(lambda).pop();

        //copy
        std::copy(_arrayPointer_P.at(lambda).at(s), _arrayPointer_P.at(lambda).at(s) +  (1 << (_n - lambda + 1)),  _arrayPointer_P.at(lambda).at(s_p));
        std::copy(_arrayPointer_C.at(lambda).at(s), _arrayPointer_C.at(lambda).at(s) +  (1 << (_n - lambda + 1)),  _arrayPointer_C.at(lambda).at(s_p));

        _arrayReferenceCount.at(lambda).at(s)--;
        _arrayReferenceCount.at(lambda).at(s_p) = 1;
        _pathIndexToArrayIndex.at(lambda).at(l) = s_p;
    }
    return _arrayPointer_P.at(lambda).at(s_p);
}

double * PolarCode::getArrayPointer_LLR(uint16_t lambda, uint16_t  l) {
    uint16_t  s = _pathIndexToArrayIndex.at(lambda).at(l);
    uint16_t s_p;
    if (_arrayReferenceCount.at(lambda).at(s) == 1) {
        s_p = s;
    }
    else {
        s_p = _inactiveArrayIndices.at(lambda).top();
        _inactiveArrayIndices.at(lambda).pop();

        //copy
        std::copy(_arrayPointer_C.at(lambda).at(s), _arrayPointer_C.at(lambda).at(s) +  (1 << (_n - lambda + 1)),  _arrayPointer_C.at(lambda).at(s_p));
        std::copy(_arrayPointer_LLR.at(lambda).at(s), _arrayPointer_LLR.at(lambda).at(s) +  (1 << (_n - lambda)),  _arrayPointer_LLR.at(lambda).at(s_p));

        _arrayReferenceCount.at(lambda).at(s)--;
        _arrayReferenceCount.at(lambda).at(s_p) = 1;
        _pathIndexToArrayIndex.at(lambda).at(l) = s_p;
    }
    return _arrayPointer_LLR.at(lambda).at(s_p);
}


uint8_t * PolarCode::getArrayPointer_C(uint16_t lambda, uint16_t  l) {
    uint16_t  s = _pathIndexToArrayIndex.at(lambda).at(l);
    uint16_t s_p;
    if (_arrayReferenceCount.at(lambda).at(s) == 1) {
        s_p = s;
    }
    else {

        s_p = _inactiveArrayIndices.at(lambda).top();
        _inactiveArrayIndices.at(lambda).pop();

        //copy
        if (_llr_based_computation )
            std::copy(_arrayPointer_LLR.at(lambda).at(s), _arrayPointer_LLR.at(lambda).at(s) +  (1 << (_n - lambda)),  _arrayPointer_LLR.at(lambda).at(s_p));
        else
            std::copy(_arrayPointer_P.at(lambda).at(s), _arrayPointer_P.at(lambda).at(s) +  (1 << (_n - lambda + 1)),  _arrayPointer_P.at(lambda).at(s_p));

        std::copy(_arrayPointer_C.at(lambda).at(s), _arrayPointer_C.at(lambda).at(s) +  (1 << (_n - lambda + 1)),  _arrayPointer_C.at(lambda).at(s_p));

        _arrayReferenceCount.at(lambda).at(s)--;
        _arrayReferenceCount.at(lambda).at(s_p) = 1;
        _pathIndexToArrayIndex.at(lambda).at(l) = s_p;

    }
    return _arrayPointer_C.at(lambda).at(s_p);
}

void PolarCode::recursivelyCalcP(uint16_t lambda, uint16_t phi) {
    if ( lambda == 0 )
        return;
    uint16_t psi = phi >> 1;
    if ( (phi % 2) == 0)
        recursivelyCalcP(lambda -1, psi);

    double sigma = 0.0f;
    for (uint16_t l = 0; l < _list_size; ++l) {
        if (_activePath.at(l) == 0)
            continue;
        double * p_lambda = getArrayPointer_P(lambda, l);
        double * p_lambda_1 = getArrayPointer_P(lambda - 1, l);

        uint8_t * c_lambda = getArrayPointer_C(lambda, l);
        for (uint16_t beta = 0; beta < (1 << (_n - lambda)); ++beta) {
            if ( (phi %2) == 0 ){
                p_lambda[2 * beta] = 0.5f * ( p_lambda_1[2*(2*beta)]*p_lambda_1[2*(2*beta+1)]
                                              + p_lambda_1[2*(2*beta) + 1]*p_lambda_1[2*(2*beta+1) + 1]);
                p_lambda[2 * beta + 1] = 0.5f * ( p_lambda_1[2*(2*beta) +1]*p_lambda_1[2*(2*beta+1)]
                                                  + p_lambda_1[2*(2*beta)]*p_lambda_1[2*(2*beta+1) + 1]);
            }
            else {
                uint8_t  u_p = c_lambda[2*beta];
                p_lambda[2 * beta] = 0.5f * p_lambda_1[2*(2*beta) + (u_p % 2)] *   p_lambda_1[2*(2*beta + 1)];
                p_lambda[2 * beta + 1] = 0.5f * p_lambda_1[2*(2*beta) + ((u_p+1) % 2)] *   p_lambda_1[2*(2*beta + 1) + 1];
            }
            sigma = std::max(sigma,  p_lambda[2 * beta]);
            sigma = std::max(sigma,  p_lambda[2 * beta + 1]);


        }
    }

    for (uint16_t l = 0; l < _list_size; ++l) {
        if (sigma == 0) // Typically happens because of undeflow
            break;
        if (_activePath.at(l) == 0)
            continue;
        double *p_lambda = getArrayPointer_P(lambda, l);
        for (uint16_t beta = 0; beta < (1 << (_n - lambda)); ++beta) {
            p_lambda[2 * beta] = p_lambda[2 * beta] / sigma;
            p_lambda[2 * beta + 1] = p_lambda[2 * beta + 1] / sigma;
        }
    }
}

void PolarCode::recursivelyCalcLLR(uint16_t lambda, uint16_t phi) {
    if ( lambda == 0 )
        return;
    uint16_t psi = phi >> 1;
    if ( (phi % 2) == 0)
        recursivelyCalcLLR(lambda -1, psi);

    for (uint16_t l = 0; l < _list_size; ++l) {
        if (_activePath.at(l) == 0)
            continue;
        double * llr_lambda = getArrayPointer_LLR(lambda, l);
        double * llr_lambda_1 = getArrayPointer_LLR(lambda - 1, l);

        uint8_t * c_lambda = getArrayPointer_C(lambda, l);
        for (uint16_t beta = 0; beta < (1 << (_n - lambda)); ++beta) {
            if ( (phi %2) == 0 ){
                if (40 > std::max(std::abs(llr_lambda_1[2 * beta]), std::abs(llr_lambda_1[2 * beta + 1]))){
                    llr_lambda[beta] = std::log ( (exp(llr_lambda_1[2 * beta] + llr_lambda_1[2 * beta + 1]) + 1) /
                                                  (exp(llr_lambda_1[2*beta]) + exp(llr_lambda_1[2*beta+1])));
                }
                else {
                    llr_lambda[beta] = (double)  ((llr_lambda_1[2 * beta] < 0) ? -1 : (llr_lambda_1[2 * beta] > 0)) *
                                       ((llr_lambda_1[2 * beta + 1] < 0) ? -1 : (llr_lambda_1[2 * beta + 1] > 0)) *
                                       std::min( std::abs(llr_lambda_1[2 * beta]), std::abs(llr_lambda_1[2 * beta + 1]));
                }
            }
            else {
                uint8_t  u_p = c_lambda[2*beta];
                llr_lambda[beta] = (1 - 2 * u_p) * llr_lambda_1[2*beta] + llr_lambda_1[2*beta + 1];
            }

        }
    }
}

void PolarCode::recursivelyUpdateC(uint16_t lambda, uint16_t phi) {

    uint16_t psi = phi >> 1;
    for (uint16_t l = 0; l < _list_size; ++l) {
        if (_activePath.at(l) == 0)
            continue;
        uint8_t *c_lambda = getArrayPointer_C(lambda, l);
        uint8_t *c_lambda_1 = getArrayPointer_C(lambda - 1, l);
        for (uint16_t beta = 0; beta < (1 << (_n - lambda)); ++beta) {
            c_lambda_1[2 * (2 * beta) + (psi % 2)] = (uint8_t) ((c_lambda[2 * beta] + c_lambda[2 * beta + 1]) % 2);
            c_lambda_1[2 * (2 * beta + 1) + (psi % 2)] = c_lambda[2 * beta + 1];
        }
    }
    if ( (psi % 2) == 1)
        recursivelyUpdateC((uint16_t) (lambda - 1), psi);

}

void PolarCode::continuePaths_FrozenBit(uint16_t phi) {
    for (uint16_t l = 0; l < _list_size; ++ l) {
        if (_activePath.at(l) == 0)
            continue;
        uint8_t  * c_m = getArrayPointer_C(_n, l);
        uint8_t z;
        double sign;
        if (_pac_code) {
            // compute dynamic frozen bit - DKWHO
            if (_shortening_mask.size() > 0 && _shortening_mask.at(phi) == 1) {
                // force bit to zero and reset state vector
                z = 0;
                sign = 1.0;
                c_m[(phi % 2)] = 0;
//                bvector_t& state = _stateVectors.at(l);
//                std::fill(state.begin(), state.end(), 0);
            } else {
                z = _conv.conv1bTrans(0, _stateVectors.at(l));
                sign = z == 0 ?  1 : -1;
                c_m[(phi % 2)] = z;
            }
            _arrayPointer_Polar.at(l)[phi] = z;
            _arrayPointer_Info.at(l)[phi] = 0;
        } else {
            z = 0;
            sign = 1.0;
            c_m[(phi % 2)] = 0; // frozen value assumed to be zero
            _arrayPointer_Info.at(l)[phi] = 0;
        }

        if (_llr_based_computation) {
            double *llr_p = getArrayPointer_LLR(_n, l);
            _pathMetric_LLR.at(l) += log(1 + exp(-sign*llr_p[0]));
        }
    }
}

void PolarCode::continuePaths_UnfrozenBit(uint16_t phi) {

    std::vector<double>  probForks((unsigned long) (2 * _list_size));
    std::vector<double> probabilities;
    std::vector<uint8_t>  contForks((unsigned long) (2 * _list_size));


    uint16_t  i = 0;
    for (unsigned l = 0; l < _list_size; ++l) {
        if (_activePath.at(l) == 0) {
            probForks.at(2 * l) = NAN;
            probForks.at(2 * l + 1) = NAN;
        }
        else {
            if (_llr_based_computation ) {
                double sign;
                uint8_t z;
                if (_pac_code) {
                    // PAC code
                    bvector_t state = _stateVectors.at(l);
                    z = _conv.conv1bTrans(0, state);
                    sign = z == 0 ?  1 : -1;
                } else {
                    sign = 1.0;
                    z = 0;
                }
                double *llr_p = getArrayPointer_LLR(_n, l);
                probForks.at(2 * l) =  - (_pathMetric_LLR.at(l) + log(1 + exp(-sign*llr_p[0])));
                probForks.at(2 * l + 1) = -  (_pathMetric_LLR.at(l) + log(1 + exp(sign*llr_p[0])));
            }
            else {
                double *p_m = getArrayPointer_P(_n, l);
                probForks.at(2 * l) = p_m[0];
                probForks.at(2 * l + 1) = p_m[1];
            }

            probabilities.push_back(probForks.at(2 * l));
            probabilities.push_back(probForks.at(2 * l +1));

            i++;
        }
    }

    uint16_t  rho = _list_size;
    if ( (2*i) < _list_size)
        rho = (uint16_t) 2 * i;

    for (uint8_t l = 0; l < 2 * _list_size; ++l) {
        contForks.at(l) = 0;
    }
    std::sort(probabilities.begin(), probabilities.end(), std::greater<double>());

    double threshold = probabilities.at((unsigned long) (rho - 1));
    uint16_t num_paths_continued = 0;

    for (uint8_t l = 0; l < 2 * _list_size; ++l) {
        if (probForks.at(l) > threshold) {
            contForks.at(l) = 1;
            num_paths_continued++;
        }
        if (num_paths_continued == rho) {
            break;
        }
    }

    if  ( num_paths_continued < rho ) {
        for (uint8_t l = 0; l < 2 * _list_size; ++l) {
            if (probForks.at(l) == threshold) {
                contForks.at(l) = 1;
                num_paths_continued++;
            }
            if (num_paths_continued == rho) {
                break;
            }
        }
    }

    for (unsigned l = 0; l < _list_size; ++l) {
        if (_activePath.at(l) == 0)
            continue;
        if ( contForks.at(2 * l)== 0 && contForks.at(2 * l + 1) == 0 )
            killPath(l);
    }

    for (unsigned l = 0; l < _list_size; ++l) {
        if ( contForks.at(2 * l) == 0 && contForks.at(2 * l + 1) == 0 )
            continue;
        uint8_t * c_m = getArrayPointer_C(_n, l);

        bvector_t state;
        double sign;
        uint8_t z;
        if (_pac_code) {
            // PAC code
            state = _stateVectors.at(l);
            z = _conv.conv1bTrans(0, state);
            sign = z == 0 ?  1 : -1;
        } else {
            z = 0;
            sign = 1.0;
        }

        if ( contForks.at(2 * l) == 1 && contForks.at(2 * l + 1) == 1 ) {

            c_m[(phi%2)] = z;
            uint16_t l_p = clonePath(l);
            c_m = getArrayPointer_C(_n, l_p);
            c_m[(phi%2)] = z^1;

            if (_pac_code) {
                std::copy(_arrayPointer_Polar.at(l), _arrayPointer_Polar.at(l) +  phi,  _arrayPointer_Polar.at(l_p));
                _arrayPointer_Polar.at(l)[phi] = z;
                _arrayPointer_Polar.at(l_p)[phi] = z^1;
            }

            std::copy(_arrayPointer_Info.at(l), _arrayPointer_Info.at(l) +  phi,  _arrayPointer_Info.at(l_p));
            _arrayPointer_Info.at(l)[phi] = 0;
            _arrayPointer_Info.at(l_p)[phi] = 1;

            if (_pac_code) {
                _stateVectors.at(l) = state;
                state.at(0) = 1;
                _stateVectors.at(l_p) = state;
            }

            if (_llr_based_computation ) {
                // update path metric
                double *llr_p = getArrayPointer_LLR(_n, l);
                _pathMetric_LLR.at(l) += log(1 + exp(-sign*llr_p[0]));
                llr_p = getArrayPointer_LLR(_n, l_p);
                _pathMetric_LLR.at(l_p) += log(1 + exp(sign*llr_p[0]));
            }

        }
        else {
            if ( contForks.at(2 * l) == 1) {
                if (_pac_code) {
                    c_m[(phi%2)] = z;
                    _arrayPointer_Polar.at(l)[phi] = z;
                    _arrayPointer_Info.at(l)[phi] = 0;
                    _stateVectors.at(l) = state;
                } else {
                    c_m[(phi%2)] = 0;
                    _arrayPointer_Info.at(l)[phi] = 0;
                }
                if (_llr_based_computation ) {
                    double *llr_p = getArrayPointer_LLR(_n, l);
                    _pathMetric_LLR.at(l) += log(1 + exp(-sign*llr_p[0]));
                }
            }
            else {
                if (_pac_code) {
                    c_m[(phi%2)] = z^1;
                    _arrayPointer_Polar.at(l)[phi] = z^1;
                    _arrayPointer_Info.at(l)[phi] = 1;
                    state.at(0) = 1;
                    _stateVectors.at(l) = state;
                } else {
                    c_m[(phi%2)] = 1;
                    _arrayPointer_Info.at(l)[phi] = 1;
                }
                if (_llr_based_computation ) {
                    double *llr_p = getArrayPointer_LLR(_n, l);
                    _pathMetric_LLR.at(l) += log(1 + exp(sign*llr_p[0]));
                }
            }
        }
    }

}

uint16_t PolarCode::findMostProbablePath(bool check) {

    uint16_t  l_p = 0;
    double p_p1 = 0;
    double p_llr = std::numeric_limits<double>::max();
    bool path_with_crc_pass = false;

    for (uint16_t l = 0; l < _list_size; ++l) {

        if (_activePath.at(l) == 0)
            continue;

        if (check && (_crc_size > 0) && (! crc_check(_arrayPointer_Info.at(l))))
            continue;

        if (check && _genie_aided && (! genie_check(_arrayPointer_Info.at(l))))
            continue;

        path_with_crc_pass = true;

        if (_llr_based_computation) {
            if (_pathMetric_LLR.at(l) < p_llr ) {
                p_llr = _pathMetric_LLR.at(l);
                l_p  = l;
            }
        }
        else {
            uint8_t * c_m = getArrayPointer_C(_n, l);
            double * p_m = getArrayPointer_P(_n, l);
            if ( p_p1 < p_m[c_m[1]]) {
                l_p = l;
                p_p1 = p_m[c_m[1]];
            }
        }
    }
    if ( path_with_crc_pass)
        return l_p;
    else
        return findMostProbablePath(false);
}


void PolarCode::create_bit_rev_order() {
    for (uint16_t i = 0; i < _block_length; ++i) {
        uint16_t to_be_reversed = i;
        _bit_rev_order.at(i) = (uint16_t) ((to_be_reversed & 1) << (_n - 1));
        for (uint8_t j = (uint8_t) (_n - 1); j; --j) {
            to_be_reversed >>= 1;
            _bit_rev_order.at(i) += (to_be_reversed & 1) << (j - 1);
        }
    }
}


double PolarCode::get_errors(double snr_db, uint8_t list_size, int max_runs, std::mt19937& generator) {

  double num_err = 0;

  uint16_t word_length = _rate_matching ? _codeword_length : _block_length;
  std::vector<uint8_t> coded_bits;
  std::vector<uint8_t> interleaved_bits;
  std::vector<uint8_t> codeword;
  std::vector<double> bpsk(word_length);
  std::vector<double> received_signal(word_length, 0);
  std::vector<uint8_t> info_bits(_info_length, 0);
  std::vector<double> noise(word_length, 0);
  std::vector<double> raw_llr(word_length);
  std::vector<double> interleaved_llr(_block_length);
  std::vector<double> llr_tmp(_block_length);
  std::vector<double> llr(_block_length);

  std::normal_distribution<double> gauss_dist(0.0f, 1.0);
  std::uniform_int_distribution<uint8_t> bin_dist(0,1);


  for (int run = 0; run < max_runs; ++run) {

    for(uint16_t i = 0; i < _info_length; ++ i ) {
      info_bits.at(i) = (uint8_t) bin_dist(generator);
    }
    for(uint16_t i = 0; i < word_length; ++ i ) {
      noise.at(i) = (double) gauss_dist(generator);
    }

    coded_bits = encode(info_bits);

    if ( _rate_matching ) {
      // sub-block interleaving
      // rate matching
      interleaved_bits = _nr_std.sub_block_interleave(coded_bits);
      codeword = _nr_std.rate_match(interleaved_bits);
    } else {
      codeword = coded_bits;
    }

    for(uint16_t i = 0; i < word_length; ++ i ) {
      bpsk.at(i) = 1.0f - 2.0f * ((double) codeword.at(i));
    }

    double nvar = std::pow(10.0f, -snr_db/10);
    double nstd = std::sqrt(nvar);

    for (uint16_t i = 0; i < word_length; ++i) {
      received_signal.at(i) = bpsk.at(i) + nstd * noise.at(i);
    }

    for (uint16_t i = 0; i < word_length; ++i) {
      raw_llr.at(i) = 2 * received_signal.at(i) / nvar;
    }

    if ( _rate_matching ) {
      // rate recovery
      // sub-block deinterleaving
      interleaved_llr = _nr_std.rate_recovery(raw_llr);
      llr_tmp = _nr_std.sub_block_deinterleave(interleaved_llr);
    } else {
      llr_tmp = raw_llr;
    }

    // move bit rev order to decoder side
    for (uint16_t i = 0; i < _block_length; ++i) {
      llr.at(i) = llr_tmp.at(_bit_rev_order.at(i));
    }

    std::vector<uint8_t> decoded_info_bits = decode_scl_llr(llr, list_size);

    bool err = false;
    for (uint16_t i = 0; i < _info_length; ++i) {
      if (info_bits.at(i) != decoded_info_bits.at(i)) {
        err = true;
        break;
      }
    }

    if (err)
      num_err++;

  } // run loop

  return num_err;

}


double PolarCode::get_bler(double snr_db, uint8_t list_size, int max_err, int max_runs, std::mt19937& generator) {
  //int max_err;
  //int max_runs = 10000;

  double bler;
  double num_err = 0;
  double num_run = 0;

  uint16_t word_length = _rate_matching ? _codeword_length : _block_length;
  std::vector<uint8_t> coded_bits;
  std::vector<uint8_t> interleaved_bits;
  std::vector<uint8_t> rate_matched_bits;
  std::vector<uint8_t> codeword;
  std::vector<double> bpsk(word_length);
  std::vector<double> received_signal(word_length, 0);
  std::vector<uint8_t> info_bits(_info_length, 0);
  std::vector<double> noise(word_length, 0);
  std::vector<double> raw_llr(word_length);
  std::vector<double> deinterleaved_llr(word_length);
  std::vector<double> interleaved_llr(_block_length);
  std::vector<double> llr_tmp(_block_length);
  std::vector<double> llr(_block_length);

  std::normal_distribution<double> gauss_dist(0.0f, 1.0);
  std::uniform_int_distribution<uint8_t> bin_dist(0,1);


  for (int run = 0; run < max_runs; ++run) {

    for(uint16_t i = 0; i < _info_length; ++ i ) {
      info_bits.at(i) = (uint8_t) bin_dist(generator);
    }
    for(uint16_t i = 0; i < word_length; ++ i ) {
      noise.at(i) = (double) gauss_dist(generator);
    }

    coded_bits = encode(info_bits);

    if ( _rate_matching ) {
      // sub-block interleaving
      // rate matching
      interleaved_bits = _nr_std.sub_block_interleave(coded_bits);
      rate_matched_bits = _nr_std.rate_match(interleaved_bits);
    } else {
      rate_matched_bits = coded_bits;
    }

    if ( _channel_interleave ) {
        codeword = _nr_std.channel_interleave(rate_matched_bits);
    } else {
        codeword = rate_matched_bits;
    }

    for(uint16_t i = 0; i < word_length; ++ i ) {
      bpsk.at(i) = 1.0f - 2.0f * ((double) codeword.at(i));
    }

    if ( num_err > max_err )
      break;

    num_run++;

    double nvar = std::pow(10.0f, -snr_db/10);
    double nstd = std::sqrt(nvar);

    for (uint16_t i = 0; i < word_length; ++i) {
      received_signal.at(i) = bpsk.at(i) + nstd * noise.at(i);
    }

    for (uint16_t i = 0; i < word_length; ++i) {
      raw_llr.at(i) = 2 * received_signal.at(i) / nvar;
    }

    if ( _channel_interleave ) {
        deinterleaved_llr = _nr_std.channel_deinterleave(raw_llr);
    } else {
        deinterleaved_llr = raw_llr;
    }

    if ( _rate_matching ) {
      // rate recovery
      // sub-block deinterleaving
      interleaved_llr = _nr_std.rate_recovery(deinterleaved_llr);
      llr_tmp = _nr_std.sub_block_deinterleave(interleaved_llr);
    } else {
      llr_tmp = raw_llr;
    }

#if 0
    std::cout << "llrs: " << std::endl;
    for (uint16_t i = 0; i < _block_length; ++i) {
        std::cout << int(i) << ":" << llr_tmp.at(i) << std::endl;
    }
    std::cout << std::endl;
#endif

    // move bit rev order to decoder side
    for (uint16_t i = 0; i < _block_length; ++i) {
      llr.at(i) = llr_tmp.at(_bit_rev_order.at(i));
    }

    std::vector<uint8_t> decoded_info_bits = decode_scl_llr(llr, list_size);

    bool err = false;
    for (uint16_t i = 0; i < _info_length; ++i) {
      if (info_bits.at(i) != decoded_info_bits.at(i)) {
        err = true;
        break;
      }
    }

    if (err)
      num_err++;

  } // run loop

  bler = num_err/num_run;

  return bler;
}
