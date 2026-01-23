//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <vector>
#include <random>
#include <thread>
#include <numeric>
#include "PolarCode.h"

namespace py = pybind11;

inline uint16_t log2(uint16_t N) {
  uint16_t n = 0;
  while (N >>= 1) ++n;
  return n;
}

class PolarSimulator {

  public:
    PolarSimulator() : generator(dev()) {}
    PolarSimulator(int seed) : generator(seed) {}

    double get_bler(int N, int list_size, double snr_db, int max_err, int max_runs,
                    py::array_t<int> frozen_bits, py::array_t<int> crc_coeffs, bool genie_aided, py::array_t<int> conv_coeffs);
    double get_bler_rm(int A, int E, int list_size, double snr_db, int max_err, int max_runs,
                    py::array_t<int> sequence, py::array_t<int> crc_coeffs, bool genie_aided, bool channel_interleave, bool downlink, py::array_t<int> conv_coeffs);

  private:
    std::random_device dev;
    std::mt19937 generator;
};

double PolarSimulator::get_bler(int N, int list_size, double snr_db, int max_err, int max_runs,
                py::array_t<int> frozen_bits, py::array_t<int> crc_coeffs, bool genie_aided, py::array_t<int> conv_coeffs) {

  if (frozen_bits.ndim() != 1 || crc_coeffs.ndim() != 1)
    throw std::runtime_error("Array ndim != 1");

  py::ssize_t size = frozen_bits.shape(0);
  std::vector<uint8_t> frozen_bits_(size);

  for (py::ssize_t i = 0; i < size; i++)
    frozen_bits_.at(i) = *frozen_bits.data(i);

  size = crc_coeffs.shape(0);
  std::vector<uint8_t> crc_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    crc_coeffs_.at(i) = *crc_coeffs.data(i);

  size = conv_coeffs.shape(0);
  std::vector<uint8_t> conv_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    conv_coeffs_.at(i) = *conv_coeffs.data(i);

  uint8_t n = log2(N);
  uint16_t k = count(frozen_bits_.begin(), frozen_bits_.end(), 0);
  uint16_t crc_size = crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0;
  uint16_t info_length = k - crc_size;

#if 0
  std::cout << "n " << int(n) << std::endl;
  std::cout << "k " << k << std::endl;
  std::cout << "crc_size " << crc_size << std::endl;
  std::cout << "info_length " << info_length << std::endl;
  std::cout << "crc_coeffs: ";
  for (py::ssize_t i = 0; i < size; i++)
    std::cout << crc_coeffs.at(i) << ",";
  std::cout << std::endl;
  std::cout << "genie_aided " << genie_aided << std::endl;
#endif

  if (k < (crc_size + 1))
    throw std::runtime_error("K must be > crc_size");

  if (genie_aided && crc_size > 0)
    throw std::runtime_error("either use crc or genie aided, not both");

  PolarCode polar_code(n, info_length, frozen_bits_, crc_coeffs_, genie_aided, conv_coeffs_);

  return polar_code.get_bler(snr_db, list_size, max_err, max_runs, generator);

}

double PolarSimulator::get_bler_rm(int A, int E, int list_size, double snr_db, int max_err, int max_runs,
                py::array_t<int> sequence, py::array_t<int> crc_coeffs, bool genie_aided, bool channel_interleave, bool downlink, py::array_t<int> conv_coeffs) {

  if (sequence.ndim() != 1 || crc_coeffs.ndim() != 1)
    throw std::runtime_error("Array ndim != 1");

  py::ssize_t size = sequence.shape(0);
  std::vector<uint16_t> sequence_(size);

  for (py::ssize_t i = 0; i < size; i++)
    sequence_.at(i) = *sequence.data(i);

  size = crc_coeffs.shape(0);
  std::vector<uint8_t> crc_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    crc_coeffs_.at(i) = *crc_coeffs.data(i);

  //uint8_t n = log2(N);
  //uint16_t k = count(frozen_bits_.begin(), frozen_bits_.end(), 0);
  uint16_t crc_size = crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0;
  //uint16_t info_length = k - crc_size;

  size = conv_coeffs.shape(0);
  std::vector<uint8_t> conv_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    conv_coeffs_.at(i) = *conv_coeffs.data(i);

#if 0
  std::cout << "n " << int(n) << std::endl;
  std::cout << "k " << k << std::endl;
  std::cout << "crc_size " << crc_size << std::endl;
  std::cout << "info_length " << info_length << std::endl;
  std::cout << "crc_coeffs: ";
  for (py::ssize_t i = 0; i < size; i++)
    std::cout << crc_coeffs.at(i) << ",";
  std::cout << std::endl;
  std::cout << "genie_aided " << genie_aided << std::endl;
#endif

  //if (k < (crc_size + 1))
  //  throw std::runtime_error("K must be > crc_size");

  if (A + crc_size > E)
    throw std::runtime_error("error K > E");

  if (genie_aided && crc_size > 0)
    throw std::runtime_error("either use crc or genie aided, not both");

  PolarCode polar_code(A, E, sequence_, crc_coeffs_, genie_aided, channel_interleave, downlink, conv_coeffs_);

  return polar_code.get_bler(snr_db, list_size, max_err, max_runs, generator);

}



class PolarSimWorker {

  public:
    PolarSimWorker(int runs, int& errors) : generator(std::random_device()()), runs(runs), errors(errors) {}
    PolarSimWorker(int seed, int runs, int& errors) : generator(seed), runs(runs), errors(errors) {}

    void set_params(int n, int info_length, double snr_db, int list_size,
                    std::vector<uint8_t>& frozen_bits,
                    std::vector<uint8_t>& crc_coeffs,
                    bool genie_aided,
                    std::vector<uint8_t>& conv_coeffs) {
      this->n = n;
      this->info_length = info_length;
      this->snr_db = snr_db;
      this->list_size = list_size;
      this->frozen_bits = frozen_bits;
      this->crc_coeffs = crc_coeffs;
      this->conv_coeffs = conv_coeffs;
      this->genie_aided = genie_aided;
      this->rate_matching = false;
      this->channel_interleave = false;
      this->downlink = false;
    }
    void set_params_rm(int A, int E, double snr_db, int list_size,
                    std::vector<uint16_t>& sequence,
                    std::vector<uint8_t>& crc_coeffs,
                    bool genie_aided,
                    bool channel_interleave,
                    bool downlink,
                    std::vector<uint8_t>& conv_coeffs) {
      this->info_length = A;
      this->codeword_length = E;
      this->snr_db = snr_db;
      this->list_size = list_size;
      this->sequence = sequence;
      this->crc_coeffs = crc_coeffs;
      this->conv_coeffs = conv_coeffs;
      this->genie_aided = genie_aided;
      this->rate_matching = true;
      this->channel_interleave = channel_interleave;
      this->downlink = downlink;
    }

    void operator() () {
      if (rate_matching) {
        PolarCode polar_code(info_length, codeword_length, sequence, crc_coeffs, genie_aided, channel_interleave, downlink, conv_coeffs);
        errors = polar_code.get_errors(snr_db, list_size, runs, generator);
      } else {
        PolarCode polar_code(n, info_length, frozen_bits, crc_coeffs, genie_aided, conv_coeffs);
        errors = polar_code.get_errors(snr_db, list_size, runs, generator);
      }
    }

  private:
    int n;
    int info_length;
    int codeword_length;
    double snr_db;
    int list_size;
    std::vector<uint16_t> sequence;
    std::vector<uint8_t> frozen_bits;
    std::vector<uint8_t> crc_coeffs;
    std::vector<uint8_t> conv_coeffs;
    bool genie_aided;
    bool rate_matching;
    bool channel_interleave;
    bool downlink;
    // static parameters
    std::mt19937 generator;
    int runs;
    int& errors;
};


class MultiThreadPolarSimulator {

  public:
    MultiThreadPolarSimulator() : runs_per_thread(1000), num_threads(8) { initialize(false); }
    MultiThreadPolarSimulator(int seed, int runs_per_thread, int num_threads) : master_seed(seed), runs_per_thread(runs_per_thread), num_threads(num_threads) { initialize(true); }

    void initialize(bool set_seed) {
      int num_cpus = std::thread::hardware_concurrency(); // Get the number of available threads
      std::cout << "Using " << num_threads << " of " << num_cpus << " CPU cores" << std::endl;
      errors_vec.resize(num_threads);
      std::fill(errors_vec.begin(), errors_vec.end(), 0);
      for (int i = 0; i < num_threads; ++i) {
          if (set_seed) {
            int seed = master_seed * num_threads + i;
            workers.emplace_back(seed, runs_per_thread, errors_vec.at(i));
          } else {
            workers.emplace_back(runs_per_thread, errors_vec.at(i));
          }
      }
    }

    double get_bler(int N, int list_size, double snr_db, int max_err, int max_runs,
                    py::array_t<int> frozen_bits, py::array_t<int> crc_coeffs, bool genie_aided, py::array_t<int> conv_coeffs);
    double get_bler_rm(int A, int E, int list_size, double snr_db, int max_err, int max_runs,
                    py::array_t<int> sequence, py::array_t<int> crc_coeffs, bool genie_aided, bool channel_interleave, bool downlink, py::array_t<int> conv_coeffs);

  private:
    int master_seed;
    int runs_per_thread;
    int num_threads;
    std::vector<PolarSimWorker> workers;
    std::vector<int> errors_vec;
};

double MultiThreadPolarSimulator::get_bler(int N, int list_size, double snr_db, int max_err, int max_runs,
                py::array_t<int> frozen_bits, py::array_t<int> crc_coeffs, bool genie_aided, py::array_t<int> conv_coeffs) {

  if (frozen_bits.ndim() != 1 || crc_coeffs.ndim() != 1)
    throw std::runtime_error("Array ndim != 1");

  py::ssize_t size = frozen_bits.shape(0);
  std::vector<uint8_t> frozen_bits_(size);

  for (py::ssize_t i = 0; i < size; i++)
    frozen_bits_.at(i) = *frozen_bits.data(i);

  size = crc_coeffs.shape(0);
  std::vector<uint8_t> crc_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    crc_coeffs_.at(i) = *crc_coeffs.data(i);

  size = conv_coeffs.shape(0);
  std::vector<uint8_t> conv_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    conv_coeffs_.at(i) = *conv_coeffs.data(i);

  uint8_t n = log2(N);
  uint16_t k = count(frozen_bits_.begin(), frozen_bits_.end(), 0);
  uint16_t crc_size = crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0;
  uint16_t info_length = k - crc_size;

  if (k < (crc_size + 1))
    throw std::runtime_error("K must be > crc_size");

  if (genie_aided && crc_size > 0)
    throw std::runtime_error("either use crc or genie aided, not both");

#if 0
  std::cout << "n " << int(n) << std::endl;
  std::cout << "k " << k << std::endl;
  std::cout << "crc_size " << crc_size << std::endl;
  std::cout << "info_length " << info_length << std::endl;
  std::cout << "crc_coeffs: ";
  for (py::ssize_t i = 0; i < size; i++)
    std::cout << crc_coeffs.at(i) << ",";
  std::cout << std::endl;
  std::cout << "genie_aided " << genie_aided << std::endl;
#endif

  std::vector<std::thread> threads;

  double num_errors = 0;
  double num_tasks_completed = 0;
  double runs_completed = 0;

  while (num_errors < max_err && runs_completed < max_runs) {

    for (int i = 0; i < num_threads; ++i) {
      workers.at(i).set_params(n, info_length, snr_db, list_size, frozen_bits_, crc_coeffs_, genie_aided, conv_coeffs_);
      threads.emplace_back(workers.at(i)); // Create and start threads
    }

    for (auto& thread : threads) {
      thread.join(); // Wait for threads to finish
    }

    threads.clear(); // Kill all threads

    num_errors = std::accumulate(errors_vec.begin(), errors_vec.end(), num_errors);
    // reset vector
    std::fill(errors_vec.begin(), errors_vec.end(), 0);
    // update counter
    num_tasks_completed += num_threads;
    runs_completed = num_tasks_completed * runs_per_thread;

    //std::cout << "num_errors " << num_errors << std::endl;
    //std::cout << "runs_completed " << runs_completed << std::endl;

  }

  double bler = num_errors / (num_tasks_completed * runs_per_thread);

  return bler;

}

double MultiThreadPolarSimulator::get_bler_rm(int A, int E, int list_size, double snr_db, int max_err, int max_runs,
                py::array_t<int> sequence, py::array_t<int> crc_coeffs,
                bool genie_aided, bool channel_interleave, bool downlink, py::array_t<int> conv_coeffs) {

  if (sequence.ndim() != 1 || crc_coeffs.ndim() != 1)
    throw std::runtime_error("Array ndim != 1");

  py::ssize_t size = sequence.shape(0);
  std::vector<uint16_t> sequence_(size);

  for (py::ssize_t i = 0; i < size; i++)
    sequence_.at(i) = *sequence.data(i);

  size = crc_coeffs.shape(0);
  std::vector<uint8_t> crc_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    crc_coeffs_.at(i) = *crc_coeffs.data(i);

  //uint8_t n = log2(N);
  //uint16_t k = count(frozen_bits_.begin(), frozen_bits_.end(), 0);
  uint16_t crc_size = crc_coeffs.size() > 0 ? crc_coeffs.size() - 1 : 0;
  //uint16_t info_length = k - crc_size;

  size = conv_coeffs.shape(0);
  std::vector<uint8_t> conv_coeffs_(size);

  for (py::ssize_t i = 0; i < size; i++)
    conv_coeffs_.at(i) = *conv_coeffs.data(i);

  //if (k < (crc_size + 1))
  //  throw std::runtime_error("K must be > crc_size");

  if (A + crc_size > E)
    throw std::runtime_error("error K > E");

  if (genie_aided && crc_size > 0)
    throw std::runtime_error("either use crc or genie aided, not both");

#if 0
  std::cout << "n " << int(n) << std::endl;
  std::cout << "k " << k << std::endl;
  std::cout << "crc_size " << crc_size << std::endl;
  std::cout << "info_length " << info_length << std::endl;
  std::cout << "crc_coeffs: ";
  for (py::ssize_t i = 0; i < size; i++)
    std::cout << crc_coeffs.at(i) << ",";
  std::cout << std::endl;
  std::cout << "genie_aided " << genie_aided << std::endl;
#endif

  std::vector<std::thread> threads;

  double num_errors = 0;
  double num_tasks_completed = 0;
  double runs_completed = 0;

  while (num_errors < max_err && runs_completed < max_runs) {

    for (int i = 0; i < num_threads; ++i) {
      workers.at(i).set_params_rm(A, E, snr_db, list_size, sequence_, crc_coeffs_, genie_aided, channel_interleave, downlink, conv_coeffs_);
      threads.emplace_back(workers.at(i)); // Create and start threads
    }

    for (auto& thread : threads) {
      thread.join(); // Wait for threads to finish
    }

    threads.clear(); // Kill all threads

    num_errors = std::accumulate(errors_vec.begin(), errors_vec.end(), num_errors);
    // reset vector
    std::fill(errors_vec.begin(), errors_vec.end(), 0);
    // update counter
    num_tasks_completed += num_threads;
    runs_completed = num_tasks_completed * runs_per_thread;

    //std::cout << "num_errors " << num_errors << std::endl;
    //std::cout << "runs_completed " << runs_completed << std::endl;

  }

  double bler = num_errors / (num_tasks_completed * runs_per_thread);

  return bler;

}




 PYBIND11_MODULE(polar, m) {

    py::class_<PolarSimulator>(m, "PolarSimulator")
      .def(py::init<>()) // default constr
      .def(py::init<int>()) // constr with seed
      .def("get_bler", &PolarSimulator::get_bler)
      .def("get_bler_rm", &PolarSimulator::get_bler_rm);

    py::class_<MultiThreadPolarSimulator>(m, "MultiThreadPolarSimulator")
      .def(py::init<>()) // default constr
      .def(py::init<int,int,int>()) // constr with seed, runs_per_thread, num_threads
      .def("get_bler", &MultiThreadPolarSimulator::get_bler)
      .def("get_bler_rm", &MultiThreadPolarSimulator::get_bler_rm);
}

