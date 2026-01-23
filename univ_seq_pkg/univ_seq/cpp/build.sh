#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

BUILD=${1:-release}

set -x

OS_PLATFORM=$(uname -s)

if [ "$OS_PLATFORM" = "Linux" ]; then
  c++ -O3 -Wall -shared -std=c++11 -pthread -fPIC $(python3 -m pybind11 --includes) PolarCode.cpp polar.cpp -o polar$(python3-config --extension-suffix)
elif [ "$OS_PLATFORM" = "Darwin" ]; then
  if [ "$BUILD" = "release" ]; then
    clang++ -O3 -Wall -shared -std=c++11 -stdlib=libc++ -undefined dynamic_lookup $(python3 -m pybind11 --includes) $(python3-config --ldflags) PolarCode.cpp polar.cpp -o polar$(python3-config --extension-suffix)
  elif [ "$BUILD" = "debug" ]; then
    clang++ -g -Wall -shared -std=c++11 -stdlib=libc++ -undefined dynamic_lookup $(python3 -m pybind11 --includes) $(python3-config --ldflags) PolarCode.cpp polar.cpp -o polar$(python3-config --extension-suffix)
  else
    echo "Error: release or debug"
  fi
else
  echo "Error: unsupported platform"
fi
