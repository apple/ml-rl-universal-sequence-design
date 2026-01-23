#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

pyenv activate oss_py310
# local editable install
pip install -e univ_seq_pkg
pip install -e sb3_pkg
# start up script
export PYTHONSTARTUP=$(pwd)/startup.py
# set root dir
export PROJ_ROOT=$(pwd)
# create dirs
mkdir data 2> /dev/null
mkdir models 2> /dev/null
mkdir logs 2> /dev/null

