#!/bin/bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

set -e # exit on error

VENV_NAME='oss_py310'
PYTHON_VER=3.10.16

# prompt user first
if pyenv virtualenvs | grep -q $VENV_NAME; then
  read -p "This will delete the virtual env '$VENV_NAME'.  Continue? [y|n] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "aborting"
    exit
  fi
fi

set -x

pyenv virtualenv-delete -f $VENV_NAME || true
pyenv virtualenv $PYTHON_VER $VENV_NAME
#pyenv activate $VENV_NAME # does not work in bash script
source ${PYENV_ROOT}/versions/${VENV_NAME}/bin/activate

#pip install --upgrade pip
pip install torch==2.6.0
pip install torch_geometric
pip install numpy scipy
pip install pybind11
pip install jsonargparse
pip install gymnasium
pip install matplotlib
pip install tensorboard
pip install pandas

pip list

