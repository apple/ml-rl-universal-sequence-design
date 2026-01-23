# ml-rl-universal-sequence-design

This software project accompanies the research paper, [A Reinforcement Learning Framework for
Universal Sequence Design in Polar Codes]().

This repo contains the source code used in the development of the work described in the paper.

## Documentation

The source code provides a locally runnable training process for demonstration only.  It does not include cluster deployment code used in the development of the sequence.

## Getting Started

Ensure you have these python virtual environment tools installed (pyenv, pyenv-virtualenv).

- Set up virtual environment
```
bash scripts/rebuild_venv.sh
```

- Set up environment
```
source setup_env.sh
```

- Test locally using scripts in `./run_scripts`
```
# example
bash run_scripts/test_run.sh
bash run_scripts/iter_learn.sh
```
