#!/usr/bin/env bash
set -euo pipefail
# This script is used AFTER installing the already-qualified original
# MuJoCo 2.0 / MetaWorld / Python 3.8/JAX 0.3.17 CPU environment.
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.1.2+cpu' 'torchvision==0.16.2+cpu'
python -m pip install \
  'transformers==4.30.2' 'huggingface-hub==0.16.4' \
  'tokenizers==0.13.3' 'scikit-learn==1.0.2' \
  'nltk==3.9.1' 'h5py==3.10.0' 'pandas==2.0.3' \
  'sentencepiece==0.2.0' 'termcolor==2.4.0' \
  'click==8.1.8' 'pyyaml==6.0.2'
python -m pip install --no-deps 'sentence-transformers==2.2.2' 'd4rl==1.1'
echo NEVO_NATIVE_PUBLIC_DEPENDENCIES_INSTALLED
python -m pip freeze | grep -Ei '^(torch|torchvision|transformers|huggingface-hub|sentence-transformers|tokenizers|h5py|d4rl|gym|mujoco-py|dm-control|dm-env|jax|jaxlib|flax)=='
