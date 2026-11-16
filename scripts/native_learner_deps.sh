#!/usr/bin/env bash
set -euo pipefail
competitor="$1"
bash scripts/install_cpu.sh "$competitor"
python -m pip install 'gym==0.24.1' 'pyyaml==6.0.2'
if test "$competitor" = nevo; then
    python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.1.2'
    python -m pip install 'sentence-transformers==2.2.2' 'transformers==4.46.3' 'huggingface-hub==0.24.6' 'scikit-learn==1.0.2' 'tokenizers==0.20.3'
else
    python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.4.1'
    python -m pip install 'sentence-transformers==3.1.1' 'transformers==4.44.2' 'huggingface-hub==0.24.6' 'scikit-learn==1.5.2' 'tokenizers==0.19.1'
    python -m pip install 'Cython==0.29.37' 'mujoco-py==2.1.2.14'
    git clone --no-checkout --filter=blob:none 'https://github.com/Farama-Foundation/Metaworld.git' public-metaworld
    git -C public-metaworld checkout --detach 0875192baaa91c43523708f55866d98eaf3facaf
    test "$(git -C public-metaworld rev-parse HEAD)" = 0875192baaa91c43523708f55866d98eaf3facaf
    python -m pip install --no-deps -e ./public-metaworld
fi
python -m pip check
