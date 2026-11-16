#!/usr/bin/env bash
set -euo pipefail

python -V
python -m pip --version
python -m pip install --upgrade 'pip<25' 'setuptools<75' 'wheel<0.46'

case "${1:-}" in
  nevo)
    # Preserve original Nevo JAX and Python versions; use a CPU-only JAXLIB wheel.
    python -m pip install --find-links=https://storage.googleapis.com/jax-releases/jax_releases.html \
      'jax==0.3.17' 'jaxlib==0.3.15' 'numpy==1.24.3' 'scipy==1.10.1' \
      'flax==0.6.2' 'optax==0.1.4' 'chex==0.1.5' \
      'tensorflow-probability==0.18.0' 'ml-collections==0.1.1' 'wandb==0.19.6'
    ;;
  wave)
    # Preserve original WAVE JAX/JAXLIB versions, without CUDA-only extras.
    python -m pip install \
      'jax==0.4.32' 'jaxlib==0.4.32' 'numpy==1.26.4' 'scipy==1.14.1' \
      'flax==0.9.0' 'optax==0.1.9' 'tensorflow-probability==0.24.0' \
      'minisom==2.3.5' 'ml-collections==0.1.1' 'wandb==0.18.0'
    ;;
  *)
    echo 'Unknown competitor' >&2
    exit 2
    ;;
esac
python -m pip check
python -m pip freeze | grep -Ei '^(jax|jaxlib|flax|numpy|scipy|optax|tensorflow-probability|wandb)==' || true
