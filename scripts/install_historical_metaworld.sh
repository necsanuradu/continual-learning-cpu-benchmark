#!/usr/bin/env bash
set -euo pipefail
competitor="$1"
bash scripts/install_cpu.sh "$competitor"
python -m pip install 'gym==0.24.1' 'Cython==0.29.37' 'glfw==2.7.0' 'imageio==2.35.1'
if test "$competitor" = nevo; then
 python -m pip install 'mujoco-py==2.0.2.8'
else
 python -m pip install 'mujoco-py==2.1.2.14'
fi
git clone --no-checkout --filter=blob:none https://github.com/Farama-Foundation/Metaworld.git public-metaworld
git -C public-metaworld checkout --detach 0875192baaa91c43523708f55866d98eaf3facaf
test "$(git -C public-metaworld rev-parse HEAD)" = 0875192baaa91c43523708f55866d98eaf3facaf
python -m pip install --no-deps -e ./public-metaworld
echo PINNED_METAWORLD=0875192baaa91c43523708f55866d98eaf3facaf
python -m pip check
