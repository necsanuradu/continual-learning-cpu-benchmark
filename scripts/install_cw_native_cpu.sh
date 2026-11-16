#!/usr/bin/env bash
set -euo pipefail
# Complete public original source stays only on disposable runner; not committed.
git clone --no-checkout --filter=blob:none https://github.com/awarelab/continual_world.git cw-original
git -C cw-original checkout --detach 73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa
test "$(git -C cw-original rev-parse HEAD)" = 73f63bb4fa0b5d00bda973e20dfb783bfcf1b8aa
python -m pip install 'tensorflow-cpu==2.10.1' 'numpy==1.23.5' 'gym==0.24.1' 'pandas==1.5.3' 'matplotlib==3.7.5' 'seaborn==0.12.2'
python -m pip install 'Cython==0.29.37' 'mujoco-py==2.0.2.8'
git clone --no-checkout --filter=blob:none https://github.com/Farama-Foundation/Metaworld.git cw-metaworld
git -C cw-metaworld checkout --detach 0875192baaa91c43523708f55866d98eaf3facaf
test "$(git -C cw-metaworld rev-parse HEAD)" = 0875192baaa91c43523708f55866d98eaf3facaf
python -m pip install --no-deps -e ./cw-metaworld
# Do NOT install or re-publish original unlicensed source.
# Python imports via sys.path on ephemeral runner only.
python -m pip check
echo "CW_ORIGINAL_UPSTREAM_INSTALL_PASS"
