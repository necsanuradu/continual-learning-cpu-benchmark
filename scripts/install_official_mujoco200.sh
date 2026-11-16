#!/usr/bin/env bash
set -euo pipefail
# Public, original-physics, non-training diagnostic. Do not use private assets.
# Download both files from their official Roboti LLC distribution.
root="${HOME}/.mujoco"
mkdir -p "$root"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl --fail --location --retry 2 --silent --show-error \
  'https://www.roboti.us/download/mujoco200_linux.zip' \
  --output "$tmp/mujoco200_linux.zip"
curl --fail --location --retry 2 --silent --show-error \
  'https://www.roboti.us/file/mjkey.txt' \
  --output "$tmp/mjkey.txt"
test -s "$tmp/mujoco200_linux.zip"
test -s "$tmp/mjkey.txt"
echo 'ORIGINAL_ENGINE_DISTRIBUTION=https://www.roboti.us/download/mujoco200_linux.zip'
echo 'ORIGINAL_ENGINE_KEY=https://www.roboti.us/file/mjkey.txt'
sha256sum "$tmp/mujoco200_linux.zip" "$tmp/mjkey.txt"
unzip -tq "$tmp/mujoco200_linux.zip" >/dev/null
unzip -q "$tmp/mujoco200_linux.zip" -d "$root"
if test -d "$root/mujoco200_linux"; then
  mv "$root/mujoco200_linux" "$root/mujoco200"
fi
test -s "$root/mujoco200/bin/libmujoco200.so" || test -s "$root/mujoco200/bin/libmujoco200nogl.so"
install -m 600 "$tmp/mjkey.txt" "$root/mjkey.txt"
install -m 600 "$tmp/mjkey.txt" "$root/mujoco200/bin/mjkey.txt"
printf 'ORIGINAL_ENGINE_INSTALL=%s\n' "$root/mujoco200"
find "$root/mujoco200/bin" -maxdepth 1 -name '*mujoco*' -printf 'BIN=%f\n'
if test -n "${GITHUB_ENV:-}"; then
  echo "LD_LIBRARY_PATH=$root/mujoco200/bin:${LD_LIBRARY_PATH:-}" >> "$GITHUB_ENV"
fi
