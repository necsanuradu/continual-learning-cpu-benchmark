#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  nevo)
    REPO='https://github.com/yeshenpy/Nevo-CRL.git'
    COMMIT='fe0456e1ae5382213d12421270209eb3f47a9a0f'
    ;;
  wave)
    REPO='https://github.com/ggtz/WAVE.git'
    COMMIT='7814979cb1f8f4324fc3a48881591b1d0c7d7989'
    ;;
  *)
    echo 'Unknown competitor' >&2
    exit 2
    ;;
esac

git clone --no-checkout --filter=blob:none "$REPO" competitor-src
git -C competitor-src checkout --detach "$COMMIT"
ACTUAL="$(git -C competitor-src rev-parse HEAD)"
test "$ACTUAL" = "$COMMIT"
printf 'COMPETITOR=%s\nSOURCE_URL=%s\nPINNED_COMMIT=%s\nACTUAL_COMMIT=%s\n' "$1" "$REPO" "$COMMIT" "$ACTUAL"
