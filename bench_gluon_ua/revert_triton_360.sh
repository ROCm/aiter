#!/usr/bin/env bash
# Revert Triton to the 3.6.0 baseline saved during the gfx950 tuning work.
# The wheel lives in bench_gluon_ua/snapshots/ (gitignored, local-only).
set -euo pipefail

WHEEL=$(ls "$(dirname "$0")"/snapshots/triton-3.6.0-*.whl 2>/dev/null | head -1)
if [ -z "${WHEEL}" ]; then
    echo "ERROR: 3.6.0 wheel not found under bench_gluon_ua/snapshots/."
    echo "       Re-download with: pip download --no-deps triton==3.6.0 -d bench_gluon_ua/snapshots/"
    exit 1
fi

echo "Reverting Triton to 3.6.0 from: ${WHEEL}"
pip uninstall -y triton
pip install --no-deps --force-reinstall "${WHEEL}"
python -c "import triton; print('restored triton', triton.__version__)"
