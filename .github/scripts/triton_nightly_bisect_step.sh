#!/usr/bin/env bash
# One `git bisect run` step for the Triton nightly: rebuild aiter + Triton at
# the commit git checked out and run only the tests that failed last night.
#
#   exit 0   -> good commit
#   exit 1   -> bad commit
#   exit 125 -> untestable (build broke here), git skips it
#
# Usage: triton_nightly_bisect_step.sh "<space separated test paths>"

set -uo pipefail

TESTS="${1:-}"
if [[ -z "${TESTS// /}" ]]; then
    echo "no failing tests passed to the bisect step" >&2
    exit 125
fi

echo "=== bisect step at $(git rev-parse --short HEAD): $(git log -1 --format=%s) ==="

git submodule update --init --recursive >/dev/null 2>&1 || true

if ! docker exec -w /workspace triton_test ./.github/scripts/build_aiter_triton.sh; then
    echo "build failed here — untestable, skipping this commit" >&2
    exit 125
fi

if docker exec -w /workspace triton_test pytest -q ${TESTS}; then
    echo "=== good ==="
    exit 0
fi
echo "=== bad ==="
exit 1
