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

# Only a real test failure (pytest exit 1) marks a commit bad. Collection
# errors (4) or "no tests ran" (5) mean the commit cannot answer the
# question — for instance the failing test does not exist yet — so hand
# those to git as 125 (skip) instead of blaming them.
docker exec -w /workspace triton_test pytest -q ${TESTS}
rc=$?
case "${rc}" in
    0) echo "=== good ==="; exit 0 ;;
    1) echo "=== bad ==="; exit 1 ;;
    *) echo "pytest could not evaluate this commit (exit ${rc}) — skipping" >&2; exit 125 ;;
esac
