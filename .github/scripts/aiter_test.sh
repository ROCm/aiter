#!/usr/bin/env bash
set -euo pipefail

MULTIGPU=${MULTIGPU:-FALSE}

files=()

if [[ "$MULTIGPU" == "TRUE" ]]; then
    # Recursively find all files under op_tests/multigpu_tests
    mapfile -t files < <(find op_tests/multigpu_tests -type f)
else
    # Recursively find all files under op_tests, excluding op_tests/multigpu_tests
    mapfile -t files < <(find op_tests -type f ! -path "op_tests/multigpu_tests/*")
fi

for file in "${files[@]}"; do
    # Run each test file with a 30-minute timeout, output to latest_test.log
    timeout 30m bash -c "python3 \"$file\" 2>&1 | tee latest_test.log"
done


