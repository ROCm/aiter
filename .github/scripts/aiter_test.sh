#!/usr/bin/env bash
set -euo pipefail

MULTIGPU=${MULTIGPU:-FALSE}

MULTIGPU_TESTS=("test_communication.py" "test_custom_allreduce.py")

files=()

if [[ "$MULTIGPU" == "TRUE" ]]; then
    mapfile -t files < <(find op_tests/multigpu_tests -type f \( -name "${MULTIGPU_TESTS[0]}" -o -name "${MULTIGPU_TESTS[1]}" \))
else
    mapfile -t files < <(find op_tests -maxdepth 1 -type f -name "*.py")
fi

for file in "${files[@]}"; do
    # Run each test file with a 30-minute timeout, output to latest_test.log
    echo "Running $file..."
    timeout 30m python3 "$file" 2>&1 | tee -a latest_test.log
done
