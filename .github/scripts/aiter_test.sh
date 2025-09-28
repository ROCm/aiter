#!/usr/bin/env bash
set -euo pipefail

MULTIGPU=${MULTIGPU:-FALSE}

files=()
failedFiles=()

testFailed=false

if [[ "$MULTIGPU" == "TRUE" ]]; then
    # Recursively find all files under op_tests/multigpu_tests
    mapfile -t files < <(find op_tests/multigpu_tests -type f -name "*.py")
else
    # Recursively find all files under op_tests, excluding op_tests/multigpu_tests
    mapfile -t files < <(find op_tests -maxdepth 1 -type f -name "*.py")
fi

for file in "${files[@]}"; do
    # Print a clear separator and test file name for readability
    if [ "$file" = "op_tests/multigpu_tests/test_dispatch_combine.py" ]; then
        echo -e "\n============================================================"
        echo -e "Skipping test: $file"
        echo -e "============================================================\n"
        continue
    fi
    echo -e "\n============================================================"
    echo -e "Running test: $file"
    echo -e "============================================================\n"
    # Run each test file with a 60-minute timeout, output to latest_test.log
    if ! timeout 60m python3 "$file" 2>&1 | tee -a latest_test.log; then
        echo -e "\n--------------------"
        echo -e "❌ Test failed: $file"
        echo -e "--------------------\n"
        testFailed=true
        failedFiles+=("$file")
    else
        echo -e "\n--------------------"
        echo -e "✅ Test passed: $file"
        echo -e "--------------------\n"
    fi
done

if [ "$testFailed" = true ]; then
    echo "Failed test files:"
    for f in "${failedFiles[@]}"; do
        echo "  $f"
    done
    exit 1
else
    echo "All tests passed."
    exit 0
fi