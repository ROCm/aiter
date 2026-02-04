#!/usr/bin/env bash
set -euo pipefail

MULTIGPU=${MULTIGPU:-FALSE}
SHARD_TOTAL=5
SHARD_IDX=${SHARD_IDX:-0}

files=()
failedFiles=()

testFailed=false

if [[ "$MULTIGPU" == "TRUE" ]]; then
    # Recursively find all files under op_tests/multigpu_tests
    mapfile -t files < <(find op_tests/multigpu_tests -type f -name "*.py" | sort)
else
    # Recursively find all files under op_tests, excluding op_tests/multigpu_tests
    mapfile -t files < <(find op_tests -maxdepth 1 -type f -name "*.py" | sort)
fi

skip_tests=(
    "op_tests/multigpu_tests/test_dispatch_combine.py"
    "op_tests/multigpu_tests/test_communication.py"
    "op_tests/multigpu_tests/test_mori_all2all.py"
    "op_tests/multigpu_tests/triton_test/test_reduce_scatter_all_gather.py"
    "op_tests/multigpu_tests/triton_test/test_fused_rs_rmsnorm_quant_ag.py"
)

# tests are split into 5 shards, each shard is called a shard
sharded_files=()
for idx in "${!files[@]}"; do
    # modulo operation is used to determine which shard the test belongs to
    if (( idx % SHARD_TOTAL == SHARD_IDX )); then
        sharded_files+=("${files[$idx]}")
    fi
done

echo "Running ${sharded_files[@]} in shard $SHARD_IDX of $SHARD_TOTAL."

for file in "${sharded_files[@]}"; do
    # Print a clear separator and test file name for readability
    {
        echo
        echo "============================================================"
        echo "Running test: $file (shard: $SHARD_IDX/$SHARD_TOTAL)"
        echo "============================================================"
        echo
    } | tee -a latest_test.log
    if [[ " ${skip_tests[@]} " =~ " $file " ]]; then
        {
            echo "Skipping test: $file"
            echo "============================================================"
            echo
        } | tee -a latest_test.log
        continue
    fi
    # Run each test file with a 60-minute timeout, output to latest_test.log
    if ! timeout 60m python3 "$file" 2>&1 | tee -a latest_test.log; then
        {
            echo
            echo "--------------------"
            echo "❌ Test failed: $file"
            echo "--------------------"
            echo
        } | tee -a latest_test.log
        testFailed=true
        failedFiles+=("$file")
    else
        {
            echo
            echo "--------------------"
            echo "✅ Test passed: $file"
            echo "--------------------"
            echo
        } | tee -a latest_test.log
    fi
done

if [ "$testFailed" = true ]; then
    {
        echo "Failed test files (shard $SHARD_IDX):"
        for f in "${failedFiles[@]}"; do
            echo "  $f"
        done
    } | tee -a latest_test.log
    exit 1
else
    echo "All tests passed in shard $SHARD_IDX." | tee -a latest_test.log
    exit 0
fi
