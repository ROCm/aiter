#!/bin/bash

echo "Verifying BlockPingpong is disabled and checking LDS usage pattern"
echo "====================================================================="

export TRITON_HIP_USE_BLOCK_PINGPONG=0

for stages in 2 3 4 5; do
    echo ""
    echo "Testing num_stages=$stages with TRITON_HIP_USE_BLOCK_PINGPONG=0"
    echo "----------------------------------------------------------------"

    # Update config
    sed -i "s/\"num_stages\": [0-9]/\"num_stages\": $stages/" aiter/ops/triton/attention/fav3_sage.py

    # Clear cache
    rm -rf ~/.triton/cache/*

    # Run benchmark (this will show the compiler debug output)
    echo "Running compilation..."
    HIP_VISIBLE_DEVICES=0 python op_tests/op_benchmarks/triton/bench_fav3_sage.py -b 1 -sq 75600 -hq 8 -d 128 2>&1 | grep -E "COMPILER DEBUG|BLOCK_PINGPONG|Error|Traceback|LDS|shared" | head -5

    # Check if compilation succeeded
    if [ $? -ne 0 ]; then
        echo "  ⚠️  Compilation may have failed"
    fi

    # Find and report LDS usage
    json_file=$(find ~/.triton/cache -name "sage_fwd.json" | head -1)

    if [ -f "$json_file" ]; then
        lds_bytes=$(cat "$json_file" | python3 -c "import sys, json; print(json.load(sys.stdin)['shared'])")
        lds_kb=$((lds_bytes / 1024))

        echo "  LDS usage: $lds_bytes bytes ($lds_kb KB)"

        # Check if within limits
        if [ $lds_bytes -gt 163840 ]; then
            echo "  ❌ EXCEEDS 160 KB limit!"
        else
            echo "  ✓ Within 160 KB limit"
        fi
    else
        echo "  ERROR: sage_fwd.json not found - compilation likely failed"
    fi
done

echo ""
echo "====================================================================="
