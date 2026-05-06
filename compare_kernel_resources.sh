#!/bin/bash

echo "Comparing kernel resource usage for different num_stages"
echo "========================================================="

for stages in 3 4; do
    echo ""
    echo "Testing num_stages=$stages..."

    # Update config
    sed -i "s/\"num_stages\": [0-9]/\"num_stages\": $stages/" aiter/ops/triton/attention/fav3_sage.py

    # Clear cache
    rm -rf ~/.triton/cache/*

    # Run benchmark to compile kernel
    echo "  Compiling kernel..."
    HIP_VISIBLE_DEVICES=0 python op_tests/op_benchmarks/triton/bench_fav3_sage.py -b 1 -sq 75600 -hq 8 -d 128 > /dev/null 2>&1

    # Find the sage_fwd.json
    json_file=$(find ~/.triton/cache -name "sage_fwd.json" | head -1)

    if [ -f "$json_file" ]; then
        lds_bytes=$(cat "$json_file" | python3 -c "import sys, json; print(json.load(sys.stdin)['shared'])")
        num_warps=$(cat "$json_file" | python3 -c "import sys, json; print(json.load(sys.stdin)['num_warps'])")

        echo "  LDS usage: $lds_bytes bytes"
        echo "  Num warps: $num_warps"

        # Calculate occupancy
        lds_per_cu=65536
        workgroup_size=$((num_warps * 64))
        max_wg_lds=$((lds_per_cu / lds_bytes))

        echo "  Workgroup size: $workgroup_size threads"
        echo "  Max workgroups/CU (LDS limit): $max_wg_lds"

        if [ $max_wg_lds -eq 1 ]; then
            echo "  ⚠️  BOTTLENECK: Only 1 workgroup fits per CU!"
        fi
    else
        echo "  ERROR: sage_fwd.json not found"
    fi
done

echo ""
echo "========================================================="
