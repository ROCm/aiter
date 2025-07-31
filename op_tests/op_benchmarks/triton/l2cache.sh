#!/bin/bash

# ROCProfiler MHA Benchmark Script
# Runs rocprofv3 experiments for different MHA configurations

set -e  # Exit on any error

# Base command template
BASE_CMD="rocprofv3 -i /workspace/input_l2.txt --kernel-trace -d"
PYTHON_CMD="python bench_mha.py -b 1 -layout bshd"

# Create base output directory
BASE_OUTPUT_DIR="/workspace/l2cache/newremap"
mkdir -p "$BASE_OUTPUT_DIR"

# Function to run experiment
run_experiment() {
    local d=$1
    local hq=$2
    local hk=$3
    local sq=$4
    local sk=$5
    
    # Create unique folder name
    local folder_name="d${d}_hq${hq}_hk${hk}_sq${sq}_sk${sk}"
    local output_dir="${BASE_OUTPUT_DIR}/${folder_name}"
    
    echo "Running experiment: $folder_name"
    echo "  Parameters: d=$d, hq=$hq, hk=$hk, sq=$sq, sk=$sk"
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Build full command
    local full_cmd="$BASE_CMD $output_dir -- $PYTHON_CMD -d $d -hq $hq -hk $hk -sq $sq -sk $sk"
    
    echo "  Command: $full_cmd"
    
    # Run the experiment
    if eval "$full_cmd"; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed with exit code $?"
        return 1
    fi
    
    echo ""
}

# Configuration 1: d=128, varying hq/hk and sq/sk
echo "=== Configuration 1: d=128 ==="
d=128
hq_hk_values=(8 16 32 64 128)
sq_sk_values=(2048 4096 8192 16384 32768 65536 131072)

for hq in "${hq_hk_values[@]}"; do
    hk=$hq  # hq = hk
    for sq in "${sq_sk_values[@]}"; do
        sk=$sq  # sq = sk
        run_experiment $d $hq $hk $sq $sk
    done
done

# Configuration 2: d=56, hq=hk=128, varying sq/sk
echo "=== Configuration 2: d=56, hq=hk=128 ==="
d=56
hq=128
hk=128
sq_sk_values=(2048 4096 8192 16384 32768 65536 131072)

for sq in "${sq_sk_values[@]}"; do
    sk=$sq  # sq = sk
    run_experiment $d $hq $hk $sq $sk
done

echo "All experiments completed!"
echo "Results are stored in: $BASE_OUTPUT_DIR"
echo "Folder structure:"
find "$BASE_OUTPUT_DIR" -type d -name "d*" | sort
