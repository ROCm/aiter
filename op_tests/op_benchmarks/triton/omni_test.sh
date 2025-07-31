#!/bin/bash

# Simple benchmark runner that generates JSON files
# Usage: ./run_benchmarks.sh [output_dir] [parallel_jobs]

OUTPUT_DIR=${1:-"outputs"}
PARALLEL_JOBS=${2:-1}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define your configs here
#declare -a CONFIGS=(
    # Format: "batch,hq,hk,sq,sk,d_head,layout,extra_flags,output_name"
    #"1,32,32,2048,2048,128,bshd,,config_1"
#    "1,8,8,128,128,128,bshd,,config_test_1_8_128"
#    "1,8,8,8192,8192,128,bshd,,config_1_8_8192"
#    "2,8,8,8192,8192,128,bshd,,config_2_8_8192"
#)

# Define your configs here
declare -a CONFIGS=(
    # Format: "batch,hq,hk,sq,sk,d_head,layout,extra_flags,output_name"

    # LLaMA3 8B - batch 1
    "1,32,8,2048,2048,128,bshd,,llama3_8b_seq_2048_batch_1"
    "1,32,8,4096,4096,128,bshd,,llama3_8b_seq_4096_batch_1"
    "1,32,8,8192,8192,128,bshd,,llama3_8b_seq_8192_batch_1"
    "1,32,8,16384,16384,128,bshd,,llama3_8b_seq_16384_batch_1"

    # LLaMA3 8B - batch 32
    "32,32,8,2048,2048,128,bshd,,llama3_8b_seq_2048_batch_32"
    "32,32,8,4096,4096,128,bshd,,llama3_8b_seq_4096_batch_32"
    "32,32,8,8192,8192,128,bshd,,llama3_8b_seq_8192_batch_32"
    "32,32,8,16384,16384,128,bshd,,llama3_8b_seq_16384_batch_32"

    # LLaMA3 70B - batch 1
    "1,64,8,2048,2048,128,bshd,,llama3_70b_seq_2048_batch_1"
    "1,64,8,4096,4096,128,bshd,,llama3_70b_seq_4096_batch_1"
    "1,64,8,8192,8192,128,bshd,,llama3_70b_seq_8192_batch_1"
    "1,64,8,16384,16384,128,bshd,,llama3_70b_seq_16384_batch_1"

    # LLaMA3 70B - batch 32
    "32,64,8,2048,2048,128,bshd,,llama3_70b_seq_2048_batch_32"
    "32,64,8,4096,4096,128,bshd,,llama3_70b_seq_4096_batch_32"
    "32,64,8,8192,8192,128,bshd,,llama3_70b_seq_8192_batch_32"
    "32,64,8,16384,16384,128,bshd,,llama3_70b_seq_16384_batch_32"

    # LLaMA3 405B - batch 1
    "1,128,8,2048,2048,128,bshd,,llama3_405b_seq_2048_batch_1"
    "1,128,8,4096,4096,128,bshd,,llama3_405b_seq_4096_batch_1"
    "1,128,8,8192,8192,128,bshd,,llama3_405b_seq_8192_batch_1"
    "1,128,8,16384,16384,128,bshd,,llama3_405b_seq_16384_batch_1"

    # LLaMA3 405B - batch 32
    "32,128,8,2048,2048,128,bshd,,llama3_405b_seq_2048_batch_32"
    "32,128,8,4096,4096,128,bshd,,llama3_405b_seq_4096_batch_32"
    "32,128,8,8192,8192,128,bshd,,llama3_405b_seq_8192_batch_32"
    "32,128,8,16384,16384,128,bshd,,llama3_405b_seq_16384_batch_32"

    # Mistral 7B - batch 1
    "1,32,8,2048,2048,128,bshd,,mistral_7b_seq_2048_batch_1"
    "1,32,8,4096,4096,128,bshd,,mistral_7b_seq_4096_batch_1"
    "1,32,8,8192,8192,128,bshd,,mistral_7b_seq_8192_batch_1"
    "1,32,8,16384,16384,128,bshd,,mistral_7b_seq_16384_batch_1"

    # Mistral 7B - batch 32
    "32,32,8,2048,2048,128,bshd,,mistral_7b_seq_2048_batch_32"
    "32,32,8,4096,4096,128,bshd,,mistral_7b_seq_4096_batch_32"
    "32,32,8,8192,8192,128,bshd,,mistral_7b_seq_8192_batch_32"
    "32,32,8,16384,16384,128,bshd,,mistral_7b_seq_16384_batch_32"

    # Mistral 22B - batch 1
    "1,48,8,2048,2048,128,bshd,,mistral_22b_seq_2048_batch_1"
    "1,48,8,4096,4096,128,bshd,,mistral_22b_seq_4096_batch_1"
    "1,48,8,8192,8192,128,bshd,,mistral_22b_seq_8192_batch_1"
    "1,48,8,16384,16384,128,bshd,,mistral_22b_seq_16384_batch_1"

    # Mistral 22B - batch 32
    "32,48,8,2048,2048,128,bshd,,mistral_22b_seq_2048_batch_32"
    "32,48,8,4096,4096,128,bshd,,mistral_22b_seq_4096_batch_32"
    "32,48,8,8192,8192,128,bshd,,mistral_22b_seq_8192_batch_32"
    "32,48,8,16384,16384,128,bshd,,mistral_22b_seq_16384_batch_32"

    # DeepSeek V3 - batch 1
    "1,128,128,2048,2048,56,bshd,,deepseek_v3_seq_2048_batch_1"
    "1,128,128,4096,4096,56,bshd,,deepseek_v3_seq_4096_batch_1"
    "1,128,128,8192,8192,56,bshd,,deepseek_v3_seq_8192_batch_1"
    "1,128,128,16384,16384,56,bshd,,deepseek_v3_seq_16384_batch_1"

    # DeepSeek V3 - batch 32
    "32,128,128,2048,2048,56,bshd,,deepseek_v3_seq_2048_batch_32"
    "32,128,128,4096,4096,56,bshd,,deepseek_v3_seq_4096_batch_32"
    "32,128,128,8192,8192,56,bshd,,deepseek_v3_seq_8192_batch_32"
    "32,128,128,16384,16384,56,bshd,,deepseek_v3_seq_16384_batch_32"
)

# Function to run a single benchmark
run_benchmark() {
    local config=$1
    local output_dir=$2
    
    IFS=',' read -r batch hq hk sq sk d_head layout extra_flags output_name <<< "$config"
    
    local csv_file="${output_dir}/${output_name}.csv"
    
    echo "Running: $output_name"
    
    # Build the command
    local cmd="omniprobe -a BasicBlockAnalysis -v -i -c ~/.triton/cache -t csv -l $csv_file -d 1 -- python single_test_mha.py"
    cmd="$cmd -b $batch -hq $hq -hk $hk -sq $sq -sk $sk -d $d_head -layout $layout"
    
    # Add extra flags if any
    if [[ -n "$extra_flags" ]]; then
        cmd="$cmd $extra_flags"
    fi
    
    echo "Command: $cmd"
    
    # Execute and capture output
    if output=$(eval "$cmd" 2>&1); then
        echo "✓ Completed: $output_name"
        echo "Output:"
        echo "$output"
    else
        echo "✗ Failed: $output_name"
        echo "Error output:"
        echo "$output"
    fi
    echo "----------------------------------------"
}

export -f run_benchmark

echo "Running ${#CONFIGS[@]} configurations with $PARALLEL_JOBS parallel jobs"
echo "Output directory: $OUTPUT_DIR"

# Run benchmarks
if [[ $PARALLEL_JOBS -eq 1 ]]; then
    # Sequential
    for config in "${CONFIGS[@]}"; do
        run_benchmark "$config" "$OUTPUT_DIR"
    done
else
    # Parallel using GNU parallel or xargs
    if command -v parallel > /dev/null; then
        printf '%s\n' "${CONFIGS[@]}" | parallel -j "$PARALLEL_JOBS" run_benchmark {} "$OUTPUT_DIR"
    else
        printf '%s\n' "${CONFIGS[@]}" | xargs -I {} -P "$PARALLEL_JOBS" bash -c "run_benchmark '{}' '$OUTPUT_DIR'"
    fi
fi

echo "All benchmarks completed. JSON files saved in $OUTPUT_DIR/"  
