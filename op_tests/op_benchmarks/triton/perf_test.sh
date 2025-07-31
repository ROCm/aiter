#!/bin/bash

# Simple benchmark runner that generates JSON files
# Usage: ./run_benchmarks.sh [output_dir] [parallel_jobs]

OUTPUT_DIR=${1:-"/workspace/aiter_outputs"}
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
#    "1,32,8,2048,2048,128,bshd,,llama3_8b_seq_2048_batch_1"
#    "1,32,8,4096,4096,128,bshd,,llama3_8b_seq_4096_batch_1"
#    "1,32,8,8192,8192,128,bshd,,llama3_8b_seq_8192_batch_1"
#    "1,32,8,16384,16384,128,bshd,,llama3_8b_seq_16384_batch_1"

    # LLaMA3 8B - batch 32
#    "32,32,8,2048,2048,128,bshd,,llama3_8b_seq_2048_batch_32"
#    "32,32,8,4096,4096,128,bshd,,llama3_8b_seq_4096_batch_32"
#    "32,32,8,8192,8192,128,bshd,,llama3_8b_seq_8192_batch_32"
#    "32,32,8,16384,16384,128,bshd,,llama3_8b_seq_16384_batch_32"

    # LLaMA3 70B - batch 1
#    "1,64,8,2048,2048,128,bshd,,llama3_70b_seq_2048_batch_1"
#    "1,64,8,4096,4096,128,bshd,,llama3_70b_seq_4096_batch_1"
#    "1,64,8,8192,8192,128,bshd,,llama3_70b_seq_8192_batch_1"
#    "1,64,8,16384,16384,128,bshd,,llama3_70b_seq_16384_batch_1"

    # LLaMA3 70B - batch 32
#    "32,64,8,2048,2048,128,bshd,,llama3_70b_seq_2048_batch_32"
#    "32,64,8,4096,4096,128,bshd,,llama3_70b_seq_4096_batch_32"
#    "32,64,8,8192,8192,128,bshd,,llama3_70b_seq_8192_batch_32"
#    "32,64,8,16384,16384,128,bshd,,llama3_70b_seq_16384_batch_32"

    # LLaMA3 405B - batch 1
#    "1,128,8,2048,2048,128,bshd,,llama3_405b_seq_2048_batch_1"
#    "1,128,8,4096,4096,128,bshd,,llama3_405b_seq_4096_batch_1"
#    "1,128,8,8192,8192,128,bshd,,llama3_405b_seq_8192_batch_1"
#    "1,128,8,16384,16384,128,bshd,,llama3_405b_seq_16384_batch_1"

    # LLaMA3 405B - batch 32
#    "32,128,8,2048,2048,128,bshd,,llama3_405b_seq_2048_batch_32"
#    "32,128,8,4096,4096,128,bshd,,llama3_405b_seq_4096_batch_32"
#    "32,128,8,8192,8192,128,bshd,,llama3_405b_seq_8192_batch_32"
#    "32,128,8,16384,16384,128,bshd,,llama3_405b_seq_16384_batch_32"

    # Mistral 7B - batch 1
#    "1,32,8,2048,2048,128,bshd,,mistral_7b_seq_2048_batch_1"
#    "1,32,8,4096,4096,128,bshd,,mistral_7b_seq_4096_batch_1"
#    "1,32,8,8192,8192,128,bshd,,mistral_7b_seq_8192_batch_1"
#    "1,32,8,16384,16384,128,bshd,,mistral_7b_seq_16384_batch_1"

   # Mistral 7B - batch 32
#    "32,32,8,2048,2048,128,bshd,,mistral_7b_seq_2048_batch_32"
#    "32,32,8,4096,4096,128,bshd,,mistral_7b_seq_4096_batch_32"
#    "32,32,8,8192,8192,128,bshd,,mistral_7b_seq_8192_batch_32"
#    "32,32,8,16384,16384,128,bshd,,mistral_7b_seq_16384_batch_32"

    # Mistral 22B - batch 1
#    "1,48,8,2048,2048,128,bshd,,mistral_22b_seq_2048_batch_1"
#    "1,48,8,4096,4096,128,bshd,,mistral_22b_seq_4096_batch_1"
#    "1,48,8,8192,8192,128,bshd,,mistral_22b_seq_8192_batch_1"
#    "1,48,8,16384,16384,128,bshd,,mistral_22b_seq_16384_batch_1"

    # Mistral 22B - batch 32
#    "32,48,8,2048,2048,128,bshd,,mistral_22b_seq_2048_batch_32"
#    "32,48,8,4096,4096,128,bshd,,mistral_22b_seq_4096_batch_32"
#    "32,48,8,8192,8192,128,bshd,,mistral_22b_seq_8192_batch_32"
#    "32,48,8,16384,16384,128,bshd,,mistral_22b_seq_16384_batch_32"

    # DeepSeek V3 - batch 1
    "1,128,128,2048,2048,56,bshd,,deepseek_v3_seq_2048_batch_1"
    "1,128,128,4096,4096,56,bshd,,deepseek_v3_seq_4096_batch_1"
    "1,128,128,8192,8192,56,bshd,,deepseek_v3_seq_8192_batch_1"
    "1,128,128,16384,16384,56,bshd,,deepseek_v3_seq_16384_batch_1"
    "1,128,128,32768,32768,56,bshd,,deepseek_v3_seq_32768_batch_1"
    "1,128,128,65536,65536,56,bshd,,deepseek_v3_seq_65536_batch_1"
    "1,128,128,131072,131072,56,bshd,,deepseek_v3_seq_131072_batch_1"

    #"1,8,8,65536,65536,128,bshd,, h8_v3_seq_65536_batch_1"
    #"1,8,8,131072,131072,128,bshd,,h8_seq_131072_batch_1"
    #"1,16,16,65536,65536,128,bshd,, h16_v3_seq_65536_batch_1"
    #"1,16,16,131072,131072,128,bshd,,h16_seq_131072_batch_1"
   # "1,32,32,65536,65536,128,bshd,, h32_v3_seq_65536_batch_1"
   # "1,32,32,131072,131072,128,bshd,,h32_seq_131072_batch_1"
   # "1,64,64,65536,65536,128,bshd,, h64_v3_seq_65536_batch_1"
   # "1,64,64,131072,131072,128,bshd,,h64_seq_131072_batch_1"
   # "1,128,128,65536,65536,128,bshd,, h128_v3_seq_65536_batch_1"
   # "1,128,128,131072,131072,128,bshd,,h128_seq_131072_batch_1"

   # DeepSeek V3 - batch 32
#    "32,128,128,2048,2048,56,bshd,,deepseek_v3_seq_2048_batch_32"
#    "32,128,128,4096,4096,56,bshd,,deepseek_v3_seq_4096_batch_32"
#    "32,128,128,8192,8192,56,bshd,,deepseek_v3_seq_8192_batch_32"
#    "32,128,128,16384,16384,56,bshd,,deepseek_v3_seq_16384_batch_32"
)

# Function to run a single benchmark
run_benchmark() {
    local config=$1
    local output_dir=$2
    
    IFS=',' read -r batch hq hk sq sk d_head layout extra_flags output_name <<< "$config"
    
    local csv_file="${output_dir}/${output_name}.csv"
    local txt_file="${output_dir}/noremap_mha_batch${batch_size}.txt"
    echo "Running: $output_name (batch=$batch_size)"
    
    # Build the command
    local cmd="python bench_mha.py"
    cmd="$cmd -b $batch_size -hq $hq -hk $hk -sq $sq -sk $sk -d $d_head -layout $layout >> $txt_file"
    
    # Add extra flags if any
    if [[ -n "$extra_flags" ]]; then
        cmd="$cmd $extra_flags"
    fi
    
    echo "Command: $cmd"
    
    # Execute and capture output
    if output=$(eval "$cmd" 2>&1); then
        echo "▒~\~S Completed: $output_name (batch=$batch_size)"
        echo "Output:"
        echo "$output"
    else
        echo "▒~\~W Failed: $output_name (batch=$batch_size)"
        echo "Error output:"
        echo "$output"
    fi

    echo "----------------------------------------"
}

export -f run_benchmark

# Define batch sizes to iterate over
BATCH_SIZES=(1 2 4 8)

echo "Running ${#CONFIGS[@]} configurations with $PARALLEL_JOBS parallel jobs"
echo "Output directory: $OUTPUT_DIR"
echo "Batch sizes: ${BATCH_SIZES[*]}"

# Main execution loop
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "========================================="
    echo "Starting batch size: $batch_size"
    echo "========================================="
    
    # Create/clear the txt file for this batch size
    txt_file="${OUTPUT_DIR}/noremap8b_mha_batch${batch_size}.txt"
    > "$txt_file"  # Clear/create the file
    
    # Run all configurations for this batch size
    if command -v parallel >/dev/null 2>&1; then
        # Use GNU parallel if available
        printf '%s\n' "${CONFIGS[@]}" | parallel -j "$PARALLEL_JOBS" run_benchmark {} "$OUTPUT_DIR" "$batch_size"
    else
        # Fallback to sequential execution
        for config in "${CONFIGS[@]}"; do
            run_benchmark "$config" "$OUTPUT_DIR" "$batch_size"
        done
    fi
    
    echo "Completed batch size: $batch_size"
    echo "Output saved to: $txt_file"
    echo ""
done

echo "All batch sizes completed!"
echo "Output files created:"
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "  - ${OUTPUT_DIR}/noremap8b_mha_batch${batch_size}.txt"
done

