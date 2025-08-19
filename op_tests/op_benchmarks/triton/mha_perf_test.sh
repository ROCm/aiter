#!/bin/bash

# Define configurations for d_head=56
CONFIGS_DHEAD_56=(
    "1,128,128,2048,2048,56,bshd,,seq_2048"
    "1,128,128,4096,4096,56,bshd,,seq_4096"
    "1,128,128,8192,8192,56,bshd,,seq_8192"
    #"1,128,128,16384,16384,56,bshd,,seq_16384"
    #"1,128,128,32768,32768,56,bshd,,seq_32768"
    #"1,128,128,65536,65536,56,bshd,,seq_65536"
    #"1,128,128,131072,131072,56,bshd,,seq_131072"
)

# Define configurations for d_head=128
CONFIGS_DHEAD_128=(
    "1,8,8,2048,2048,128,bshd,,seq_2048_h8"
    "1,8,8,4096,4096,128,bshd,,seq_4096_h8"
    "1,8,8,8192,8192,128,bshd,,seq_8192_h8"
    #"1,8,8,16384,16384,128,bshd,,seq_16384_h8"
    #"1,8,8,32768,32768,128,bshd,,seq_32768_h8"
    #"1,8,8,65536,65536,128,bshd,,seq_65536_h8"
    #"1,8,8,131072,131072,128,bshd,,seq_131072_h8"

    "1,16,16,2048,2048,128,bshd,,seq_2048_h16"
    "1,16,16,4096,4096,128,bshd,,seq_4096_h16"
    "1,16,16,8192,8192,128,bshd,,seq_8192_h16"
    #"1,16,16,16384,16384,128,bshd,,seq_16384_h16"
    #"1,16,16,32768,32768,128,bshd,,seq_32768_h16"
    #"1,16,16,65536,65536,128,bshd,,seq_65536_h16"
    #"1,16,16,131072,131072,128,bshd,,seq_131072_h16"

    "1,32,32,2048,2048,128,bshd,,seq_2048_h32"
    "1,32,32,4096,4096,128,bshd,,seq_4096_h32"
    "1,32,32,8192,8192,128,bshd,,seq_8192_h32"
    #"1,32,32,16384,16384,128,bshd,,seq_16384_h32"
    #"1,32,32,32768,32768,128,bshd,,seq_32768_h32"
    #"1,32,32,65536,65536,128,bshd,,seq_65536_h32"
    #"1,32,32,131072,131072,128,bshd,,seq_131072_h32"

    "1,64,64,2048,2048,128,bshd,,seq_2048_h64"
    "1,64,64,4096,4096,128,bshd,,seq_4096_h64"
    "1,64,64,8192,8192,128,bshd,,seq_8192_h64"
    #"1,64,64,16384,16384,128,bshd,,seq_16384_h64"
    #"1,64,64,32768,32768,128,bshd,,seq_32768_h64"
    #"1,64,64,65536,65536,128,bshd,,seq_65536_h64"
    #"1,64,64,131072,131072,128,bshd,,seq_131072_h64"

    "1,128,128,2048,2048,128,bshd,,seq_2048_h128"
    "1,128,128,4096,4096,128,bshd,,seq_4096_h128"
    "1,128,128,8192,8192,128,bshd,,seq_8192_h128"
    #"1,128,128,16384,16384,128,bshd,,seq_16384_h128"
    #"1,128,128,32768,32768,128,bshd,,seq_32768_h128"
    #"1,128,128,65536,65536,128,bshd,,seq_65536_h128"
    #"1,128,128,131072,131072,128,bshd,,seq_131072_h128"
)

# Define d_head values
DHEAD_VALUES=(56 128)

# Function to get configs for a given d_head
get_configs_for_dhead() {
    local d_head=$1
    if [[ "$d_head" == "56" ]]; then
        printf '%s\n' "${CONFIGS_DHEAD_56[@]}"
    elif [[ "$d_head" == "128" ]]; then
        printf '%s\n' "${CONFIGS_DHEAD_128[@]}"
    fi
}
MAPPING_MODES=(
#    "0:true"   # aiter_fa with remap
#    "0:false"  # aiter_fa without remap
    "1:false"  # head_first without remap
#    "2:false"  # triton_fa without remap
)

# Define batch sizes to iterate over
BATCH_SIZES=(1 2)

# Set parallel jobs (adjust as needed)
PARALLEL_JOBS=${PARALLEL_JOBS:-4}

# Set output directory
OUTPUT_DIR=${OUTPUT_DIR:-"/workspace/hf_tmp_benchmark_results"}
mkdir -p "$OUTPUT_DIR"
INPUT_DIR="op_tests/op_benchmarks/triton"

run_benchmark() {
    local config=$1
    local output_dir=$2
    local batch_size=$3
    local mapping_mode=$4
    local use_remap=$5
    local d_head=$6

    IFS=',' read -r batch hq hk sq sk d_head_config layout extra_flags output_name <<< "$config"

    # Create output filename based on all parameters
    local txt_file="${output_dir}/mode${mapping_mode}_remap${use_remap}_dhead${d_head}_batch${batch_size}.txt"

    echo "Running: $output_name (batch=$batch_size, mode=$mapping_mode, remap=$use_remap, d_head=$d_head)"

    # Build the command
    local cmd="python ${INPUT_DIR}/bench_mha.py"
    cmd="$cmd -b $batch_size -hq $hq -hk $hk -sq $sq -sk $sk -d $d_head --layout $layout"
    cmd="$cmd -mapping_mode $mapping_mode"

    # Add remap flag (default is true, so only add -no_remap when false)
    if [[ "$use_remap" == "false" ]]; then
        cmd="$cmd -no_remap"
    fi

    # Add extra flags if any
    if [[ -n "$extra_flags" ]]; then
        cmd="$cmd $extra_flags"
    fi

    cmd="$cmd >> $txt_file"

    echo "Command: $cmd"

    # Execute and capture output
    if output=$(eval "$cmd" 2>&1); then
        echo "✓ Completed: $output_name (batch=$batch_size, mode=$mapping_mode, remap=$use_remap)"
        echo "Output:"
        echo "$output"
    else
        echo "✗ Failed: $output_name (batch=$batch_size, mode=$mapping_mode, remap=$use_remap)"
        echo "Error output:"
        echo "$output"
    fi
    echo "----------------------------------------"
}

export -f run_benchmark

echo "Starting comprehensive benchmark..."
echo "Output directory: $OUTPUT_DIR"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Mapping modes: 0=aiter_fa, 1=head_first, 2=triton_fa"
echo "D_head values: ${DHEAD_VALUES[*]}"

# Main execution loop
for d_head in "${DHEAD_VALUES[@]}"; do
    echo "========================================="
    echo "Starting d_head: $d_head"
    echo "========================================="

    # Get configs for this d_head value
    mapfile -t current_configs < <(get_configs_for_dhead "$d_head")

    for batch_size in "${BATCH_SIZES[@]}"; do
        echo "--- Batch size: $batch_size ---"

        for mapping_mode_config in "${MAPPING_MODES[@]}"; do
            IFS=':' read -r mapping_mode use_remap <<< "$mapping_mode_config"
            echo "--- Mapping mode: $mapping_mode, use_remap: $use_remap ---"

            # Create/clear the txt file for this combination
            txt_file="${OUTPUT_DIR}/mode${mapping_mode}_remap${use_remap}_dhead${d_head}_batch${batch_size}.txt"
            > "$txt_file"  # Clear/create the file

            # Run all configurations for this combination
            if command -v parallel >/dev/null 2>&1; then
                # Use GNU parallel if available
                printf '%s\n' "${current_configs[@]}" | parallel -j "$PARALLEL_JOBS" run_benchmark {} "$OUTPUT_DIR" "$batch_size" "$mapping_mode" "$use_remap" "$d_head"
            else
                # Fallback to sequential execution
                for config in "${current_configs[@]}"; do
                    run_benchmark "$config" "$OUTPUT_DIR" "$batch_size" "$mapping_mode" "$use_remap" "$d_head"
                done
            fi

            echo "Completed: $mapping_mode, remap=$use_remap, d_head=$d_head, batch=$batch_size"
            echo "Output saved to: $txt_file"
            echo ""
        done
    done
done

echo "All benchmarks completed!"
echo "Output files created in: $OUTPUT_DIR"
echo "File naming pattern: mode{0/1/2}_remap{true/false}_dhead{56/128}_batch{1/2/4/8}.txt"
