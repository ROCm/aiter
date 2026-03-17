# Grouped-Query Attention (GQA) Benchmark Guide

## Overview

`bench_gqa.py` is a performance benchmarking tool for **Grouped-Query Attention (GQA)** operations. GQA is a variant of attention where multiple query heads share fewer key/value heads (HQ > HK > 1), which provides a balance between memory efficiency and model quality compared to standard Multi-Head Attention (MHA).

### What is Grouped-Query Attention (GQA)?

- **GQA**: Multiple query heads (HQ) share fewer key/value heads (HK), where HK > 1 and HK < HQ
- **MHA**: Each query head has its own key/value head (HQ = HK)
- **MQA**: All query heads share a single key/value head (HK = 1)

Common GQA ratios include:
- **2:1** (HQ=32, HK=16)
- **4:1** (HQ=32, HK=8)
- **8:1** (HQ=64, HK=8)
- **16:1** (HQ=128, HK=8)

GQA is particularly useful for:
- Balancing memory efficiency and model quality
- Improving throughput for large language models
- Optimizing KV cache usage while maintaining attention diversity
- Used in models like LLaMA-2, LLaMA-3, and other modern architectures

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Command-Line Arguments](#command-line-arguments)
- [Usage Examples](#usage-examples)
- [Benchmark Configurations](#benchmark-configurations)
- [Output Format](#output-format)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **CUDA-capable GPU** with compute capability 7.0 or higher
2. **Python 3.8+** with PyTorch and Triton installed
3. **AITER environment** properly configured

## Quick Start

### Basic Usage

```bash
# Run with default GQA configurations (common ratios automatically set)
python bench_gqa.py

# Run with custom parameters
python bench_gqa.py -b 4 -hq 32 -hk 8 -sq 2048 -d 128 --dtype fp16 -metric throughput

# Run with model configurations
python bench_gqa.py --model llama3-70B -b 1 -sq 4096
```

## Command-Line Arguments

### Core Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-b` | int | 0 | Batch size |
| `-hq` | int | 0 | Number of query heads |
| `-hk` | int | 0 | Number of key/value heads (defaults to HQ/4 for GQA, must be > 1 and < HQ) |
| `-sq` | int | 0 | Query sequence length |
| `-sk` | int | 0 | Key sequence length (defaults to `-sq` if not specified) |
| `-d` | int | 0 | Head dimension for Q and K |
| `-dv` | int | 0 | Head dimension for V (defaults to `-d` if not specified) |

### Data Type and Precision

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dtype` | str | `fp16` | Data type: `fp16`, `bf16`, or `fp32` |
| `-fp8` | flag | False | Enable FP8 precision |

### Kernel Mode

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-mode` | str | `fwd` | Kernel mode: `fwd` (forward) or `bwd` (backward) |

### Attention Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-causal` | bool | None | Enable causal attention mask (auto-set with `--model`) |
| `-sink` | flag | False | Use attention sink mechanism |

### Layout and Sequence Handling

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--layout` | str | `bshd` | Layout: `bshd` (batch-first) or `thd` (variable-length) |
| `-equal_seqlens` | flag | False | Use equal sequence lengths with `thd` layout |

### Performance Metrics

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-metric` | str | `throughput` | Metric: `time`, `throughput`, or `bandwidth` |

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | None | Model name from `utils/model_configs.json` |
| `--model-configs` | str | `utils/model_configs.json` | Path to model config file |

### Advanced Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-fused_bwd` | flag | False | Use fused backward kernel |
| `-test_mode` | flag | False | Test correctness against PyTorch SDPA |
| `-bench_torch` | flag | False | Benchmark PyTorch implementation for comparison |
| `-print_vgpr` | flag | False | Print VGPR (vector register) usage |
| `-o` | flag | False | Save results to CSV file |

## Usage Examples

### Example 1: Basic Forward Pass Benchmark

```bash
python bench_gqa.py \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128 \
    --dtype fp16 \
    -metric throughput \
    -mode fwd
```

This benchmarks GQA with:
- Batch size: 4
- Query heads: 32
- Key/value heads: 8 (4:1 ratio)
- Sequence length: 2048
- Head dimension: 128
- Data type: FP16
- Metric: Throughput (TFLOPS)

### Example 2: Backward Pass Benchmark

```bash
python bench_gqa.py \
    -mode bwd \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128 \
    --dtype fp16 \
    -fused_bwd
```

### Example 3: Using Model Configurations

```bash
# Benchmark a specific model
python bench_gqa.py --model llama3-70B -b 1 -sq 4096 -metric throughput

# Benchmark all models in a family
python bench_gqa.py --model llama3 -b 1 -sq 4096

# Benchmark all available models
python bench_gqa.py --model all -b 1 -sq 4096
```

**Available Model Names:**
- `llama3-8B`, `llama3-70B`, `llama3-405B`
- `mixtral-7B`, `mixtral-22B`
- `deepseek-V3`

You can also use just the family name (e.g., `llama3`) to benchmark all models in that family.

**Note:** Model names are case-sensitive. Use capital letters for model sizes (e.g., `70B` not `70b`).

### Example 4: Variable-Length Sequences

```bash
python bench_gqa.py \
    --layout thd \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128 \
    --dtype fp16
```

### Example 5: FP8 Precision

```bash
python bench_gqa.py \
    -fp8 \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128
```

### Example 6: Causal Attention

```bash
python bench_gqa.py \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128 \
    -causal true \
    --dtype fp16
```

### Example 7: Correctness Testing

```bash
python bench_gqa.py \
    -test_mode \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 1024 \
    -d 128 \
    --dtype fp16
```

This verifies that the Triton implementation matches PyTorch's SDPA implementation.

### Example 8: Save Results to CSV

```bash
python bench_gqa.py \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -d 128 \
    -o
```

Results will be saved to a CSV file in the current directory.

## Benchmark Configurations

### Default Configurations

When run without custom parameters, `bench_gqa.py` uses predefined configurations optimized for GQA:

**Non-varlen (bshd layout):**
- Batch sizes: [1, 4, 16]
- Query heads: [32, 64, 128]
- Key/value heads: Varied with ratios 2:1, 4:1, 8:1 (e.g., HQ=32 → HK=[16, 8, 4])
- Query sequence lengths: [1, 1024, 4096]
- Key sequence lengths: [163, 8192]

**Varlen (thd layout):**
- Batch sizes: [1, 4, 8]
- Query heads: [32, 64, 128]
- Key/value heads: Varied with ratios 2:1, 4:1, 8:1
- Query sequence lengths: [1, 1024, 4096]
- Key sequence lengths: [163, 8192]

### Custom Configurations

When using custom parameters, you must provide:
- Batch size (`-b`)
- Query heads (`-hq`)
- Query sequence length (`-sq`)
- Head dimension (`-d`)

Key/value heads (`-hk`) defaults to `HQ/4` for GQA if not specified. The benchmark will automatically adjust HK to ensure:
- HK > 1 (for GQA, use MQA benchmark if HK=1)
- HK < HQ
- HQ is divisible by HK

## Output Format

### Console Output

The benchmark prints a table showing performance metrics:

```
BATCH  HQ  HK  N_CTX_Q  N_CTX_K  fwd(TFLOPS)
-----  --  --  -------  -------  -----------
1      32  8   1024     1024     245.67
4      32  8   1024     1024     312.45
16     32  8   1024     1024     298.23
```

### CSV Output

When using `-o`, results are saved to a CSV file with columns:
- Configuration parameters (BATCH, HQ, HK, N_CTX_Q, N_CTX_K, etc.)
- Performance metrics (time, throughput, bandwidth)

## Performance Metrics

### Time (ms)

Measures the execution time of the kernel in milliseconds. Lower is better.

```bash
python bench_gqa.py -metric time -b 4 -hq 32 -hk 8 -sq 2048 -d 128
```

### Throughput (TFLOPS)

Measures the computational throughput in TeraFLOPS. Higher is better.

```bash
python bench_gqa.py -metric throughput -b 4 -hq 32 -hk 8 -sq 2048 -d 128
```

**FLOPS Calculation:**
- Forward: `2 × BATCH × HQ × N_CTX_Q × N_CTX_K × (D_HEAD + D_HEAD_V)`
- Backward: `2.5 × Forward FLOPS` (includes recomputation)

### Bandwidth (GB/s)

Measures memory bandwidth in gigabytes per second. Higher is better.

```bash
python bench_gqa.py -metric bandwidth -b 4 -hq 32 -hk 8 -sq 2048 -d 128
```

**Memory Calculation:**
- Forward: Read (Q, K, V) + Write (O)
- Backward: Read (Q, K, V, dO) + Write (dQ, dK, dV)

## GQA-Specific Considerations

### Head Configuration

- **HK must be > 1 and < HQ** for true GQA
- **HQ must be divisible by HK** (enforced by the benchmark)
- Common ratios: 2:1, 4:1, 8:1, 16:1 (HQ:HK)
- If `-hk` is specified with HK=1, a warning is issued (consider MQA benchmark)
- If `-hk` is specified with HK >= HQ, an error is raised (use MHA benchmark)

### Memory Efficiency

GQA provides a balance between MHA and MQA:
- **MHA**: Memory scales with `HQ × HK` (where HQ = HK)
- **GQA**: Memory scales with `HQ × HK` (where HK < HQ)
- **MQA**: Memory scales with `HQ × 1` (only one KV head)

For example, with HQ=32:
- MHA (HK=32): 32 × 32 = 1024 KV heads
- GQA (HK=8): 32 × 8 = 256 KV heads (4× reduction)
- MQA (HK=1): 32 × 1 = 32 KV heads (32× reduction)

### Performance Characteristics

GQA typically shows:
- **Better memory efficiency** than MHA while maintaining more attention diversity than MQA
- **Improved throughput** for inference workloads compared to MHA
- **Reduced KV cache size** compared to MHA (critical for long sequences)
- **Better quality** than MQA due to multiple KV heads providing more representational capacity

## Troubleshooting

### Common Issues

#### 1. "HQ must be divisible by HK"

**Problem:** The number of query heads is not divisible by key/value heads.

**Solution:** Ensure `HQ % HK == 0`. Choose HK as a divisor of HQ.

```bash
# Correct: HQ=32, HK=8 (32 % 8 == 0)
python bench_gqa.py -hq 32 -hk 8

# Incorrect: HQ=32, HK=3 (32 % 3 != 0)
python bench_gqa.py -hq 32 -hk 3
```

#### 2. "HK must be less than HQ"

**Problem:** You specified `-hk` with a value >= HQ.

**Solution:** For GQA, ensure HK < HQ. If HK = HQ, use MHA benchmark. If HK = 1, use MQA benchmark.

```bash
# Correct: HQ=32, HK=8 (8 < 32)
python bench_gqa.py -hq 32 -hk 8

# Incorrect: HQ=32, HK=32 (use MHA benchmark instead)
python bench_gqa.py -hq 32 -hk 32

# Incorrect: HQ=32, HK=1 (use MQA benchmark instead)
python bench_gqa.py -hq 32 -hk 1
```

#### 3. "GQA benchmark expects HK > 1"

**Problem:** You specified `-hk` with a value of 1.

**Solution:** For HK=1, use the MQA benchmark (`bench_mqa.py`) instead.

```bash
# For HK=1, use MQA benchmark
python bench_mqa.py -hq 32

# For HK > 1 and < HQ, use GQA benchmark
python bench_gqa.py -hq 32 -hk 8
```

#### 4. Out of Memory (OOM)

**Problem:** GPU runs out of memory with large configurations.

**Solution:** 
- Reduce batch size (`-b`)
- Reduce sequence length (`-sq`, `-sk`)
- Use FP16/BF16 instead of FP32 (`--dtype fp16`)
- Use variable-length layout (`--layout thd`)
- Increase GQA ratio (reduce HK relative to HQ)

#### 5. Model Not Found

**Problem:** `--model` argument doesn't match any model in config file.

**Solution:**
- Check available models: `python bench_gqa.py --help`
- Verify model name spelling (case-sensitive)
- Check `utils/model_configs.json` for available models

#### 6. Test Mode Failures

**Problem:** `-test_mode` shows mismatches with PyTorch SDPA.

**Solution:**
- Verify CUDA and PyTorch versions are compatible
- Check that Triton kernels are properly compiled
- Ensure data types match between implementations

## Best Practices

1. **Start with default configs**: Run without parameters first to see baseline performance
2. **Use model configs**: For real-world scenarios, use `--model` flag
3. **Benchmark both directions**: Test both forward (`-mode fwd`) and backward (`-mode bwd`) passes
4. **Compare metrics**: Use different metrics (`time`, `throughput`, `bandwidth`) to understand performance characteristics
5. **Test correctness**: Use `-test_mode` before relying on performance numbers
6. **Save results**: Use `-o` to save results for later analysis
7. **Choose appropriate ratios**: Common GQA ratios are 2:1, 4:1, 8:1 - choose based on your memory and quality requirements

## Related Benchmarks

- **`bench_mha.py`**: Multi-Head Attention (MHA) benchmark
- **`bench_mqa.py`**: Multi-Query Attention (MQA) benchmark
- **`bench_fp8_mqa_logits.py`**: FP8 MQA logits computation benchmark
- **`bench_batch_prefill.py`**: Batch prefill attention benchmark

## Additional Resources

- [AITER Documentation](../../../../README.md)
- [Operation Benchmarks README](../README.md)
- [Model Configuration Guide](utils/model_configs.json)

## Support

For issues or questions:
1. Check this guide and the troubleshooting section
2. Review the benchmark script help: `python bench_gqa.py --help`
3. Check existing GitHub issues
4. Create a new issue with benchmark configuration and error details

## Performance Benchmarks

### How to Run GQA

Run Grouped-Query Attention performance benchmarks:

```bash
# Navigate to benchmarks directory
cd op_tests/op_benchmarks/triton

# List available models (check --help for available model names)
python bench_gqa.py --help

# Run GQA benchmark with model configurations
# Note: Model names use format "family-size" with capital letters (e.g., "70B" not "70b")
# Note: HK is automatically set based on model config or defaults to HQ/4
python bench_gqa.py --model llama3-70B

# Run all models in a family
python bench_gqa.py --model llama3

# Run with custom parameters (without model config)
# Note: HK defaults to HQ/4 for GQA if not specified, but should be > 1 and < HQ
python bench_gqa.py \
    -b 4 \
    -hq 32 \
    -hk 8 \
    -sq 2048 \
    -sk 2048 \
    -d 128 \
    --dtype fp16 \
    -metric throughput \
    -mode fwd

# Run with FP8 precision
python bench_gqa.py -fp8 -b 4 -hq 32 -hk 8 -sq 2048 -sk 2048 -d 128

# Run backward pass benchmark
python bench_gqa.py -mode bwd -b 4 -hq 32 -hk 8 -sq 2048 -sk 2048 -d 128

# Run with model-specific configurations
python bench_gqa.py --model llama3-70B -b 1 -sq 4096 -metric throughput

# Test correctness against PyTorch SDPA
python bench_gqa.py -test_mode -b 4 -hq 32 -hk 8 -sq 1024 -sk 1024 -d 128
```

**Available Model Names:**

The model configuration file (`utils/model_configs.json`) contains the following models:
- `llama3-8B`, `llama3-70B`, `llama3-405B`
- `mixtral-7B`, `mixtral-22B`
- `deepseek-V3`

You can also use just the family name (e.g., `llama3`) to benchmark all models in that family, or use `all` to benchmark all available models.

**Important:** Model names are case-sensitive. Use capital letters for model sizes (e.g., `70B` not `70b`).

**Common GQA Benchmark Arguments:**

- `-b`: Batch size
- `-hq`: Number of query heads
- `-hk`: Number of key/value heads (defaults to HQ/4 for GQA, must be > 1 and < HQ)
- `-sq`: Query sequence length
- `-sk`: Key sequence length
- `-d`: Head dimension (Q and K)
- `-dv`: Value head dimension (optional)
- `--dtype`: Data type (`fp16`, `bf16`, `fp32`)
- `-mode`: Kernel mode (`fwd` for forward, `bwd` for backward)
- `-metric`: Performance metric (`throughput`, `time`, `bandwidth`)
- `-causal`: Enable causal attention mask
- `-fp8`: Use FP8 precision
- `--model`: Model name (uses model configs from `utils/model_configs.json`)
- `-test_mode`: Run correctness tests comparing to PyTorch SDPA
- `-o`: Save results to CSV file

**Comprehensive Benchmark Script:**

Here's a bash script to run `bench_gqa.py` with all combinations of flags:

```bash
#!/bin/bash
# Comprehensive benchmark script for bench_gqa.py
# Tests all combinations of parameters without using --model flag
# Note: For GQA, HK must be > 1 and < HQ, and HQ must be divisible by HK
# Total combinations: 4 (batch) × 4 (seqlen) × 3 (hq) × 2 (hk ratios) × 3 (head_dim) × 2 (mode) × 3 (dtype) × 2 (causal) × 2 (fp8) × 3 (metric) = 10,368 combinations
# Note: Some combinations are skipped (fp8 flag without fp8 dtype), so actual number may be slightly less
# Note: sq and sk use the same value (seqlen) for both query and key sequence lengths

OUTPUT_DIR="gqa_benchmark_results"
mkdir -p $OUTPUT_DIR

echo "Starting comprehensive GQA benchmarks with all combinations (using custom parameters, no --model flag)..."
echo "Note: HK must be > 1 and < HQ, and HQ must be divisible by HK"

# Define parameter arrays
BATCH_SIZES=(1 4 8 16)
SEQLENS=(512 1024 2048 4096)
HQ_VALUES=(32 64 128)
# Common GQA ratios: 2:1, 4:1, 8:1
# For each HQ, we'll compute valid HK values
HEAD_DIMS=(64 128 256)
MODES=(fwd bwd)
DTYPES=(fp16 bf16 fp8)
CAUSAL_FLAGS=("" "-causal")
FP8_FLAGS=("" "-fp8")
METRICS=(time throughput bandwidth)

# Counter for tracking progress
TOTAL=0
CURRENT=0

# Pre-calculate total (approximate, as we skip invalid combinations)
for BATCH in "${BATCH_SIZES[@]}"; do
    for SEQLEN in "${SEQLENS[@]}"; do
        for HQ in "${HQ_VALUES[@]}"; do
            # Calculate valid HK values for this HQ (ratios 2:1, 4:1, 8:1)
            for ratio in 2 4 8; do
                if [ $((HQ % ratio)) -eq 0 ]; then
                    HK=$((HQ / ratio))
                    if [ $HK -gt 1 ] && [ $HK -lt $HQ ]; then
                        for HEAD_DIM in "${HEAD_DIMS[@]}"; do
                            for MODE in "${MODES[@]}"; do
                                for DTYPE in "${DTYPES[@]}"; do
                                    for CAUSAL in "${CAUSAL_FLAGS[@]}"; do
                                        for FP8 in "${FP8_FLAGS[@]}"; do
                                            for METRIC in "${METRICS[@]}"; do
                                                # Skip invalid combinations (fp8 requires fp8 dtype)
                                                if [ -n "$FP8" ] && [ "$DTYPE" != "fp8" ]; then
                                                    continue
                                                fi
                                                TOTAL=$((TOTAL + 1))
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    fi
                fi
            done
        done
    done
done

echo "Total combinations to test: $TOTAL"

# Nested loops to test all combinations
for BATCH in "${BATCH_SIZES[@]}"; do
    for SEQLEN in "${SEQLENS[@]}"; do
        for HQ in "${HQ_VALUES[@]}"; do
            # Calculate valid HK values for this HQ (ratios 2:1, 4:1, 8:1)
            for ratio in 2 4 8; do
                if [ $((HQ % ratio)) -eq 0 ]; then
                    HK=$((HQ / ratio))
                    if [ $HK -gt 1 ] && [ $HK -lt $HQ ]; then
                        for HEAD_DIM in "${HEAD_DIMS[@]}"; do
                            for MODE in "${MODES[@]}"; do
                                for DTYPE in "${DTYPES[@]}"; do
                                    for CAUSAL in "${CAUSAL_FLAGS[@]}"; do
                                        for FP8 in "${FP8_FLAGS[@]}"; do
                                            for METRIC in "${METRICS[@]}"; do
                                                CURRENT=$((CURRENT + 1))
                                                
                                                # Skip invalid combinations (fp8 requires fp8 dtype)
                                                if [ -n "$FP8" ] && [ "$DTYPE" != "fp8" ]; then
                                                    continue
                                                fi
                                                
                                                # Build filename from parameters
                                                FILENAME="b${BATCH}_s${SEQLEN}_hq${HQ}_hk${HK}_d${HEAD_DIM}_${MODE}_${DTYPE}"
                                                if [ -n "$CAUSAL" ]; then
                                                    FILENAME="${FILENAME}_causal"
                                                fi
                                                if [ -n "$FP8" ]; then
                                                    FILENAME="${FILENAME}_fp8"
                                                fi
                                                FILENAME="${FILENAME}_${METRIC}"
                                                
                                                echo "[$CURRENT/$TOTAL] Testing: batch=$BATCH, seqlen=$SEQLEN, hq=$HQ, hk=$HK (ratio=$ratio:1), d=$HEAD_DIM, mode=$MODE, dtype=$DTYPE, causal=${CAUSAL:--}, fp8=${FP8:--}, metric=$METRIC"
                                                
                                                # Build command without --model flag (sq and sk use same value)
                                                CMD="python bench_gqa.py -b $BATCH -hq $HQ -hk $HK -sq $SEQLEN -sk $SEQLEN -d $HEAD_DIM -mode $MODE --dtype $DTYPE $CAUSAL $FP8 -metric $METRIC -o"
                                                
                                                # Run benchmark
                                                $CMD > /dev/null 2>&1
                                                
                                                # Move CSV file if it exists
                                                if ls *.csv 1> /dev/null 2>&1; then
                                                    mv *.csv $OUTPUT_DIR/${FILENAME}.csv 2>/dev/null || true
                                                fi
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    fi
                fi
            done
        done
    done
done

# Test VGPR usage separately (only need to run once)
echo "Testing VGPR usage..."
python bench_gqa.py -b 4 -hq 32 -hk 8 -sq 2048 -sk 2048 -d 128 -print_vgpr > $OUTPUT_DIR/vgpr_usage.txt 2>&1

echo "Benchmarking complete! Results saved in $OUTPUT_DIR/"
echo "Total combinations tested: $CURRENT"
```

Save this script as `bench_gqa_comprehensive.sh`, make it executable, and run it:

```bash
chmod +x bench_gqa_comprehensive.sh
./bench_gqa_comprehensive.sh
```

**Note**: The script includes logic to skip invalid combinations (e.g., `-fp8` flag requires `--dtype fp8`). You may want to adjust the parameter arrays based on your specific testing needs, as the combinations can take a very long time to complete.
