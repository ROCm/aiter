# Attention Kernel Benchmarking

Benchmark different attention kernel implementations using either randomly generated tensors or real captured inputs from model inference.

## Benchmarking Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Random Inputs** | Generate random Q, K, V tensors with specified dimensions | Simulated performance testing, |
| **Captured Inputs** | Load real tensors captured during model inference | More realistic workload and production performance validation |

## Available Attention Kernels

| Flag | Kernel | Description |
|------|--------|-------------|
| (none) | FA v2 | Flash Attention v2 (Triton), no quantization |
| `-fp8` | FA v3 FP8 | Flash Attention v3 with FP8 quantization |
| `-qk_int8` | SageAttn v1 | INT8 quantized Q/K with per-block scaling |
| `-fav3_sage` | SageAttn v1 (FA3) | SageAttention v1 fused on FA3 backend |

## Performance Metrics

| Metric | Flag | Description |
|--------|------|-------------|
| Time | `-metric time` | Kernel execution time in milliseconds |
| Throughput | `-metric throughput` | Computational throughput in TFLOPS |
| Bandwidth | `-metric bandwidth` | Memory bandwidth in GB/s |
| Arithmetic Intensity | `-metric arithmetic_intensity` | FLOP/byte ratio |
| All | `-metric all` | Report all metrics |

## Quick Start

### Benchmark with Random Inputs

```bash
# Single configuration - FA v2
python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    -b 2 -hq 30 -sq 17776 -d 64 \
    -metric throughput

# Single configuration - SageAttn v1 (INT8)
python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    -b 2 -hq 30 -sq 17776 -d 64 \
    -qk_int8 \
    -metric all

# Single configuration - FA v3 FP8
python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    -b 2 -hq 30 -sq 17776 -d 64 \
    -fp8 \
    -metric all

# Single configuration - SageAttn v1 on FA3
python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    -b 2 -hq 30 -sq 17776 -d 64 \
    -fav3_sage \
    -metric all
```

### Benchmark with Captured Inputs

```bash
# Step 1: Capture inputs during model inference
python op_tests/sagev1_tests/sageattn_cogvideo.py \
    --attention_type sagev1 \
    --save_inputs \
    --input_dir ./captured_inputs \
    --max_captures 10

# Step 2: Benchmark all kernels on captured inputs
./bench_quantized_attention_captured.sh

# Or run individually:
python op_tests/op_benchmarks/triton/bench_diffusion_attention.py \
    --load_captured \
    --captured_dir ./captured_inputs \
    -fav3_sage \
    -metric all
```

## Batch Benchmarking Scripts

Two shell scripts are provided at the repository root for running comprehensive benchmarks:

### `bench_quantized_attention.sh` - Random Inputs

Runs benchmarks across multiple pre-defined configurations with random tensors:

```bash
./bench_quantized_attention.sh
```

This script tests all kernel types across these configurations:

| Batch | Heads | Seq Length |
|-------|-------|------------|
| 1 | 5 | 75,600 |
| 1 | 24 | 16,452 |
| 1 | 3 | 118,808 |
| 2 | 2 | 29,760 |

### `bench_quantized_attention_captured.sh` - Captured Inputs

Runs benchmarks using captured inputs from model inference:

```bash
# Use default directory (./captured_inputs)
./bench_quantized_attention_captured.sh

# Use custom directory
CAPTURED_DIR=/path/to/inputs ./bench_quantized_attention_captured.sh
```

## CLI Reference for `bench_diffusion_attention.py`

### Required Arguments (Random Mode)

| Argument | Type | Description |
|----------|------|-------------|
| `-b` | int | Batch size |
| `-hq` | int | Number of query heads |
| `-sq` | int | Query sequence length |
| `-d` | int | Head dimension (Q and K) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-hk` | int | same as `-hq` | Number of key/value heads (for GQA/MQA) |
| `-sk` | int | same as `-sq` | Key/value sequence length |
| `-dv` | int | same as `-d` | Value head dimension |
| `--dtype` | str | `fp16` | Data type: `fp16`, `bf16`, `fp32` |
| `-causal` | bool | `False` | Enable causal masking |
| `-metric` | str | `throughput` | Metric to report (see above) |
| `-o` | flag | - | Write results to CSV file |

### Kernel Selection Flags

| Argument | Description |
|----------|-------------|
| `-fp8` | Use FA v3 with FP8 quantization |
| `-qk_int8` | Use SageAttn v1 (INT8 Q/K) |
| `-fav3_sage` | Use SageAttn v1 fused on FA3 |
| `-no_k_smooth` | Disable K smoothing for INT8 kernels |

### Captured Input Mode

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--load_captured` | flag | - | Enable captured input mode |
| `--captured_dir` | str | `./captured_inputs` | Directory with captured `.pt` files |

### Validation & Debugging

| Argument | Type | Description |
|----------|------|-------------|
| `--compare_to_ref` | flag | Compare output against reference kernel |
| `-ref` | str | Reference kernel: `fav2`, `qk_int8`, `fav3_fp8`, or torch |
| `-print_compare_stats` | flag | Print comparison statistics |
| `-print_vgpr` | flag | Print VGPR usage for Triton kernels |
| `--save_output` | flag | Save output tensors for comparison |
| `--output_dir` | str | Directory for saved outputs |
