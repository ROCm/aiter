# Multi-Query Attention (MQA) Benchmark Guide

## Overview

`bench_mqa.py` is a performance benchmarking tool for **Multi-Query Attention (MQA)** operations. MQA is a variant of attention where all query heads share a single key/value head (HK=1), which reduces memory usage and computational cost compared to standard Multi-Head Attention (MHA).

### What is Multi-Query Attention (MQA)?

- **MQA**: All query heads (HQ) share a single key/value head (HK=1)
- **MHA**: Each query head has its own key/value head (HQ = HK)
- **GQA**: A generalization where multiple query heads share fewer key/value heads (HQ > HK > 1)

MQA is particularly useful for:
- Reducing memory footprint during inference
- Improving throughput for large language models
- Optimizing KV cache usage

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
# Run with default MQA configurations (HK=1 automatically set)
python bench_mqa.py

# Run with custom parameters
python bench_mqa.py -b 4 -hq 32 -sq 2048 -d 128 --dtype fp16 -metric throughput

# Run with model configurations
python bench_mqa.py --model llama3-70B -b 1 -sq 4096
```

## Command-Line Arguments

### Core Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-b` | int | 0 | Batch size |
| `-hq` | int | 0 | Number of query heads |
| `-hk` | int | 0 | Number of key/value heads (defaults to 1 for MQA) |
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
python bench_mqa.py \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128 \
    --dtype fp16 \
    -metric throughput \
    -mode fwd
```

This benchmarks MQA with:
- Batch size: 4
- Query heads: 32
- Key/value heads: 1 (automatically set for MQA)
- Sequence length: 2048
- Head dimension: 128
- Data type: FP16
- Metric: Throughput (TFLOPS)

### Example 2: Backward Pass Benchmark

```bash
python bench_mqa.py \
    -mode bwd \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128 \
    --dtype fp16 \
    -fused_bwd
```

### Example 3: Using Model Configurations

```bash
# Benchmark a specific model
python bench_mqa.py --model llama3-70B -b 1 -sq 4096 -metric throughput

# Benchmark all models in a family
python bench_mqa.py --model llama3 -b 1 -sq 4096

# Benchmark all available models
python bench_mqa.py --model all -b 1 -sq 4096
```

**Available Model Names:**
- `llama3-8B`, `llama3-70B`, `llama3-405B`
- `mixtral-7B`, `mixtral-22B`
- `deepseek-V3`

You can also use just the family name (e.g., `llama3`) to benchmark all models in that family.

**Note:** Model names are case-sensitive. Use capital letters for model sizes (e.g., `70B` not `70b`).

### Example 4: Variable-Length Sequences

```bash
python bench_mqa.py \
    --layout thd \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128 \
    --dtype fp16
```

### Example 5: FP8 Precision

```bash
python bench_mqa.py \
    -fp8 \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128
```

### Example 6: Causal Attention

```bash
python bench_mqa.py \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128 \
    -causal true \
    --dtype fp16
```

### Example 7: Correctness Testing

```bash
python bench_mqa.py \
    -test_mode \
    -b 4 \
    -hq 32 \
    -sq 1024 \
    -d 128 \
    --dtype fp16
```

This verifies that the Triton implementation matches PyTorch's SDPA implementation.

### Example 8: Save Results to CSV

```bash
python bench_mqa.py \
    -b 4 \
    -hq 32 \
    -sq 2048 \
    -d 128 \
    -o
```

Results will be saved to a CSV file in the current directory.

## Benchmark Configurations

### Default Configurations

When run without custom parameters, `bench_mqa.py` uses predefined configurations optimized for MQA:

**Non-varlen (bshd layout):**
- Batch sizes: [1, 4, 16]
- Query heads: [16, 48, 64]
- Key/value heads: 1 (MQA)
- Query sequence lengths: [1, 1024, 4096]
- Key sequence lengths: [163, 8192]

**Varlen (thd layout):**
- Batch sizes: [1, 4, 8]
- Query heads: [16, 48, 64]
- Key/value heads: 1 (MQA)
- Query sequence lengths: [1, 1024, 4096]
- Key sequence lengths: [163, 8192]

### Custom Configurations

When using custom parameters, you must provide:
- Batch size (`-b`)
- Query heads (`-hq`)
- Query sequence length (`-sq`)
- Head dimension (`-d`)

Key/value heads (`-hk`) defaults to 1 for MQA. If you specify a different value, a warning will be issued.

## Output Format

### Console Output

The benchmark prints a table showing performance metrics:

```
BATCH  HQ  HK  N_CTX_Q  N_CTX_K  fwd(TFLOPS)
-----  --  --  -------  -------  -----------
1      32  1   1024     1024     245.67
4      32  1   1024     1024     312.45
16     32  1   1024     1024     298.23
```

### CSV Output

When using `-o`, results are saved to a CSV file with columns:
- Configuration parameters (BATCH, HQ, HK, N_CTX_Q, N_CTX_K, etc.)
- Performance metrics (time, throughput, bandwidth)

## Performance Metrics

### Time (ms)

Measures the execution time of the kernel in milliseconds. Lower is better.

```bash
python bench_mqa.py -metric time -b 4 -hq 32 -sq 2048 -d 128
```

### Throughput (TFLOPS)

Measures the computational throughput in TeraFLOPS. Higher is better.

```bash
python bench_mqa.py -metric throughput -b 4 -hq 32 -sq 2048 -d 128
```

**FLOPS Calculation:**
- Forward: `2 × BATCH × HQ × N_CTX_Q × N_CTX_K × (D_HEAD + D_HEAD_V)`
- Backward: `2.5 × Forward FLOPS` (includes recomputation)

### Bandwidth (GB/s)

Measures memory bandwidth in gigabytes per second. Higher is better.

```bash
python bench_mqa.py -metric bandwidth -b 4 -hq 32 -sq 2048 -d 128
```

**Memory Calculation:**
- Forward: Read (Q, K, V) + Write (O)
- Backward: Read (Q, K, V, dO) + Write (dQ, dK, dV)

## MQA-Specific Considerations

### Head Configuration

- **HK must be 1** for true MQA (default behavior)
- **HQ must be divisible by HK** (enforced by the benchmark)
- If `-hk` is specified with a value other than 1, a warning is issued

### Memory Efficiency

MQA reduces memory usage compared to MHA:
- **MHA**: Memory scales with `HQ × HK`
- **MQA**: Memory scales with `HQ × 1` (only one KV head)

### Performance Characteristics

MQA typically shows:
- **Lower memory usage** than MHA
- **Similar or better throughput** for inference workloads
- **Reduced KV cache size** (critical for long sequences)

## Troubleshooting

### Common Issues

#### 1. "HQ must be divisible by HK"

**Problem:** The number of query heads is not divisible by key/value heads.

**Solution:** Ensure `HQ % HK == 0`. For MQA, use `HK=1`.

```bash
# Correct: HQ=32, HK=1
python bench_mqa.py -hq 32 -hk 1

# Incorrect: HQ=32, HK=3 (32 % 3 != 0)
python bench_mqa.py -hq 32 -hk 3
```

#### 2. "MQA benchmark expects HK=1"

**Problem:** You specified `-hk` with a value other than 1.

**Solution:** For MQA, don't specify `-hk` (it defaults to 1), or explicitly set `-hk 1`.

```bash
# Correct: Let HK default to 1
python bench_mqa.py -hq 32

# Also correct: Explicitly set HK=1
python bench_mqa.py -hq 32 -hk 1
```

#### 3. Out of Memory (OOM)

**Problem:** GPU runs out of memory with large configurations.

**Solution:** 
- Reduce batch size (`-b`)
- Reduce sequence length (`-sq`, `-sk`)
- Use FP16/BF16 instead of FP32 (`--dtype fp16`)
- Use variable-length layout (`--layout thd`)

#### 4. Model Not Found

**Problem:** `--model` argument doesn't match any model in config file.

**Solution:**
- Check available models: `python bench_mqa.py --help`
- Verify model name spelling (case-sensitive)
- Check `utils/model_configs.json` for available models

#### 5. Test Mode Failures

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

## Related Benchmarks

- **`bench_mha.py`**: Multi-Head Attention (MHA) benchmark
- **`bench_fp8_mqa_logits.py`**: FP8 MQA logits computation benchmark
- **`bench_batch_prefill.py`**: Batch prefill attention benchmark

## Additional Resources

- [AITER Documentation](../../../../README.md)
- [Operation Benchmarks README](../README.md)
- [Model Configuration Guide](utils/model_configs.json)

## Support

For issues or questions:
1. Check this guide and the troubleshooting section
2. Review the benchmark script help: `python bench_mqa.py --help`
3. Check existing GitHub issues
4. Create a new issue with benchmark configuration and error details
