# Operation Benchmarks

This directory contains performance benchmarks for various operations implemented in AITER. The benchmarks are organized by backend (Triton and HIP) and cover a wide range of operations including GEMM, attention mechanisms, MoE, and other kernel operations.

## Table of Contents

- [User Guide](#user-guide)
  - [Available Benchmarks](#available-benchmarks)
    - [Triton Benchmarks](#triton-benchmarks)
    - [HIP Benchmarks](#hip-benchmarks)
  - [Prerequisites](#prerequisites)
  - [How to Benchmark](#how-to-benchmark)
    - [Running FlashAttention Tests](#running-flashattention-tests)
  - [Running Individual Benchmarks](#running-individual-benchmarks)
  - [Common Command-Line Arguments](#common-command-line-arguments)
  - [Example Usage](#example-usage)
  - [Output Format](#output-format)
  - [Benchmark Schema](#benchmark-schema)
- [Benchmark Schema Reference](#benchmark-schema-reference)
- [Tips for Benchmarking](#tips-for-benchmarking)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## User Guide

### Available Benchmarks

#### Triton Benchmarks

##### GEMM Operations

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_gemm_a8w8.py` | INT8 activation, INT8 weight GEMM | M, N, K | TFLOPS |
| `bench_gemm_a8w8_blockscale.py` | INT8 GEMM with block-wise scaling | M, N, K | TFLOPS |
| `bench_gemm_a8w8_per_token_scale.py` | INT8 GEMM with per-token scaling | M, N, K | TFLOPS |
| `bench_gemm_a8wfp4.py` | INT8 activation, FP4 weight GEMM | M, N, K | TFLOPS |
| `bench_gemm_a16w16.py` | FP16/BF16 activation, FP16/BF16 weight GEMM | M, N, K | TFLOPS |
| `bench_gemm_a16w16_gating.py` | FP16/BF16 GEMM with gating mechanism | M, N, K | TFLOPS |
| `bench_gemm_afp4wfp4.py` | FP4 activation, FP4 weight GEMM | M, N, K | TFLOPS |
| `bench_gemm_afp4wfp4_pre_quant_atomic.py` | FP4 GEMM with pre-quantization (atomic) | M, N, K | TFLOPS |

##### Batched GEMM Operations

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_batched_gemm_a8w8.py` | Batched INT8 GEMM | batch, M, N, K | TFLOPS |
| `bench_batched_gemm_a16w16.py` | Batched FP16/BF16 GEMM | batch, M, N, K | TFLOPS |
| `bench_batched_gemm_a16wfp4.py` | Batched INT8 activation, FP4 weight GEMM | batch, M, N, K | TFLOPS |
| `bench_batched_gemm_afp4wfp4.py` | Batched FP4 GEMM | batch, M, N, K | TFLOPS |

##### Attention Operations

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_mha.py` | Multi-Head Attention | BATCH, HQ, HK, N_CTX_Q, N_CTX_K | fwd(TFLOPS) |
| `bench_mla_decode.py` | Multi-head Latent Attention (decode) | model, BS, HQ, HK, SEQ_LEN, HEAD_DIM | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_mla_decode_rope.py` | MLA decode with RoPE | model, BS, HQ, HK, SEQ_LEN, HEAD_DIM | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_pa_decode.py` | Paged Attention (decode) | model, BS, HQ, HK, SEQ_LEN, HEAD_DIM | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_pa_prefill.py` | Paged Attention (prefill) | model, BS, HQ, HK, MAX_SEQ_LEN, HEAD_DIM | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_la.py` | Linear Attention | OP, BS, HQ, HK, SEQ_LEN, HEAD_DIM | Time(ms) |
| `bench_la_paged_decode.py` | Paged Linear Attention (decode) | OP, BS, HQ, HK, SEQ_LEN, HEAD_DIM | Time(ms) |
| `bench_extend_attention.py` | Extended Attention | B, H, prefix, extend, kv_lora_rank, qk_rope_head_dim, v_head_dim, attn_impl | fwd_Time(ms) |
| `bench_batch_prefill.py` | Batch Prefill Attention | Various | Various |
| `bench_hstu_attn.py` | HSTU (Hierarchical Sparse Transformer Unit) Attention | batch_size, max_seq_len, sparsity, heads, attn_dim, hidden_dim | TFLOPS |
| `bench_deepgemm_attention.py` | DeepGEMM Attention | Various | Various |
| `bench_fav3_sage.py` | FAV3 SAGE Attention | Various | Various |
| `bench_fav3_sage_mxfp4.py` | FAV3 SAGE Attention with MXFP4 | Various | Various |
| `bench_fp8_mqa_logits.py` | FP8 Multi-Query Attention Logits | Various | Various |

##### MoE (Mixture of Experts) Operations

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_moe.py` | MoE forward pass | model, M, N, K, E, top_k | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_moe_mx.py` | MoE with matrix operations | model, M, N, K, E, top_k | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_moe_align_block_size.py` | MoE block size alignment | model, M, N, K, E, top_k | Time(ms), Bandwidth(GB/s) |
| `bench_moe_routing_sigmoid_top1_fused.py` | MoE routing with sigmoid top-1 fusion | M, N, K | Time(ms), TFLOPS, Bandwidth(GB/s) |
| `bench_moe_gemm_a8w8.py` | MoE INT8 GEMM | Various | Various |
| `bench_moe_gemm_a8w8_blockscale.py` | MoE INT8 GEMM with block scaling | Various | Various |
| `bench_moe_gemm_a8w4.py` | MoE INT8 activation, INT4 weight GEMM | Various | Various |
| `bench_moe_gemm_a4w4.py` | MoE INT4 GEMM | Various | Various |
| `bench_moe_gemm_int8_smoothquant.py` | MoE INT8 GEMM with SmoothQuant | Various | Various |

##### Normalization and Other Operations

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_rmsnorm.py` | RMS Normalization | model_name, M, N | Bandwidth(GB/s) |
| `bench_rope.py` | Rotary Position Embedding (RoPE) | model, M (seq_len) | Time(ms), Total FLOPS |
| `bench_topk.py` | Top-K selection | M (batch), N (vocab), topk | Time(ms) |
| `bench_gmm.py` | Gaussian Mixture Model | Various | Various |
| `bench_ff_a16w16_fused.py` | Fused Feed-Forward with FP16/BF16 | Various | Various |

##### Model Benchmarking Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `model_benchmarking_tool/bench_models.py` | Comprehensive model-level benchmarking | `python bench_models.py --model <model_name> --M <sizes> --TP <tensor_parallel>` |
| `model_benchmarking_tool/bench_attn_models.py` | Attention kernel benchmarking across models | `python bench_attn_models.py --kernel <kernels> --model <model_filter>` |

#### HIP Benchmarks

| Benchmark | Description | Input Parameters | Output Metrics |
|-----------|-------------|------------------|----------------|
| `bench_topk_topp_sampling.py` | Top-K/Top-P sampling kernel | batch_size, vocab_size, k, p | Latency(ms), Throughput(tokens/s) |

### Prerequisites

#### Installing AITER

First, install AITER from source:

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
python3 setup.py develop
```

If you forgot the `--recursive` flag during clone, run:

```bash
git submodule sync && git submodule update --init --recursive
```

#### Required Dependencies

- Python 3.x
- PyTorch
- Triton (for Triton benchmarks)
- ROCm/HIP (for HIP benchmarks)
- CUDA-capable GPU (AMD GPU recommended for HIP benchmarks)

### How to Benchmark

#### Running FlashAttention Tests

FlashAttention tests can be run in two ways: correctness tests and performance benchmarks.

##### Correctness Tests

Run the FlashAttention correctness tests to verify the implementation:

```bash
# Run basic FlashAttention output tests
cd op_tests
python test_mha.py --batch_size 4 --nheads 6 --seqlen_q 1024 --seqlen_k 1024 --dtype fp16

# Run with specific parameters
python test_mha.py \
    --batch_size 4 \
    --nheads 32 \
    --seqlen_q 2048 \
    --seqlen_k 2048 \
    --d_qk_v 128 128 \
    --dtype bf16 \
    --causal True \
    --mha_type mha
```

##### Performance Benchmarks

Run FlashAttention performance benchmarks:

```bash
# Navigate to benchmarks directory
cd op_tests/op_benchmarks/triton

# Run FlashAttention benchmark with default model configurations
python bench_mha.py --model llama3-70b

# Run with custom parameters
python bench_mha.py \
    --b 4 \
    --hq 32 \
    --hk 32 \
    --sq 2048 \
    --sk 2048 \
    --d 128 \
    --dtype fp16 \
    --metric throughput \
    --mode fwd

# Run with FP8 precision
python bench_mha.py --fp8 --b 4 --hq 32 --hk 32 --sq 2048 --sk 2048 --d 128

# Run backward pass benchmark
python bench_mha.py --mode bwd --b 4 --hq 32 --hk 32 --sq 2048 --sk 2048 --d 128

# Run with model-specific configurations
python bench_mha.py --model llama3-70b --b 1 --sq 4096 --metric throughput

# Test correctness against PyTorch SDPA
python bench_mha.py --test_mode --b 4 --hq 32 --hk 32 --sq 1024 --sk 1024 --d 128
```

**Common FlashAttention Benchmark Arguments:**

- `--b`: Batch size
- `--hq`: Number of query heads
- `--hk`: Number of key/value heads
- `--sq`: Query sequence length
- `--sk`: Key sequence length
- `--d`: Head dimension (Q and K)
- `--dv`: Value head dimension (optional)
- `--dtype`: Data type (`fp16`, `bf16`, `fp8`)
- `--mode`: Kernel mode (`fwd` for forward, `bwd` for backward)
- `--metric`: Performance metric (`throughput`, `time`, `bandwidth`)
- `--causal`: Enable causal attention mask
- `--fp8`: Use FP8 precision
- `--model`: Model name (uses model configs from `utils/model_configs.json`)
- `--test_mode`: Run correctness tests comparing to PyTorch SDPA
- `-o`: Save results to CSV file

##### KV Cache Tests

Run FlashAttention with KV cache tests:

```bash
# Run KV cache correctness tests
cd op_tests/triton_tests/attention
pytest test_flash_attn_kvcache.py -v
```

### Running Individual Benchmarks

Most benchmark scripts follow a similar pattern. Navigate to the appropriate directory and run the benchmark script:

```bash
# For Triton benchmarks
cd op_tests/op_benchmarks/triton
python bench_gemm_a8w8.py [options]

# For HIP benchmarks
cd op_tests/op_benchmarks/hip
python bench_topk_topp_sampling.py [options]
```

### Common Command-Line Arguments

Most benchmarks support similar arguments:

- `--shape` or `-s`: Specify input dimensions (varies by benchmark)
- `--metric`: Choose metric to report (`time`, `throughput`, `bandwidth`, `TFLOPS`)
- `-o` or `--output`: Save results to CSV file
- `--model`: Specify model configuration (for model-specific benchmarks)
- `--layout`: GEMM layout (`TN`, `TT`, `NN`, `NT`)
- `--dtype`: Data type (`bf16`, `fp16`, `fp32`, etc.)
- `--warmup`: Number of warmup iterations
- `--rep`: Number of benchmark iterations

### Example Usage

#### Running a GEMM Benchmark

```bash
# Run with default parameters
python bench_gemm_a8w8.py

# Run with custom shape and save results
python bench_gemm_a8w8.py --shape 1024 2048 4096 -o results.csv

# Run with specific metric
python bench_gemm_a8w8.py --metric throughput
```

#### Running an Attention Benchmark

```bash
# Run MHA benchmark
python bench_mha.py --batch 32 --hq 32 --hk 32 --n_ctx_q 2048 --n_ctx_k 2048

# Run with model configuration
python bench_pa_prefill.py --model llama3-70b
```

#### Running MoE Benchmarks

```bash
# Run MoE benchmark
python bench_moe.py --model llama3-70b --M 4096 --N 8192 --K 4096 --E 8 --top_k 2

# Run MoE routing benchmark
python bench_moe_routing_sigmoid_top1_fused.py --M 4096 --N 8192 --K 4096
```

#### Running Model Benchmarking Tools

For comprehensive model-level benchmarking:

```bash
# Benchmark all models
cd model_benchmarking_tool
python bench_models.py

# Benchmark specific model
python bench_models.py --model llama3 --M 512 1024 2048 --TP 8

# Benchmark attention models
python bench_attn_models.py --kernel mha mla --model llama3
```

### Output Format

Benchmarks typically output:
- Performance metrics (TFLOPS, GB/s, latency in ms)
- Input parameters used
- Optionally: CSV files with detailed results when using `-o` flag

### Benchmark Schema

The benchmark schema is defined in `triton/bench_schema.yaml`, which specifies:
- Input columns for each benchmark
- Output columns/metrics reported

## Benchmark Schema Reference

The benchmark schema (`triton/bench_schema.yaml`) defines the input and output columns for each benchmark. This schema is used for:
- Validating benchmark inputs
- Generating consistent output formats
- Documentation purposes

### Schema Format

Each benchmark entry in the schema follows this structure:

```yaml
benchmark_name:
  input_columns: [param1, param2, ...]
  output_columns: [metric1, metric2, ...]
```

### Common Output Metrics

- **TFLOPS**: Tera Floating Point Operations Per Second
- **Time(ms)**: Execution time in milliseconds
- **Bandwidth(GB/s)**: Memory bandwidth in gigabytes per second
- **fwd(TFLOPS)**: Forward pass TFLOPS
- **fwd_Time(ms)**: Forward pass time in milliseconds

## Tips for Benchmarking

1. **Warmup**: Always allow sufficient warmup iterations to ensure accurate measurements
2. **Multiple Runs**: Run benchmarks multiple times and average results for consistency
3. **GPU State**: Ensure GPU is in a clean state before benchmarking (no other processes)
4. **Memory**: Be aware of GPU memory constraints when running large benchmarks
5. **Model Configs**: Use appropriate model configurations from `utils/model_configs.json` when available

## Troubleshooting

- **Import Errors**: Ensure all dependencies are installed and the Python path includes the project root
- **CUDA Errors**: Verify GPU availability and CUDA/ROCm installation
- **Out of Memory**: Reduce batch sizes or sequence lengths
- **Slow Performance**: Check GPU utilization and ensure no thermal throttling

## Contributing

When adding new benchmarks:
1. Follow the existing naming convention (`bench_<operation>.py`)
2. Update `bench_schema.yaml` with input/output columns
3. Include command-line argument parsing with `--help` support
4. Support CSV output via `-o` flag
5. Add documentation to this README
