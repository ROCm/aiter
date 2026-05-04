# Unified Attention Testing Suite

Performance testing tools for CK Tile vs Triton unified attention implementations.

## Quick Start

### Single Shape Test (Correctness)
```bash
python3 test_single_shape.py -b 256 -sq 1 -sk 8192 -hq 64 -hk 8 -d 64 --test
```

### Single Shape Benchmark
```bash
python3 test_single_shape.py -b 256 -sq 1 -sk 8192 -hq 64 -hk 8 -d 64 \
  --warmup 20 --iters 50
```
Note: CUDA graph mode is enabled by default (use `--no-graph` to disable).

### Profiling with ROCProfiler v3
```bash
./rocprof_bench.sh -b 256 -sq 1 -sk 8192 -hq 64 -hk 8 -d 64 \
  --warmup 20 --iters 50
```
Generates summary and automatically cleans up trace files.

### Batch Testing from CSV
```bash
python3 bench_ck_vs_triton_csv_rows.py pawel-2d-3d_50rows_verified.csv
```

## Command-Line Arguments

### Required Shape Parameters
- `-b, --batch` - Number of sequences
- `-sq, --seqlen-q` - Query sequence length (1 = decode)
- `-sk, --seqlen-k` - KV/context length
- `-hq, --num-q-heads` - Query heads
- `-hk, --num-kv-heads` - KV heads (for GQA)
- `-d, --head-size` - Head dimension (64 or 128)

### Testing Options
- `--test` - Run correctness check (torch.testing.assert_allclose)
- `--warmup N` - Warmup iterations (default: 10)
- `--iters N` - Benchmark iterations (default: 50)
- `--no-graph` - Disable CUDA graph mode (graph mode is default)
- `--only-ck` - Run only CK kernel
- `--only-triton` - Run only Triton kernel

### Optional Parameters
- `--block-size N` - KV cache block size (default: 32)
- `--dtype {bf16,fp16}` - Data type (default: bf16)
- `--window-left N` - Sliding window (default: -1)
- `--softcap F` - Softcap value (default: 0.0)

## Files

- `test_single_shape.py` - Single-shape testing and benchmarking
- `rocprof_bench.sh` - ROCProfiler v3 wrapper with auto-cleanup
- `parse_kernel_trace.py` - Kernel trace parser (handles 2D/3D variants)
- `parse_rocprof.py` - Alternative stats parser
- `bench_ck_vs_triton_csv_rows.py` - Batch testing from CSV
- `pawel-2d-3d_50rows_verified.csv` - Reference test shapes

## Performance Summary

See `PERFORMANCE.md` for detailed results.

**TL;DR:**
- **High batch (b≥64)**: CK wins 1.1-1.7x
- **Low batch (b≤32)**: Triton wins 1.4-4x (uses 3D split-KV kernel)
- **Crossover**: Between b=32 and b=64

## Hardware

Tested on AMD MI350 (gfx950) with ROCm 7.0.0.
