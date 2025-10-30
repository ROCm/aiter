# AITER GEMM FLOPS Benchmark

Benchmark A4W4 and A8W8 GEMM operations (including blockscale variants) on AMD GPUs with FLOPS measurements.

## Usage

```bash
git clone --recursive https://github.com/aiter-dev/aiter.git
cd aiter
./run_benchmark.sh
```

## Operations Benchmarked

- **A4W4**: 4-bit GEMM (ASM kernel, per-1×32 scale)
- **A4W4-BLK**: 4-bit blockscale (CK kernel)
- **A8W8-INT8/FP8**: 8-bit GEMM (CK kernel, per-token scale)
- **A8W8BLK-INT8/FP8**: 8-bit blockscale (CK kernel, 128×128 blocks)