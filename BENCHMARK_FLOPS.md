# AITER GEMM FLOPS Benchmark

Benchmark A4W4 and A8W8 GEMM operations on AMD GPUs with FLOPS measurements.

## Usage

```bash
git clone --recursive https://github.com/aiter-dev/aiter.git
cd aiter
./run_benchmark.sh
```

Results: `gemm_benchmark_results.csv`


## Implementation

- **Script**: `benchmark_gemm.py` (73 lines)
- **Runner**: `run_benchmark.sh` (5 lines, uses Docker)
- **Warmup**: 10 iterations
- **Measurement**: Median of 50 runs
- **Shapes**: 21 shapes covering inference to training workloads

## Files

- `benchmark_gemm.py` - Benchmark implementation
- `run_benchmark.sh` - Docker runner (auto-installs AITER)
- `gemm_benchmark_results.csv` - Output
- `BENCHMARK_FLOPS.md` - This file

