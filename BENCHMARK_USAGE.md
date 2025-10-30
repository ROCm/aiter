# AITER GEMM Benchmark - Enhanced CLI

## Changes Made

Added command-line arguments to `benchmark_gemm.py`:
- `--shapes`: Specify custom M,N,K dimensions
- `--kernel`: Select specific kernel to benchmark  
- `--output`: Specify output CSV file (supports append)
- `--cold-iters`: Warmup iterations (default: 10)
- `--hot-iters`: Measurement iterations for median calculation (default: 50)

## Usage Examples

### 1. Default behavior (all kernels, all shapes)
```bash
./run_benchmark.sh
```

### 2. Benchmark specific kernel with specific shape
```bash
python benchmark_gemm.py --kernel gemm_a4w4_asm --shapes "2048,8192,8192"
```

### 3. Multiple shapes for one kernel
```bash
python benchmark_gemm.py --kernel gemm_a4w4_asm --shapes "1024,1280,8192 2048,1280,8192"
```

### 4. Custom output file (creates new)
```bash
python benchmark_gemm.py --kernel gemm_a8w8_fp8 --shapes "2048,8192,8192" --output custom.csv
```

### 5. Append to existing CSV
```bash
python benchmark_gemm.py --kernel gemm_a4w4_asm --shapes "2048,8192,8192" --output results.csv
python benchmark_gemm.py --kernel gemm_a8w8_fp8 --shapes "2048,8192,8192" --output results.csv
# Both results now in results.csv!
```

### 6. Custom warmup and measurement iterations
```bash
python benchmark_gemm.py --kernel gemm_a4w4_asm --shapes "2048,8192,8192" --cold-iters 20 --hot-iters 100
# 20 warmup iterations, 100 measurement iterations (median of 100)
```

## Available Kernels

- `gemm_a4w4_asm` - A4W4 ASM (fastest, GFX950 only)
- `gemm_a4w4_blockscale` - A4W4 CK blockscale (portable)
- `gemm_a8w8_i8` - A8W8 INT8 
- `gemm_a8w8_fp8` - A8W8 FP8
- `gemm_a8w8_blockscale_i8` - A8W8 INT8 blockscale
- `gemm_a8w8_blockscale_fp8` - A8W8 FP8 blockscale

## CSV Format

```csv
M,N,K,dtype,backend,time_us,TFLOPS,GB/s
2048,8192,8192,a4w4,ASM,67.52,4070.999,1118.13
2048,8192,8192,a8w8_fp8,CK,125.12,2196.897,938.62
```

Appended rows integrate seamlessly with existing data!
