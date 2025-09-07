# sytrd (HIP)

Householder symmetric tridiagonalization and benchmark on ROCm. Includes a small Python/LAPACK reference (`lapack_sytrd.py`).

## Algorithm overview
- **larfg**: Given scalar alpha and vector x, computes a Householder vector v and scalar tau so that (I - tau v v^T)[alpha; x] = [beta; 0]. In this code, `larfg_kernel` forms v in-place in the lower part of the current column, stores beta in E and tau in TAU.
- **sytd2**: Reduces a symmetric matrix to tridiagonal by applying reflectors one column at a time. For each j, form v with larfg, compute w = A22 v, apply a small correction, and update A22 <- A22 - v w^T - w v^T. Outputs D (diagonal), E (subdiagonal), and TAU, with v written to the lower triangle. Implemented by `hip_sytd2` (and a fused `small_sytd2_kernel` for tiny sizes).
- **latrd**: Blocked variant that processes a panel of nb columns. For each column, accumulates coupling with prior columns, builds v, forms w that includes cross terms with previously formed V and W, then stores w into W. After the panel, applies a symmetric rank-2k trailing update A <- A - V W^T - W V^T via GEMM. Implemented by `hip_latrd` and helper kernels.
- **sytrd**: Blocked driver that calls latrd on panels, applies the trailing updates, then finishes any remainder with sytd2; finally extracts D and E. This implementation works on the lower triangle (rocBLAS/rocSOLVER use `rocblas_fill_lower`). Implemented by `hip_sytrd_template`.

## Build
1. Edit `Makefile` to set ROCm include/lib paths and `--offload-arch`.
2. Run `make` in this folder to produce `sytrd_bench`.

## Usage
```
./sytrd_bench -n <sizes...> [-p float|double] [-v] [-i ITERS] [-w WARMUP]
```
- `-n` list of matrix sizes (required)
- `-p` precision (default: float)
- `-v` validate against rocSOLVER
- `-i` iterations (default: 10)
- `-w` warmup runs (default: 3)

## Benchmark results

Performance benchmarks were conducted on an AMD Instinct MI300X GPU using ROCm 6.4.3.

| Matrix Size | hip_ssytrd (ms) | rocSOLVER (ms) | Speedup |
|-------------|-----------------|----------------|---------|
| 4           | 0.070           | 0.090          | 1.286   |
| 8           | 0.076           | 0.186          | 2.443   |
| 16          | 0.122           | 0.369          | 3.028   |
| 32          | 0.246           | 0.752          | 3.052   |
| 64          | 0.610           | 1.526          | 2.504   |
| 128         | 2.200           | 3.000          | 1.364   |
| 256         | 4.524           | 6.772          | 1.497   |
| 512         | 8.963           | 11.320         | 1.263   |
| 1024        | 18.797          | 21.844         | 1.162   |
| 2048        | 41.532          | 46.473         | 1.119   |
| 4096        | 108.605         | 115.951        | 1.068   |
| 8192        | 356.602         | 397.679        | 1.115   |
| 16384       | 2097.780        | 2306.929       | 1.100   |

## Python reference (optional)
Requires NumPy and SciPy. Runs a CPU reference tridiagonalization and checks against LAPACK:
```
python lapack_sytrd.py
```
