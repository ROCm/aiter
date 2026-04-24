# Opus GEMM Tune (a16w16 / BF16)

## Quick Start

1. Install aiter:
```bash
cd /wksp/aiter
python3 setup.py develop
```

2. Run tuning with a CSV file containing M, N, K shapes:
```bash
python3 csrc/opus_gemm/opus_gemm_tune.py \
    -i aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv \
    -o aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv
```
The first run triggers a JIT build of `module_deepgemm_opus` (~30s).

3. Results are written to the output CSV:

| cu_num | M   | N    | K    | kernelId | splitK | us    | kernelName | tflops | bw    | errRatio |
|--------|-----|------|------|----------|--------|-------|------------|--------|-------|----------|
| 256    | 1   | 5120 | 2880 | 3        | 0      | 47.7  | opus_gemm_256x128x128x32_2x2_16x16x32_0x0x0 | 0.62 | 618.9 | 0.0 |

## Input CSV Format

The tune script reads `M`, `N`, `K` columns from any CSV. Extra columns (bias, dtype, etc.) are ignored.
If a `batch` column exists it is used; otherwise batch defaults to 1.

```csv
M,N,K
1,128,2880
256,5120,2880
```

## Kernel Candidates

Candidates are defined in `opus_gemm_common.py` under `a16w16_kernels_list`.
The tuner sweeps all candidates for each (M, N, K) shape and picks the fastest
one that passes the accuracy threshold.

Current candidates:

| id | BLOCK_SIZE | B_M | B_N | B_K | T_N | MFMA       | Notes |
|----|------------|-----|-----|-----|-----|------------|-------|
| 3  | 256        | 128 | 128 | 32  | 2   | 16x16x32   | small tile, 2 blocks/CU |
| 4  | 256        | 128 | 256 | 32  | 2   | 16x16x32   | wide N |
| 5  | 256        | 256 | 128 | 32  | 2   | 16x16x32   | wide M |
| 6  | 512        | 128 | 128 | 64  | 4   | 16x16x32   | |
| 7  | 512        | 256 | 128 | 64  | 4   | 16x16x32   | |
| 8  | 512        | 128 | 256 | 64  | 4   | 16x16x32   | |
| 9  | 512        | 256 | 256 | 64  | 4   | 16x16x32   | original default |

## Known Limitation

The a16w16 pipeline uses a double-buffered software-pipelined main loop that
steps by 2 K-tiles and prefetches up to `tile+3`. When `ceil_div(K, B_K)` is
**odd**, the last prefetch reads one tile past the valid K range, pulling data
from the next M row in contiguous memory. This corrupts the accumulator for
M > 1.

**Workaround**: candidates with B_K=32 (ids 3-5) produce even loop counts for
K values like 2880 (2880/32=90), avoiding the bug. The tuner automatically
discards candidates that fail the accuracy check, so only correct kernels are
selected.

| K    | B_K=64 loops | B_K=32 loops |
|------|-------------|-------------|
| 2880 | 45 (odd)    | 90 (even)   |
| 2048 | 32 (even)   | 64 (even)   |
| 4096 | 64 (even)   | 128 (even)  |

## Testing

Verify kernel correctness on specific shapes:
```bash
python3 op_tests/test_opus_deepgemm.py -t a16w16 -b 1 -m 256 -n 256 -k 256

# Test with shapes from a CSV file:
python3 op_tests/test_opus_deepgemm.py --csv_file aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv
```

## Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --untune_file` | `aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv` | Input shapes CSV |
| `-o, --tune_file` | `$AITER_OPUS_A16W16_TUNED_CSV` (default `aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv`) | Output tuned CSV (opus-private) |
| `-o2, --profile_file` | `""` | Save all candidates (not just best) |
| `--mp` | GPU count | Number of parallel tuning processes |
| `-k, --splitK` | `False` | Enable split-K sweep (reserved for future) |
| `--errRatio` | `0.05` | Max tolerable error ratio |
| `--batch` | `100` | Shapes per tuning batch |
| `--warmup` | `5` | Warmup iterations |
| `--iters` | `101` | Profiling iterations |
| `--timeout` | `None` | Timeout per task group (seconds) |
| `-v, --verbose` | `False` | Verbose logging |
| `--all` | `False` | Retune already-tuned shapes |
| `--sort` | `True` | Sort output by key columns |
