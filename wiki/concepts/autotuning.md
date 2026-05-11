---
title: "Autotuning Pipeline"
last_verified: 2026-04-06
source_files:
  - docs/autotuning_pipeline.md
  - aiter/utility/base_tuner.py
  - aiter/configs/
tags: [autotuning, configs, performance, ci]
---

# Autotuning Pipeline

## Overview
aiter uses an automated tuning system to find the best kernel configuration for each operator shape and GPU. The tuning pipeline benchmarks multiple kernel variants and records the fastest one in CSV config files. These configs are loaded at runtime to dispatch to the optimal kernel.

## How It Works

1. **Input shapes** are defined (M, N, K for GEMM; other params for attention/MoE)
2. **Tuning scripts** in `aiter/csrc/` benchmark each kernel variant for each shape
3. **Results** are written to CSV files in `aiter/configs/` with columns: cu_num, M, N, K, kernelName, splitK
4. At **runtime**, operators call `get_GEMM_config(M, N, K)` which indexes the CSV by (cu_num, M, N, K) and returns the best config
5. If no tuned config exists, a **default config** is used (from `*_untuned_*.csv` files)

## Config File Format

Each CSV row represents a tuned configuration:
- `cu_num` -- GPU compute unit count (arch-specific, e.g., 304 for MI300X)
- `M`, `N`, `K` -- matrix dimensions
- `kernelName` -- the specific kernel variant to use (determines ASM vs CK vs CK Tile)
- `splitK` -- split-K parallelism factor

Example: shape M=1, N=7168, K=2048 on a 304-CU GPU might map to kernel `ck_gemm_a8w8_blockscale` with splitK=1.

## CI Pipelines

Two CI pipeline modes:

### Manual Pipeline
- GitHub Actions: `operators-tuning.yaml`
- Triggered manually from Actions page
- Allows selecting specific shapes to tune (e.g., `ck_gemm_a8w8, ck_gemm_a8w8_blockscale`)
- Steps: pre-tune benchmark -> tune -> diff CSVs -> post-tune benchmark -> upload artifacts
- Additional arguments supported via `base_tuner.py` options

### Scheduled Pipeline
- Runs nightly or weekly
- Generates all tuned CSV files automatically
- Results uploaded to the aiter repository

## Config Files (19 total)

Tuned/untuned pairs for each GEMM variant:
- A8W8: `a8w8_tuned_gemm.csv`, `a8w8_untuned_gemm.csv`
- A8W8 block-scale: `a8w8_blockscale_tuned_gemm.csv`, `a8w8_blockscale_untuned_gemm.csv`
- A8W8 pre-shuffle: `a8w8_bpreshuffle_tuned_gemm.csv`, `a8w8_bpreshuffle_untuned_gemm.csv`
- A8W8 block-scale + pre-shuffle: `a8w8_blockscale_bpreshuffle_tuned_gemm.csv`, `..._untuned_...`
- A4W4 block-scale: `a4w4_blockscale_tuned_gemm.csv`, `a4w4_blockscale_untuned_gemm.csv`
- BF16: `bf16_tuned_gemm.csv`, `bf16_untuned_gemm.csv`
- BF16 batched: `bf16_tuned_batched_gemm.csv`, `bf16_untuned_batched_gemm.csv`
- A8W8 batched: `a8w8_tuned_batched_gemm.csv`, `a8w8_untuned_batched_gemm.csv`
- ASM A8W8: `asm_a8w8_gemm.csv`
- Fused MoE: `tuned_fmoe.csv`, `untuned_fmoe.csv`

## Related Pages
- [[operators/gemm]] -- GEMM operators that consume these configs
- [[concepts/backend-selection]] -- how configs drive backend selection
- [[operators/moe]] -- MoE-specific tuned configs

## Source Files
- `docs/autotuning_pipeline.md` -- CI pipeline documentation
- `aiter/utility/base_tuner.py` -- base tuner class and CLI arguments
- `aiter/configs/` -- all 19 CSV config files
