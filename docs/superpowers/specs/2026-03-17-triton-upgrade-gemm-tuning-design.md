# Triton Upgrade & GEMM Kernel Tuning Pipeline

**Date**: 2026-03-17
**Status**: Draft
**Scope**: Basic GEMM kernels (Phase 1), extensible to all kernel categories

## Problem

Upgrading Triton from 3.4 to latest main introduces async copy, which increases LDS usage. Existing block size configurations exceed the 160KB LDS limit on MI355X (gfx950), causing out-of-resources failures. Reducing block sizes to fit within LDS changes performance characteristics, requiring all kernels to be retuned.

## Goal

Migrate all basic GEMM kernels from Triton 3.4 to latest Triton main while maintaining performance:
- Per-shape regression: < 3%
- Per-kernel geomean across all shapes: >= Triton 3.4 baseline

## Pipeline Overview

Three sequential phases:

1. **Baseline Collection** (Triton 3.4): Benchmark all kernels/shapes, store reference times
2. **Triton Upgrade + Tuning** (latest Triton): Install latest, compute LDS-safe configs, run tuning sweeps
3. **Validation & Comparison**: Re-benchmark, compare against baseline, generate report

## Kernels In Scope

All kernels in `aiter/ops/triton/gemm/basic/`:

| Kernel | Tuning Script | Status |
|--------|--------------|--------|
| gemm_a16w16.py | ut_a16w16_gemm.py | Exists |
| gemm_a16w16_agnostic.py | ut_a16w16_gemm_agnostic.py | NEW |
| gemm_a16w16_atomic.py | ut_a16w16_gemm_atomic.py | NEW |
| gemm_a16w16_gated.py | ut_a16w16_gemm_gated.py | NEW |
| gemm_a16w8_blockscale.py | ut_a16w8_gemm_blockscale.py | NEW (non-preshuffle) |
| gemm_a16wfp4.py | ut_a16wfp4_gemm.py | NEW |
| gemm_a8w8.py | ut_a8w8_gemm.py | Exists |
| gemm_a8w8_blockscale.py | ut_a8w8_gemm_blockscale.py | Exists |
| gemm_a8w8_per_token_scale.py | ut_a8w8_gemm_per_token_scale.py | Exists |
| gemm_a8wfp4.py | ut_a8wfp4_gemm.py | NEW |
| gemm_afp4wfp4.py | ut_afp4wfp4_gemm.py | Exists |
| gemm_afp4wfp4_pre_quant_atomic.py | ut_afp4wfp4_gemm_pre_quant_atomic.py | NEW |

## Shape Collection

Shapes are the union of three sources per kernel:

1. **Config files** (`aiter/ops/triton/configs/gemm/gfx950-GEMM-*.json`):
   - Suffixed files (e.g., `gfx950-GEMM-A8W8-N=7168-K=2048.json`): N,K from filename
   - Unsuffixed files (e.g., `gfx950-GEMM-A8W8.json`): treated as default config, N,K come from model_shapes.json and test scripts only
   - M from keys: `M_LEQ_64` -> M=64, `M_LEQ_128` -> M=128, ..., `any` -> use standard M list
2. **Model shapes** (`op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json`): N,K per model; M from standard batch sizes [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
3. **Benchmark/test scripts**: Explicit (M,N,K) tuples from `get_x_vals()` functions

**Power-of-2 constraint**: `screen.py` requires M to be a power of 2. `collect_shapes.py` rounds non-power-of-2 M values up to the next power of 2 and deduplicates.

Output: `shapes_<kernel>.json` per kernel.

## LDS-Aware Config Space Filtering

MI355X LDS limit: 160KB (163840 bytes).

### Heuristic

The formula uses separate dtype sizes for A and W tiles, since mixed-precision kernels have different element sizes per operand:

```
lds_a = BLOCK_SIZE_M * BLOCK_SIZE_K * A_dtype_size
lds_w = BLOCK_SIZE_N * BLOCK_SIZE_K * W_dtype_size
lds_tiles = (lds_a + lds_w) * num_stages
lds_total = lds_tiles + scale_overhead
lds_total <= 163840
```

Where:
- `num_stages`: from the tuning sweep (2 or 3 preferred; 1 disables pipelining and is generally inferior). LDS holds `num_stages` copies of each tile simultaneously.
- `scale_overhead`: kernel-dependent (see table below)
- For **preshuffled kernels**: the preshuffled operand may bypass LDS entirely (loaded directly into registers via shuffle instructions). In that case, only the non-preshuffled operand's tile contributes to `lds_tiles`. The filter checks whether the kernel uses preshuffle and zeros out the corresponding tile's LDS contribution.

### Per-Kernel Parameters

| Kernel | A_dtype_size | W_dtype_size | Scale Overhead | BK Constraint |
|--------|-------------|-------------|----------------|---------------|
| a16w16 | 2 | 2 | 0 | None |
| a16w16_agnostic | 2 | 2 | 0 | None |
| a16w16_atomic | 2 | 2 | 0 | None |
| a16w16_gated | 2 | 2 | 0 (N is pre-gating full dim) | None |
| a16w8_blockscale | 2 | 1 | `(BM*BK/128 + BN*BK/128) * 1` | BK must be multiple of 128 |
| a16wfp4 | 2 | 0.5 | `BN * BK/32` (w_scales e8m0) | BK >= 64 |
| a8w8 | 1 | 1 | 0 | None |
| a8w8_blockscale | 1 | 1 | `(BM*BK/128 + BN*BK/128) * 1` | BK = 128 only |
| a8w8_per_token_scale | 1 | 1 | `BM * 4` (per-token fp32 scale) | None |
| a8wfp4 | 1 | 0.5 | `BM*BK/32 + BN*BK/32` (e8m0 scales) | BK >= 64 |
| afp4wfp4 | 1 (packed uint8) | 1 (packed uint8) | `BM*BK/32 + BN*BK/32` | BK >= 256 |
| afp4wfp4_pre_quant_atomic | 1 (packed uint8) | 1 (packed uint8) | `BM*BK/32 + BN*BK/32` | BK >= 256 |

**Notes on fp4 packing**: For `afp4wfp4`, both A and W are loaded as packed uint8 (2 fp4 values per byte). BK in the kernel refers to logical K elements, but loads are `BK/2` bytes. The dtype_size of 1 in the table reflects the uint8 load size, with BK adjusted accordingly in the kernel. The LDS formula uses the loaded byte count.

**Notes on gated kernels**: For `a16w16_gated`, N in the shape refers to the full pre-gating dimension. The kernel loads tiles of size `BN * BK` from the full N-dimension weight matrix. No special LDS adjustment needed.

**Notes on num_stages**: The tuning sweep should explore `num_stages` in `[2, 3]` as primary candidates. `num_stages=1` (no pipelining) should be included as a fallback but is generally inferior. `num_stages=3` keeps 3 tiles resident, requiring more LDS but potentially hiding more memory latency.

### Filtering Process

`lds_filter.py`:
1. For each kernel, look up A_dtype_size, W_dtype_size, scale_overhead formula, BK constraints
2. Enumerate all `(BM, BN, BK)` from `screen.py` default ranges
3. For each `num_stages` in `[2, 3]` (primary) and `[1]` (fallback), compute `lds_total`
4. Remove combinations where `lds_total > 163840` for that `num_stages` value
5. For preshuffle kernel variants, zero out the preshuffled operand's LDS contribution
5. Also remove combinations violating kernel-specific BK constraints
6. Output per-`num_stages` filtered ranges as CLI args for `screen.py`

If the heuristic is too conservative (excludes valid configs) or too permissive (includes OOR configs), `screen.py` handles runtime failures gracefully by skipping and moving on.

## GPU Parallelization

8 MI355X GPUs available. `screen.py` accepts a GPU ID argument (`G`).

Strategy:
1. Build work queue: list of `(kernel, M, N, K)` tuples
2. Assign round-robin across GPUs 0-7
3. Launch up to 8 concurrent `screen.py` processes
4. As each finishes, pull next item from queue
5. Each run writes: `screen-{ut_filename}-{M}-{N}-{K}.log` (screen.py's default naming)

Same parallelization for baseline collection (bench scripts instead of screen.py).

## Baseline Collection (Phase 1, Triton 3.4)

For each kernel and shape, run using `rocprofv3` (matching the profiler version used by the existing tuning infrastructure):
```
rocprofv3 --kernel-trace -f csv -o baseline_<kernel>_<M>_<N>_<K> -- python bench_gemm_<variant>.py --shape <M> <N> <K> --metric time --layout TN
```

Parse the `*_kernel_trace.csv` output:
- Filter rows by kernel name substring (e.g., `_gemm_a8w8_kernel`)
- Sum kernel + reduce kernel times if split-K is used
- Handle case where kernel name is not found (log warning, skip shape)
- Store in `baseline_triton34.json`

```json
{
  "gemm_a8w8": {
    "16-4096-7168": {"time_ns": 12345, "kernel_name": "_gemm_a8w8_kernel_..."},
    "32-4096-7168": {"time_ns": 23456, "kernel_name": "..."}
  }
}
```

**Missing bench scripts**: Some kernels (a16w16_agnostic, a16w16_atomic, afp4wfp4_pre_quant_atomic) may not have dedicated bench scripts. For these, baseline collection will use the `ut_*.py` scripts with `rocprofv3` directly, since the ut scripts also exercise the kernel with `run_profile()`.

## Tuning (Phase 2, Latest Triton)

For each kernel and shape:
```
python screen.py <M> <N> <K> <GPU_ID> ut_<kernel>.py \
    --block-size-m-range <LDS-filtered values> \
    --block-size-n-range <LDS-filtered values> \
    --block-size-k-range <LDS-filtered values>
```

After all shapes for a kernel complete:
```
python view-screen.py ut_<kernel>.py --n-list <N1> <N2> ... --k-list <K1> <K2> ...
```

`view-screen.py` requires paired N,K lists (`len(n_list) == len(k_list)`). The orchestrator tracks which (N,K) pairs were tuned per kernel and passes them as matched lists.

Generated JSON configs are copied to `aiter/ops/triton/configs/gemm/`.

## Validation & Comparison (Phase 3)

Re-run baseline collection process with new configs on latest Triton. Compare:

- **Per-shape delta**: `(new_time - baseline_time) / baseline_time * 100%`
- **Flag**: any shape with regression > 3%
- **Geomean ratio**: across all shapes per kernel (must be >= 1.0)
- **Pass/fail**: geomean >= 1.0

Report format:
```
=== gemm_a8w8 ===
Shape           Baseline(ns)  New(ns)   Delta    Status
16-4096-7168    12345         11800     -4.4%    OK
32-4096-7168    23456         24500     +4.5%    REGRESSION
...
Geomean ratio: 1.02x (PASS)
```

## New Tuning Scripts

Each follows the pattern from `ut_template.py` / `ut_a8w8_gemm.py`:

1. Import kernel function and input generator from corresponding test file
2. Parse shape + config from `sys.argv` via `get_input_shape_and_config_list`
3. Generate inputs for the shape
4. Loop over configs, call `run_profile(fn)`

7 new scripts, each ~40 lines.

## Orchestration Scripts

All under `aiter/ops/triton/utils/_triton/tunning/`:

| Script | Purpose |
|--------|---------|
| `orchestrate.py` | Main entry point, drives full pipeline |
| `collect_shapes.py` | Collects shapes from configs, model_shapes.json, tests |
| `lds_filter.py` | Computes LDS-safe block size ranges per kernel |
| `collect_baseline.py` | Runs rocprofv3 benchmarks, parses kernel_trace.csv |
| `run_tuning.py` | Dispatches screen.py across 8 GPUs with work queue |
| `compare_results.py` | Compares baseline vs new, generates report |

### orchestrate.py Interface

```bash
# Phase 1: Baseline on Triton 3.4
python orchestrate.py baseline --kernels all --gpus 0-7

# Phase 2: After installing latest Triton, tune
python orchestrate.py tune --kernels all --gpus 0-7

# Phase 3: Validate and compare
python orchestrate.py validate --kernels all --gpus 0-7

# Target specific kernels
python orchestrate.py tune --kernels a8w8,a16w16 --gpus 0-3

# Dry run: show work queue and estimated time without executing
python orchestrate.py tune --kernels all --gpus 0-7 --dry-run
```

### Checkpointing & Resume

The orchestrator tracks completed work items in `progress.json`:
```json
{
  "baseline": {"gemm_a8w8": {"16-4096-7168": "done", "32-4096-7168": "done"}},
  "tuning": {"gemm_a8w8": {"16-4096-7168": "done"}},
  "validation": {}
}
```

On restart, completed items are skipped. The `--force` flag overrides this to re-run everything.

### Output Directory Structure

All intermediate and final outputs stored under `tunning/results/`:
```
tunning/results/
├── shapes/                    # collected shape files
│   ├── shapes_gemm_a8w8.json
│   └── ...
├── baseline/                  # Triton 3.4 baseline
│   ├── baseline_triton34.json
│   └── traces/                # raw rocprofv3 output
├── tuning/                    # screen.py logs and rocprofv3 output
│   ├── screen-ut_a8w8_gemm-16-4096-7168.log
│   └── ...
├── configs/                   # generated JSON config files
│   ├── gfx950-GEMM-A8W8-N=4096-K=7168.json
│   └── ...
├── validation/                # latest Triton benchmark results
│   └── validation_latest.json
├── progress.json              # checkpointing state
└── report.txt                 # final comparison report
```

## File Organization

```
aiter/ops/triton/utils/_triton/tunning/
├── (existing files unchanged)
├── ut_a16w16_gemm_gated.py                # NEW
├── ut_a16w16_gemm_atomic.py               # NEW
├── ut_a16w16_gemm_agnostic.py             # NEW
├── ut_a16wfp4_gemm.py                     # NEW
├── ut_a8wfp4_gemm.py                      # NEW
├── ut_afp4wfp4_gemm_pre_quant_atomic.py   # NEW
├── ut_a16w8_gemm_blockscale.py            # NEW
├── orchestrate.py                         # NEW
├── collect_shapes.py                      # NEW
├── lds_filter.py                          # NEW
├── collect_baseline.py                    # NEW
├── run_tuning.py                          # NEW
├── compare_results.py                     # NEW
└── results/                               # NEW (output directory)
```

## Extensibility

Once basic GEMMs are validated, the same pipeline extends to all kernel categories:
- **Batched GEMMs**: Add ut scripts, kernel-specific LDS formulas
- **Fused GEMMs**: Same pattern
- **Attention kernels**: Different config parameters (BLOCK_M, BLOCK_N vs BLOCK_SIZE_M, BLOCK_SIZE_N), but same workflow
- **MOE kernels**: Same pattern with expert-aware shapes

Each new kernel category requires:
1. New `ut_*.py` scripts
2. LDS formula for that kernel type in `lds_filter.py`
3. Shape collection logic in `collect_shapes.py`
4. Kernel name matching pattern in `collect_baseline.py`
