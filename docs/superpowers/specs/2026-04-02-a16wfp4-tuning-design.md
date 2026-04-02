# gemm_a16wfp4 Triton 3.6 Tuning Design

**Date**: 2026-04-02
**Status**: Draft
**Scope**: gemm_a16wfp4 kernel (non-atomic, atomic, preshuffle variants)

## Problem

The gemm_a16wfp4 kernel regresses ~0.84x geomean on Triton 3.6 vs 3.4 for both non-atomic and atomic modes. A previous tuning attempt (v2) improved non-atomic large M but catastrophically worsened N=512 shapes (+169%) and atomic mode (0.718x) because:

1. Both modes share the same config file but have different optimal configs
2. The search space missed BK=512 + high split-K combos critical for small N shapes
3. screen.py's rocprofv3 batching is derailed by PassManager compilation crashes for certain (block_size, warps) combos

## Solution

### 1. Separate Config Files

Modify `_get_config()` in `_triton_kernels/gemm/basic/gemm_a16wfp4.py` to accept `atomic_add` parameter:

```python
def _get_config(M, N, K, shuffle=False, atomic_add=False):
    shuffle_suffix = "_PRESHUFFLED" if shuffle else ""
    atomic_suffix = "-ATOMIC" if atomic_add else ""
    config_name = f"GEMM-A16WFP4{shuffle_suffix}{atomic_suffix}"
    return get_gemm_config(config_name, M, N, 2 * K)
```

Three config families (atomic + preshuffle is not a valid combo):
- `gfx950-GEMM-A16WFP4*.json` — non-atomic, non-preshuffle
- `gfx950-GEMM-A16WFP4-ATOMIC*.json` — atomic, non-preshuffle
- `gfx950-GEMM-A16WFP4_PRESHUFFLED*.json` — preshuffle (prequant, no atomic)

The wrapper `gemm_a16wfp4_()` passes `atomic_add` to `_get_config` when no config is provided. Existing configs are copied to all families initially so behavior is unchanged until tuning.

### 2. Crash Resilience in run_profile()

Modify `_utils.py` `run_profile()` to catch Triton compilation crashes:

```python
def run_profile(fn: callable, n_run: int = 250):
    di = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    try:
        for _ in range(n_run):
            cache.zero_()
            di.synchronize()
            fn()
            di.synchronize()
    except Exception as e:
        print(f"[run_profile] Skipping config: {type(e).__name__}: {e}",
              file=sys.stderr, flush=True)
        return
    d = torch.empty(128, dtype=torch.float32, device="cuda")
    cache.zero_()
    di.synchronize()
```

This catches PassManager::run failed and other compilation errors per-config, letting good configs in the same batch still produce results via screen.py.

### 3. New Tuning Script

Create `ut_a16wfp4_gemm_atomic.py` — identical to `ut_a16wfp4_gemm.py` but with `atomic_add=True` and `dtype=torch.float32` (atomic bf16 has precision issues).

### 4. Search Space

BK includes 128 (unlike v2 which started at 256). Full split-K range for M < 1024.

| M Range | BM | BN | BK | Split-K |
|---------|----|----|-----|---------|
| M ≤ 16 | 4, 8, 16 | 16, 32, 64, 128 | 128, 256, 512, 1024 | default range |
| M 32-64 | 16, 32, 64 | 32, 64, 128 | 128, 256, 512 | default range |
| M 128-512 | 64, 128, 256 | 64, 128, 256 | 128, 256, 512 | default range |
| M ≥ 1024 | 128, 256, 512 | 128, 256 | 128, 256, 512 | 1 only |

Other params: `--num-stages-range 1 2 3`, `--num-warps-range 1 2 4 8`, `--matrix-instr-nonkdim-range 16`, `--cache-modifier-range 0 1`, `--timeout 900`.

### 5. NK Pairs

Non-atomic and atomic (3 pairs, config K = 2 × real K):
- N=512, K=3584 (config K=7168)
- N=7168, K=1024 (config K=2048)
- N=8192, K=4096 (config K=8192, fallback)

Preshuffle (2 pairs):
- N=2112, K=7168 (config K=14336)
- N=8192, K=8192 (config K=16384, fallback)

### 6. Execution Order

1. Implement code changes (_get_config, _utils.py, ut script, initial config files)
2. Test crash resilience with known crashing config
3. Tune non-atomic (3 NK × 8 M on GPUs 2-5)
4. Tune atomic (3 NK × 8 M)
5. Tune preshuffle (2 NK × 8 M)
6. Per-shape merge: keep old config entries where they're still better on 3.6
7. Final validation: UTs + bench for all modes
8. Commit

### 7. Validation

Benchmark with `rocprof --stats` using:
- `bench_gemm_a16wfp4.py --shape M N K --metric time --layout TN`
- `bench_gemm_a16wfp4.py --shape M N K --metric time --layout TN --atomic`
- `bench_gemm_a16wfp4.py --shape M N K --metric time --layout TN -preshuffle` (if added)

Baselines already collected on Triton 3.4/main for non-atomic and atomic.

### 8. Success Criteria

- Geomean >= 1.0 for each mode independently
- No individual shape regressing more than 0.5% + 500ns vs Triton 3.4 baseline
- All UTs pass
