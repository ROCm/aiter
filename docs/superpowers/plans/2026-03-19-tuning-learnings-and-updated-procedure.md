# Triton 3.6 GEMM Tuning — Learnings & Updated Per-Kernel Procedure

**Date**: 2026-03-19
**Status**: Active
**Context**: Learnings from successfully tuning `gemm_a8w8` on Triton 3.6

## Learnings from a8w8 Tuning

### 1. Baseline Collection
- Use `rocprof --stats` (deprecated but reliable), NOT `rocprofv3`
- Command: `rocprof --stats python bench_gemm_<variant>.py --shape <M> <N> <K> --metric time --layout TN`
- Parse `results.stats.csv`: the kernel row has `AverageNs` in the 4th column
- Filter by kernel name substring (e.g., `gemm_a8w8`) to find the right row
- Must install `matplotlib` for bench scripts to work

### 2. Shape Selection
- Primary shapes come from config files (`configs/gemm/gfx950-GEMM-*.json`) and `model_shapes.json`
- One fallback shape for kernels without specific shapes: **M=[8,16,32,64,128,256,512,8192], N=8192, K=8192**
- Fallback config file is named **without** N,K suffix (e.g., `gfx950-GEMM-A8W8.json`) so all untuned shapes hit it

### 3. M-Dependent Block Size Ranges (Critical for Performance)
The tuning search space MUST be tailored to M. Using the full default range wastes time and misses optimal configs for large M.

| M Range | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K |
|---------|-------------|-------------|-------------|
| M <= 16 | 4, 8, 16 | 16, 32 | 256, 512, 1024 |
| M 32-64 | 16, 32, 64 | 32, 64 | 256, 512 |
| M 128-512 | 64, 128, 256 | 64, 128 | 128, 256 |
| M >= 1024 | 128, 256, 512 | 128, 256 | 128, 256 |

### 4. num_stages
- `num_stages=3` is optimal for most shapes on Triton 3.6
- `num_stages=2` is close second, sometimes wins for large M
- Always sweep `--num-stages-range 2 3`

### 5. GPU Assignment
- Pass GPU ID directly to `screen.py` as the `G` positional argument
- screen.py sets `HIP_VISIBLE_DEVICES` internally — do NOT set it at the parent level
- Verify with `rocm-smi --showpidgpus` that processes are on different GPUs

### 6. Validation
- Run same `rocprof --stats python bench_gemm_<variant>.py` on both Triton versions
- Compare `AverageNs` from `results.stats.csv`
- This gives apples-to-apples comparison

---

## Updated Per-Kernel Procedure

For each kernel (e.g., `gemm_a8w8`, `gemm_a16w16`, etc.):

### Phase 1: Baseline (on Triton 3.4, aiter main branch)

```bash
# 1. Switch to main branch, install Triton 3.4
git checkout main
cd /app/triton_3_4 && pip install -e .

# 2. Run UTs to confirm everything works
HIP_VISIBLE_DEVICES=0 python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_<variant>.py -q --tb=no -k "triton- and not gluon"

# 3. Collect baseline with rocprof --stats
for M in 8 16 32 64 128 256 512 8192; do
    HIP_VISIBLE_DEVICES=0 rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_<variant>.py --shape $M 8192 8192 --metric time --layout TN 2>/dev/null 1>/dev/null
    grep "gemm_<variant>" results.stats.csv | head -1
    rm -f results.csv results.stats.csv results.copy_stats.csv results.sysinfo.txt
done
```

Record the `AverageNs` values.

### Phase 2: Tune (on Triton 3.6, feature branch)

```bash
# 1. Switch to feature branch, install Triton 3.6
git checkout alizaidy/gfx950-kernel-fixes
cd /tmp/triton-latest && pip install -e .

# 2. Run UTs to see what breaks (expect LDS OOR for bf16 kernels)
HIP_VISIBLE_DEVICES=0 python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_<variant>.py -q --tb=no -k "triton- and not gluon"

# 3. Tune each M value on a separate GPU with M-appropriate block sizes
cd aiter/ops/triton/utils/_triton/tunning/

# Small M (8-64): use small blocks, one GPU each
for i in 0 1 2 3; do
    M_VALS=(8 16 32 64)
    M=${M_VALS[$i]}
    python screen.py $M 8192 8192 $i ut_<variant>_gemm.py \
        --block-size-m-range 4 8 16 32 64 \
        --block-size-n-range 16 32 64 \
        --block-size-k-range 256 512 1024 \
        --num-stages-range 2 3 > /dev/null 2>&1 &
done

# Medium M (128-512): use medium blocks
for i in 4 5 6; do
    M_VALS=(128 256 512)
    M=${M_VALS[$((i-4))]}
    python screen.py $M 8192 8192 $i ut_<variant>_gemm.py \
        --block-size-m-range 64 128 256 \
        --block-size-n-range 64 128 \
        --block-size-k-range 128 256 \
        --num-stages-range 2 3 > /dev/null 2>&1 &
done

# Large M (8192): use large blocks
python screen.py 8192 8192 8192 7 ut_<variant>_gemm.py \
    --block-size-m-range 128 256 512 \
    --block-size-n-range 128 256 \
    --block-size-k-range 128 256 \
    --num-stages-range 2 3 > /dev/null 2>&1 &

wait  # Wait for all GPUs to finish

# 4. Generate config
python view-screen.py ut_<variant>_gemm.py --n-list 8192 --k-list 8192

# 5. Copy as default (unsuffixed) config
cp gfx950-GEMM-<VARIANT>-N=8192-K=8192.json /app/aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-<VARIANT>.json
```

### Phase 3: Validate (on Triton 3.6 with new config)

```bash
# Same rocprof --stats as baseline
cd /app/aiter
for M in 8 16 32 64 128 256 512 8192; do
    HIP_VISIBLE_DEVICES=0 rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_<variant>.py --shape $M 8192 8192 --metric time --layout TN 2>/dev/null 1>/dev/null
    grep "gemm_<variant>" results.stats.csv | head -1
    rm -f results.csv results.stats.csv results.copy_stats.csv results.sysinfo.txt
done
```

Compare `AverageNs` against Phase 1 baseline. Acceptance: geomean >= 1.0, no shape >3% regression.

---

## Kernel-Specific Notes

### Kernels that need LDS-constrained block sizes (bf16 = 2 bytes/element)
- `gemm_a16w16`, `gemm_a16w16_gated`, `gemm_a16w16_atomic`
- These will hit LDS OOR with existing configs on Triton 3.6
- Must reduce block sizes: max ~BM=128, BN=128, BK=128 for num_stages=2

### Kernels that work without LDS changes (fp8 = 1 byte/element)
- `gemm_a8w8`, `gemm_a8w8_blockscale`, `gemm_a8w8_per_token_scale`
- These pass all tests on Triton 3.6 but still benefit from retuning

### Kernels with additional constraints
- `a8w8_blockscale`: BK must be 128
- `afp4wfp4`: BK must be >= 256
- `a16w8_blockscale`: BK must be multiple of 128

---

## a8w8 Results (Completed)

| M | Triton 3.4 (us) | Triton 3.6 (us) | Delta |
|---|-----------------|-----------------|-------|
| 8 | 56.0 | 12.1 | -78.4% |
| 16 | 55.9 | 12.2 | -78.3% |
| 32 | 56.2 | 13.7 | -75.6% |
| 64 | 56.9 | 16.1 | -71.7% |
| 128 | 56.7 | 21.8 | -61.6% |
| 256 | 57.9 | 30.0 | -48.2% |
| 512 | 60.1 | 43.4 | -27.8% |
| 8192 | 841.3 | 555.9 | -33.9% |

Config: `gfx950-GEMM-A8W8.json` (default, unsuffixed)
