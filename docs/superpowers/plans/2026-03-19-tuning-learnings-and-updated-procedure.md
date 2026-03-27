# Triton 3.6 GEMM Tuning — Learnings & Updated Per-Kernel Procedure

**Date**: 2026-03-19 (updated 2026-03-24)
**Status**: Active
**Context**: Learnings from tuning `gemm_a8w8`, `gemm_a16w16`, `gemm_afp4wfp4` on Triton 3.6

## Learnings

### 1. Baseline & Validation Collection
- Use `rocprof --stats` (deprecated but reliable), NOT `rocprofv3`
- **MUST be sequential on a single GPU** — parallel runs cause `results.stats.csv` collisions and corrupt data
- Command: `rocprof --stats python bench_gemm_<variant>.py --shape <M> <N> <K> --metric time --layout TN`
- Parse `results.stats.csv`: kernel row has `AverageNs` in 4th column
- Filter by kernel name substring (e.g., `gemm_a8w8`) to find the right row
- Must install `matplotlib` for bench scripts to work
- **Kill all stray GPU processes** before collecting data — verify with `rocm-smi --showpidgpus`

### 2. Shape Selection
- Primary shapes come from config files (`configs/gemm/gfx950-GEMM-*.json`) and `model_shapes.json`
- One fallback shape for kernels without specific shapes: **M=[8,16,32,64,128,256,512,8192], N=8192, K=8192**
- Fallback config file is named **without** N,K suffix (e.g., `gfx950-GEMM-A8W8.json`) so all untuned shapes hit it

### 3. M-Dependent Block Size Ranges (Critical for Performance)
The tuning search space MUST be tailored to M. Using the full default range wastes time and misses optimal configs for large M.

**For fp8 kernels (1 byte/element):**

| M Range | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K |
|---------|-------------|-------------|-------------|
| M <= 16 | 4, 8, 16 | 16, 32 | 256, 512, 1024 |
| M 32-64 | 16, 32, 64 | 32, 64 | 256, 512 |
| M 128-512 | 64, 128, 256 | 64, 128 | 128, 256 |
| M >= 1024 | 128, 256, 512 | 128, 256 | 128, 256 |

**For bf16 kernels (2 bytes/element) — MUST include BK=64:**

| M Range | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K |
|---------|-------------|-------------|-------------|
| M <= 16 | 4, 8, 16 | 16, 32 | 64, 128, 256 |
| M 32-64 | 16, 32, 64 | 32, 64 | 64, 128, 256 |
| M 128-512 | 64, 128, 256 | 64, 128 | 64, 128 |
| M >= 1024 | 128, 256, 512 | 128, 256 | **64**, 128 |

**For fp4 kernels (packed uint8, BK >= 256):**

| M Range | BLOCK_SIZE_M | BLOCK_SIZE_N | BLOCK_SIZE_K |
|---------|-------------|-------------|-------------|
| M <= 16 | 4, 8, 16 | 16, 32, 64, 128, 256 | 256, 512, 1024 |
| M 32-64 | 16, 32, 64 | 32, 64, 128 | 256, 512 |
| M 128-512 | 64, 128, 256 | 64, 128, 256 | 256, 512 |
| M >= 1024 | 128, 256 | 128, 256 | 256, 512 |

**Critical learning from a16w16:** For large M with bf16, reducing BK from 128 to **64** halves LDS per tile, enabling BM=256 and BN=128/256 with num_stages=3. Example: `(256*64 + 128*64) * 2 * 3 = 147456 bytes` fits in 160KB LDS.

### 4. num_stages
- `num_stages=3` is optimal for most shapes on Triton 3.6
- `num_stages=2` is close second, sometimes wins for large M with large BK
- `num_stages=1` should also be swept — it occasionally wins for specific shapes
- Always sweep `--num-stages-range 1 2 3`
- **Critical: num_stages=3 + split-K can be dramatically better than num_stages=2 + split-K** (e.g., M=128 N=8192 K=32768 per_token_scale: 87us→63us just from stages 2→3)

### 5. split-K (NUM_KSPLIT)
- For medium/large M with large K, split-K can help significantly — do NOT restrict to SPK=1 for medium M
- Always include split-K values in the sweep for shapes where K is large relative to N
- Valid split-K values depend on K and BLOCK_SIZE_K: `SPLITK_BLOCK_SIZE = ceil(K/SPK)` must be >= BK
- For per_token_scale kernel: split-K=4 with stages=3 was optimal for M=128 K=32768

### 6. matrix_instr_nonkdim
- **nonkdim=16**: Default, works well for most shapes
- **nonkdim=32**: Critical for fp4 kernels with large M and large N,K — can be dramatically faster
- Also helps fp8 kernels for medium-large M (M>=64): consistently chosen as best for per_token_scale M=64-512
- Always sweep `--matrix-instr-nonkdim-range 16 32` for fp4 kernels and for fp8 kernels with M>=64

### 6. GPU Assignment
- Pass GPU ID directly to `screen.py` as the `G` positional argument
- screen.py sets `HIP_VISIBLE_DEVICES` internally — do NOT set it at the parent level
- Verify with `rocm-smi --showpidgpus` that processes are on different GPUs

### 7. BLOCK_M Constraints
- The tuner may pick BLOCK_M > M (e.g., BM=32 for M=16). This works via masking and is often faster.
- **Do NOT blindly enforce BLOCK_M <= M** — it regresses many shapes (tested: 15 regressions from BM<=M, 25 from BM<=M + min BM)
- Instead: after tuning, compare constrained vs unconstrained per-shape and selectively apply the constraint only where it improves performance
- In our testing, only 2 out of 136 shapes benefited from the BM<=M constraint

### 8. fp4 K Naming Convention
- For afp4wfp4: `_get_config` does `K = 2 * K` internally where K is the tensor's K dimension (K/2 uint8 values for K fp4 elements)
- The benchmark `--shape M N K` uses K = number of fp4 elements
- The config filename K matches the benchmark K directly (NOT K*2)
- **Do NOT rename config files with K*2** — this was a mistake that caused catastrophic regressions

### 9. Parallel vs Sequential
- **Tuning** (screen.py): Can run in parallel across GPUs — each GPU runs a different shape, no file conflicts
- **Baseline/validation** (rocprof --stats): MUST be sequential on single GPU — parallel runs corrupt `results.stats.csv`
- When running multiple N,K pairs on different GPUs for tuning, use separate subshells or `wait` between shapes on the same GPU

### 10. Verify Long Tasks Early
- **Always check progress 1-2 minutes after launching any task that will run >10 minutes**
- Verify screen logs are producing `screencase` entries (not just `Running case` lines with 0 results)
- Common silent failures: missing SPLITK_BLOCK_SIZE in config, wrong kernel name in rprof.py filter, stale GPU contexts
- A task running for 10+ minutes with 0 results is broken — kill and investigate immediately

---

## Updated Per-Kernel Procedure

### Phase 1: Baseline (on Triton 3.4, aiter main branch)

```bash
# 1. Kill stray processes, switch to main branch, install Triton 3.4
pkill -9 -f "screen.py|rocprof|bench_gemm" 2>/dev/null
git checkout main
cd /app/triton_3_4 && pip install -e .

# 2. Run UTs to confirm everything works
HIP_VISIBLE_DEVICES=0 python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_<variant>.py -q --tb=no

# 3. Collect baseline SEQUENTIALLY with rocprof --stats
for M in 8 16 32 64 128 256 512 8192; do
    HIP_VISIBLE_DEVICES=0 rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_<variant>.py \
        --shape $M <N> <K> --metric time --layout TN 2>/dev/null 1>/dev/null
    grep "gemm_<variant>" results.stats.csv | head -1
    rm -f results.csv results.stats.csv results.copy_stats.csv results.sysinfo.txt
done
```

### Phase 2: Tune (on Triton 3.6, feature branch)

```bash
# 1. Switch to feature branch, install Triton 3.6
git checkout alizaidy/gfx950-kernel-fixes-cherry-picked
cd /tmp/triton-latest && pip install -e .

# 2. Run UTs to see what breaks
HIP_VISIBLE_DEVICES=0 python -m pytest op_tests/triton_tests/gemm/basic/test_gemm_<variant>.py -q --tb=no

# 3. Tune — one shape per GPU, use M-appropriate block sizes
cd aiter/ops/triton/utils/_triton/tunning/

# For each (N,K) pair, tune all M values:
for i in 0 1 2 3 4 5 6 7; do
    M_VALS=(8 16 32 64 128 256 512 8192)
    M=${M_VALS[$i]}
    # Select block ranges based on M and kernel dtype (see tables above)
    python screen.py $M <N> <K> $i ut_<variant>_gemm.py \
        --block-size-m-range <BM_RANGE> \
        --block-size-n-range <BN_RANGE> \
        --block-size-k-range <BK_RANGE> \
        --matrix-instr-nonkdim-range 16 32 \
        --num-stages-range 1 2 3 > /dev/null 2>&1 &
done
wait

# 4. Generate config
python view-screen.py ut_<variant>_gemm.py --n-list <N> --k-list <K>

# 5. Copy config (use correct K naming for the kernel type)
cp gfx950-GEMM-<VARIANT>-N=<N>-K=<K>.json /app/aiter/aiter/ops/triton/configs/gemm/
```

### Phase 3: Validate (on Triton 3.6 with new config)

```bash
# SEQUENTIAL on single GPU — same method as baseline
cd /app/aiter
for M in 8 16 32 64 128 256 512 8192; do
    HIP_VISIBLE_DEVICES=0 rocprof --stats python op_tests/op_benchmarks/triton/bench_gemm_<variant>.py \
        --shape $M <N> <K> --metric time --layout TN 2>/dev/null 1>/dev/null
    grep "gemm_<variant>" results.stats.csv | head -1
    rm -f results.csv results.stats.csv results.copy_stats.csv results.sysinfo.txt
done
```

### Phase 4: Post-tuning Optimization

After initial tuning and validation:
1. Identify regressions vs baseline
2. For regressed shapes, try retuning with wider search (more BN values, all num_stages, both nonkdim)
3. Compare constrained (BM<=M) vs unconstrained per-shape — selectively apply constraint only where it helps
4. Re-validate any changes sequentially

---

## Completed Kernel Results

### gemm_a8w8 — 2.70x geomean, 0 regressions
| M | Triton 3.4 (us) | Triton 3.6 (us) | Delta |
|---|-----------------|-----------------|-------|
| 8 | 55.4 | 12.3 | -77.7% |
| 16 | 55.9 | 12.3 | -78.0% |
| 32 | 55.9 | 13.8 | -75.3% |
| 64 | 56.3 | 16.4 | -70.8% |
| 128 | 56.4 | 21.7 | -61.5% |
| 256 | 57.6 | 30.1 | -47.8% |
| 512 | 56.9 | 43.7 | -23.2% |
| 8192 | 841.7 | 550.6 | -34.6% |

### gemm_a16w16 — 2.58x geomean, 2 regressions (72 shapes)
Regressions:
- M=256 N=128 K=2880: 8.7 -> 10.1us (+15.8%)
- M=512 N=128 K=2880: 9.2 -> 10.2us (+11.2%)

### gemm_afp4wfp4 — 1.73x geomean, 7 regressions (56 shapes)
Regressions:
- M=8 N=1280 K=8192: 4.6 -> 4.9us (+5.3%)
- M=8 N=2112 K=7168: 4.5 -> 4.8us (+6.1%)
- M=16 N=2112 K=7168: 4.8 -> 7.3us (+52.1%)
- M=64 N=7168 K=2048: 6.4 -> 6.8us (+6.1%)
- M=8 N=8192 K=8192: 9.8 -> 10.8us (+10.8%)
- M=8192 N=8192 K=28672: 1203.3 -> 1421.1us (+18.1%)
- M=8 N=16384 K=16384: 24.2 -> 27.2us (+12.4%)

### Overall — 2.20x geomean across 136 shapes
- 121 improved, 9 regressed, 6 neutral
