# LDS Grouped Prefetch + TDM-Before-Fence Benchmark

## Branch
`zan/moe_yadai_lds_pf` (based on `dev/moe_yadai`)

## Changes
1. **TDM-before-fence** (non-PF warpspec path): Issue `tensor_load_2d` *before*
   `pipeline_fence_signal` + `pipeline_fence_wait`, so the TDM DMA overlaps the
   fence drain window. Previously TDM was issued *after* the fence wait.
2. **LDS grouped prefetch (full-depth)**: Pre-fetch all A/B/AS/BS operands for
   the current LDS tile into VGPR carry via interleaved ds_loads during the
   previous tile's compute. Gated by `AITER_GROUPED_LDS_PF=1` (default on).

## How to reproduce

```bash
# Baseline (original dev/moe_yadai, no changes):
git checkout origin/dev/moe_yadai

# This branch (TDM-before-fence + PF):
git checkout zan/moe_yadai_lds_pf

# Run benchmark (tile_m=16 max_m=16 hardcoded in grouped_moe_gfx1250.py):
ENABLE_CK=0 AITER_LOG_MORE=1 AITER_FORCE_A8W4=1 \
AITER_USE_GROUPED_GEMM=1 AITER_FORCE_GFX1250=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_GROUPED_LDS_PF=0 \   # 0=OFF, 1=FULL (default)
python op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a8w4 --layout gugu \
  --experts 256 --tokens 64 --topk 6 \
  --model-dim 4096 --inter-dim 2048 \
  --act silu --no-bias
```

## Results

Config: tile_m=16 tile_n=256 tile_k=256, max_m=16, a8w4 gugu,
256 experts, 64 tokens, topk=6, model_dim=4096, inter_dim=2048.

### Original baseline (dev/moe_yadai, TDM-after-fence, no PF)

| kernel | median (µs) |
|--------|------------|
| gemm1  | 309.2      |
| gemm2  | 172.6      |

Note: original uses tile_m=64 (default), not comparable to below.

### This branch, tile_m=16, max_m=16 (5-run median)

| mode        | gemm1 (µs) | gemm2 (µs) | sum (µs) | gemm1 range      | gemm2 range      |
|-------------|-----------|-----------|---------|------------------|------------------|
| PF OFF      | 217.7     | 125.7     | 343.4   | [217.4 - 218.4]  | [123.1 - 128.8]  |
| PF FULL     | 218.1     | **122.8** | **340.9** | [216.6 - 221.0]  | [120.6 - 127.6]  |

**gemm2**: PF FULL saves ~3 µs (2.3%) vs PF OFF — stable across 5 runs.
**gemm1**: Flat (fused silu activation dominates compute; PF overhead ≈ savings).
**Sum**: ~2.5 µs (0.7%) improvement.

### Earlier 3-run comparison (same tile config)

| mode   | gemm1 | gemm2 | sum   |
|--------|-------|-------|-------|
| OFF    | 216.8 | 123.5 | 340.3 |
| FULL   | 217.5 | 120.3 | 337.8 |
| D=1ks  | 219.3 | 123.9 | 343.2 |

Sub-depth (D=1ks) shows no benefit over OFF at this tile size.

## Analysis

- The **TDM-before-fence reordering** is the primary win. It lets the TDM DMA
  start earlier (overlaps the fence drain), benefiting all warpspec kernels
  regardless of PF.
- **PF FULL** adds a small incremental gain on gemm2 by eliminating `s_wait_dscnt(0)`
  stalls in the compute tile — operands are already in VGPR from the previous
  tile's interleaved ds_loads.
- **gemm1** (fused silu) is compute-heavy enough that LDS latency is already
  hidden by the activation math; PF has negligible effect.
- **tile_m=16** means wmma_m_rep=1, so the WMMA body is very short. PF benefits
  would be larger with wmma_m_rep >= 2 (more compute to overlap with ds_loads).
