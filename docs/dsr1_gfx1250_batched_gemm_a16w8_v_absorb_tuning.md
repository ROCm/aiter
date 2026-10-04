# DSR1 V-absorb BMM on gfx1250: `batched_gemm_a16w8` tuning (B=128, N=128, K=512)

DeepSeek-R1 MLA decode V up-projection (W_UV absorb): bf16 activations `(B=128 heads, M tokens,
K=512)` times fp8 weights `(B, N=128, K)`, per-tensor weight scale. It runs on the gfx1250 Gluon
kernel `_batched_gemm_a16w8_gfx1250_persistent_kernel`. All 13 M buckets (1 to 8192) were tuned
with `aiter/ops/triton/utils/_triton/tuning` on 4x gfx1250 (256 CUs each).

## What this branch adds

- `aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/BATCHED_GEMM-A16W8-B=128-N=128-K=512.json`:
  the tuned table (`M_LEQ_1` to `M_LEQ_8192`). `any` is the `any` entry of `DEFAULT.json`.
- `aiter/ops/triton/configs/gfx1250/gluon/gemm/batched_gemm_a16w8/DEFAULT.json`: the family
  default with `"waves_per_eu": 1` added so the tuner sweeps that key. It doesn't change runtime
  behavior, because the wrapper already defaults it to 1.

## Needed from outside this branch (not on main yet)

- **The kernel, wrapper and this config directory** come from `8f753621a` ("Add gluon batched
  GEMM for gfx1250 and tune it"), which is only on `satya/temp_delete`. When it lands, its
  `DEFAULT.json` will conflict with this one, since both add the file. Keep the
  `waves_per_eu` line.
- **The wrapper has to pass `B`** or it never reads this B=128 file and keeps reading
  `BATCHED_GEMM-A16W8-N=128-K=512.json`. `batched_gemm_bf16` already does this. The change is in
  `aiter/ops/triton/gemm/batched/batched_gemm_a16w8.py`:
  `get_gemm_config("BATCHED_GEMM-A16W8", M, N, K, B=B, backend="gluon")`.
- **The tuner registration** is a `batched_gemm_a16w8` entry in `tuning/kernels.py` plus
  `WG_PER_CU`, `STORE_MODE` and `C_PAD` in `SEARCH_SPACE`. It is built on the refactored tuner
  (`ee7e304fb`), which is also not on main.

## Results

These numbers come from a steady-state A/B on one GPU after install. For each bucket the old
config, today's `N=128-K=512` entry, was timed against the new one, alternating, 3 rounds each,
median in microseconds. Timing used the tuner's CUDA-graph timer over cold input copies, after a
warm-up.

Validation:
- Every new entry's output was checked against an fp32 CPU reference in two layouts: the
  contiguous tuning layout, and ATOM's layout (strided X, `transpose_bm` output). All passed.
- `op_tests/triton_tests/gemm/batched/test_batched_gemm_a16w8.py`: 170 passed.

| M bucket | old config | old us | new config | new us | speedup |
|---:|---|---:|---|---:|---:|
| 1 | BM16 BN64 BK256 w4 buf2 wg2 st2 wpe1 | 2.94 | BM16 BN16 BK512 w1 buf2 wg4 st1 wpe4 | 2.35 | 1.25x |
| 4 | BM16 BN64 BK256 w4 buf2 wg1 st2 wpe2 | 3.04 | BM16 BN16 BK512 w1 buf2 wg4 st1 wpe4 | 2.46 | 1.24x |
| 8 | BM16 BN64 BK256 w4 buf2 wg1 st2 wpe2 | 3.10 | BM16 BN16 BK512 w1 buf2 wg4 st1 wpe4 | 2.54 | 1.22x |
| 16 | BM16 BN64 BK256 w4 buf2 wg2 st2 wpe1 | 3.21 | BM16 BN16 BK256 w1 buf3 wg4 st1 wpe8 | 2.74 | 1.17x |
| 32 | BM16 BN128 BK256 w2 buf2 wg2 st1 wpe2 | 3.25 | BM32 BN32 BK256 w2 buf3 wg2 st2 wpe1 | 3.22 | 1.01x |
| 64 | BM64 BN64 BK256 w4 buf2 wg2 st1 wpe1 | 3.93 | BM32 BN128 BK256 w4 buf2 wg1 st1 wpe4 | 3.61 | 1.09x |
| 128 | BM64 BN128 BK256 w4 buf2 wg1 st2 wpe1 | 4.45 | BM64 BN128 BK256 w4 buf2 wg1 st2 wpe1 | 4.46 | 1.00x |
| 256 | BM128 BN128 BK256 w4 buf2 wg4 st2 wpe1 | 5.95 | BM64 BN128 BK128 w2 buf3 wg2 st2 wpe2 | 5.81 | 1.02x |
| 512 | BM256 BN128 BK128 w4 buf2 wg1 st2 wpe1 | 8.65 | BM128 BN128 BK128 w2 buf2 wg2 st2 wpe1 | 8.29 | 1.04x |
| 1024 | BM256 BN128 BK128 w4 buf2 wg1 st2 wpe1 | 19.87 | BM256 BN128 BK128 w4 buf2 wg2 st2 wpe2 | 18.51 | 1.07x |
| 2048 | BM256 BN128 BK128 w4 buf2 wg1 st2 wpe1 | 32.30 | BM128 BN128 BK64 w2 buf2 wg4 st1 wpe2 | 30.90 | 1.04x |
| 4096 | BM256 BN128 BK128 w4 buf3 wg1 st1 wpe1 | 63.50 | BM256 BN128 BK64 w2 buf3 wg4 st1 wpe1 | 60.39 | 1.05x |
| 8192 | BM256 BN128 BK128 w4 buf3 wg1 st1 wpe1 | 129.43 | BM256 BN128 BK64 w4 buf3 wg4 st1 wpe2 | 123.97 | 1.04x |

Geomean over the 13 buckets: 1.09x.

The decode buckets M <= 16 gain 1.17x to 1.25x. Their winner moves from a 16x64 tile with 4 warps
to a 16x16 tile with 1 warp at 4 WG/CU, which keeps 1024 tiles in flight. M=128 is unchanged
because the existing entry was already the best of 41k candidates.

## Method

- **Layout:** contiguous X `(B, M, K)` and Y `(B, M, N)`. ATOM's `_v_up_proj` passes a transposed
  `(M, B, K)` view and `transpose_bm=True`. The new configs pass numerics there but were not timed
  on that layout.
- **Search space:** the full `SEARCH_SPACE` for the family's `DEFAULT.json` keys, minus three kinds
  of pruning.
  - **Kernel limits:**
    - WMMA multiples (`BLOCK_SIZE_K % 32`, `BLOCK_SIZE_N % 16`).
    - The `STORE_MODE` 2 pad-interval fallback.
    - Duplicates from `NUM_WGS = min(tiles, CUs * WG_PER_CU)`.
    - LDS, including tile padding and the TDM store's C staging tile.
  - **Register pressure:** skip if the fp32 accumulator per lane plus 32 exceeds 1024 /
    `waves_per_eu`. Warps split N in two, then M, and gfx1250 has 1024 VGPRs per lane. Checked
    against 153 compiled kernels, the rule caught spills and never pruned a config that compiled
    without spilling. No spilling config came within 1.69x of the best.
  - **12-hour time box, M >= 512 only:** no `BLOCK_SIZE_N` 16, no `num_warps` 1, no
    `NUM_BUFFERS` 1 or 8. These were 1.2x to 3.2x off the best in calibration samples.
- **Volume:** 278k candidates in about 4h10m on 4 GPUs, at 0.2 to 0.5 s each. 107 were rejected
  at launch for LDS (see follow-up 2).

## Follow-ups

1. **Cold-start bias in the tuner's final round.** The final round times the installed baseline
   first, in a freshly started worker. At large M the first two or three timings run about 15%
   slow: M=2048 measured 37.2 us cold against about 32 us warm. So the installer's "vs installed"
   gains for M >= 2048 read 1.14x to 1.22x, while the steady state above is 1.04x to 1.05x.
   - The same bias can mis-rank the top 10. A warm re-time found other top-10 configs 2% to 7%
     faster than the installed ones for M >= 1024, partly within run-to-run noise.
   - Fix: warm the GPU before the first timed candidate, or time the baseline last, then rerun
     those final rounds.
2. **The LDS estimate omits the `STORE_MODE` 1 epilogue's layout-conversion scratch** (about
   16 KB). That caused the OutOfResources rejections at the 320 KB boundary.
3. **The large-M time box can be lifted.** With the Triton cache warm, large-M candidates cost
   0.1 to 0.7 s, so the untrimmed space (about 56k per bucket) is a few more hours.

## Reproduce

From `aiter/ops/triton/utils/_triton/tuning` on a tree with the kernel, the wrapper change and
the tuner registration:

    export PYTHONPATH=/path/to/aiter CU_NUM=256 GPU_ARCHS=gfx1250
    export TRITON_CACHE_DIR=/dev/shm/triton_cache_a16w8 TRITON_STORE_BINARY_ONLY=1
    python3 sweep_configs.py batched_gemm_a16w8 --B 128 --all-buckets --N 128 --K 512 \
        --gpu 0 1 2 3 --batch 500

`TRITON_STORE_BINARY_ONLY` keeps the Triton cache at about 100 KB per kernel instead of 450 KB.
At this candidate count, a full cache would fill a 100 GB disk.
