# preshuffle_gemm_blockscaled — reproduce & tune

FP8 blockscaled GEMM `C(M,N) = A(M,K) @ B(N,K).T` for gfx950 / MI355X.
Kernel: `aiter/ops/flydsl/kernels/preshuffle_gemm_blockscaled.py`.

This doc covers: (1) environment, (2) how to reproduce the recorded numbers,
(3) how to run a tile/swizzle sweep to tune a new shape, (4) the patterns that
make tuning fast, and (5) how to record a winner.

---

## 1. Environment

- **GPU**: MI355X (gfx950). Pin to one device (sweeps assume GPU 6 below):
  `CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6`
- **Python**: the project venv. Activate with `source /opt/venv/bin/activate`,
  or call `/opt/venv/bin/python` directly with
  `PYTHONPATH=$PWD:$PYTHONPATH` from the repo root.
- The `setlocale: LC_ALL` warning is harmless.

Sanity import:
```
source /opt/venv/bin/activate
PYTHONPATH=$PWD python -c \
 "from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import compile_preshuffle_gemm_blockscaled; print('ok')"
```

---

## 2. Scale formats (pick the right ABI)

| scale_format     | A/B scale ABI                              | when to use                         |
|------------------|--------------------------------------------|-------------------------------------|
| `ue8m0` (default)| sa i32 (M,K/512), sb i32 (N/128,K/512)     | scales pre-packed UE8M0; FASTEST    |
| `fp32`           | sa fp32 (M,K/128), sb fp32 (N/128,K/128)   | fp32 scales already pow-2 (lossy)   |
| `fp32_post_mfma` | sa fp32 (M,K/128), sb fp32 (N/128,K/128)   | arbitrary fp32 scales; EXACT        |

`fp32_post_mfma` is the true-FP32 blockscale path (uses the handcraft compute:
A double-buffer + 1-deep MFMA→FMA interleave). The recorded sweep numbers below
are for this path.

ABI (all formats): `launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream)`
- arg_c bf16 (M,N); arg_a fp8 e4m3 (M,K); arg_b fp8 preshuffled (N/16,K/64,4,16,16).

---

## 3. Reproduce the recorded performance

A single-config bench/correctness runner lives at
`preshuffle_gemm_blockscaled_perf/bench_and_tune.py`. It prints
`OK <TFLOPS> <us> nf=<#elems off vs fp32-ref>`.

```
CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
 /opt/venv/bin/python preshuffle_gemm_blockscaled_perf/bench_and_tune.py \
   --M 32768 --N 12288 --K 2048 --tm 128 --tn 128 --tk 128 --bsw 2 \
   --scale fp32_post_mfma
# -> OK 1703.5 968.1 nf=0
```

Recorded winners (fp32_post_mfma, MI355X, handcraft compute):

| M×N×K              | tile (tm×tn×tk) | bsw | TFLOPS |
|--------------------|-----------------|-----|--------|
| 8192×8192×8192     | 128×128×128     | 8   | ~2014  |
| 32768×2048×4096    | 64×256×128      | 4   | ~1939  |
| 16384×2048×4096    | 64×256×128      | 4   | ~1920  |
| 16384×12288×2048   | 128×128×128     | 8   | ~1799  |
| 32768×12288×2048   | 128×128×128     | 2   | ~1695  |

Notes:
- `nf` (#elements > 0.15 abs-err vs an fp32 reference) is usually 0; a tiny
  nonzero (e.g. 1) is a ±1 bf16-ULP rounding at large magnitudes, NOT a bug
  (it is bit-identical to the older batched compute — verified).
- Timing is min over 6×120 iters with a 1500-iter warmup. A long warmup matters:
  short benches let the GPU clock idle between launches and under-report TFLOPS.
  If a number looks ~20× low, the clock didn't ramp — raise warmup / iters.
- Bit-exactness is enforced by `op_tests/test_preshuffle_gemm_blockscaled.py`.

---

## 4. Tune a new shape (tile + block-swizzle sweep)

The sweep tries tile_m∈{64,128,256} × tile_n∈{128,256} × tile_k∈{128,256} ×
block_swizzle_n∈{0,1,2,4,8} (bsw pruned to divisors of N/tile_n). It shells out
**one subprocess per config** — required, because the flydsl AST rewriter can
corrupt state across many in-process compiles.

```
CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
 /opt/venv/bin/python preshuffle_gemm_blockscaled_perf/bench_and_tune.py \
   --M 32768 --N 12288 --K 2048 --scale fp32_post_mfma --sweep
```
Prints every config + a `WINNER:` line at the end. A full sweep is ~50-60
configs (a few minutes to ~10 min depending on K depth).

To run several shapes back-to-back without GPU contention, chain them in one
shell so they run sequentially (don't launch concurrent sweeps on the same GPU):
```
for shp in "32768 12288 2048" "32768 2048 4096"; do
  set -- $shp
  CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 PYTHONPATH=$PWD \
   /opt/venv/bin/python preshuffle_gemm_blockscaled_perf/bench_and_tune.py \
     --M $1 --N $2 --K $3 --scale fp32_post_mfma --sweep
done
```

---

## 5. What drives the tile choice (tune faster)

Observed across the M/N/K sweeps — use these to prune the grid:

- **N drives the tile** (independent of M):
  - wide N (e.g. 12288) → `tile 128×128`
  - narrow N (e.g. 2048) → `tile 64×256` (bigger N-tile = more A reuse)
- **M scales time, not the winner** — the same (N,K) picks the same tile at
  M=16384 and M=32768. Tune at one M and reuse for other M of that (N,K).
- **tile_k = 128 essentially always wins.** tile_k=256 is slower and, with big
  tiles + deep K, overflows the 160 KB LDS budget.
- **tile_m = 256 (and tile_n=256 combined) spill/overflow** — 128 VGPR of
  accumulators is the wall; these land at the bottom of every sweep. Skip them
  unless N is very narrow.
- **block_swizzle_n is shape-specific** (~+13% on wide-N): try {2,4,8}. It must
  divide `N/tile_n`. It tunes XCD/L2 locality, so the optimum shifts with the
  grid shape.

Quick recipe for a brand-new shape: try `128×128×128` (bsw 2/4/8) first; if N is
narrow (≤2048) also try `64×256×128` (bsw 2/4). The winner is almost always one
of those.

---

## 6. Record a winner

Two tables in the kernel file:
- `_AUTOTUNE_WINNERS` — ue8m0 (fused-scale) path; consumed by
  `compile_preshuffle_gemm_blockscaled_auto`.
- `_FP32_POST_MFMA_WINNERS` — fp32_post_mfma path (documentation of swept
  winners; the auto dispatcher is scale-format-agnostic, so apply these by
  passing the tile explicitly).

Add a line `(M, N, K): (tm, tn, tk, bsw),  # <TFLOPS>` to the appropriate table.

Full per-shape sorted sweep tables and analysis:
`blockscale_per_analysis/autotune_sweep_handcraft.md` and
`blockscale_per_analysis/handcraft_kernel_performance.md`.
