# `jagged_dense_bmm_bwd.py` — Backward Pass Explained

This file implements the **backward pass** of a fused operation called
`jagged_dense_bmm_broadcast_add` (jdbba).

## The operation being differentiated

The forward pass computes, for each group `b` over its packed row slice `[s, e)`:

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b]
```

A "jagged" tensor is a batch of variable-length sequences packed contiguously
into one `(L, K)` buffer, with `SEQ_OFFSETS` marking where each group
starts/ends. Each group `b` gets multiplied by its own dense weight matrix
`Dense[b]` `(K, N)` and bias.

The backward pass takes the upstream gradient `dOut` `(L, N)` and produces three
gradients:

```
dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)
dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
dBias[b]        = sum_m dOut[s:e, :]                (N,)
```

The key insight that drives the design: **which axis the contraction runs over**
determines the strategy.

## Three gradients, three strategies

### 1. `dJagged` — a clean per-row GEMM (`grad_jagged_kernel` / `grad_jagged`)

`dJagged[m,k] = sum_n dOut[m,n] · Dense[b][k,n]` contracts over the **static**
`N` axis (size 128). Because the contraction dimension is fixed at compile time
and each output row is independent, this is essentially the same GEMM shape as
the forward pass. So it reuses the forward kernel's machinery directly:

- MFMA (matrix-core) instructions via `tiled_mma` (16×16×16 bf16).
- A **double-buffered software pipeline** (`run_pipeline_stage`) that overlaps
  loading the next contraction tile from global memory with computing the
  current one, staging `A` (dOut) through LDS/shared memory with swizzling.
- bf16 in/out, fp32 accumulate, then truncate back to bf16 on store.

Grid is `(bm * KOUT_BLOCKS, 1, n_groups)` — one z-slice per group, tiled over
output rows and the K columns of the output.

### 2. `dBias` — split-reduction sum (`grad_bias_*`)

`dBias[b][n] = sum_m dOut[m,n]` is just a column-wise sum over the sequence.
Simple, but the reduction axis (`m`) is dynamic. To avoid serializing one long
reduction, it's done in **two passes** with a split factor `SPLIT = 4`:

- **Partials pass**: each of the `SPLIT` blocks sums a strided subset of rows
  (`r = off_s, off_s+SPLIT, ...`) into fp32 scratch. Thread `tid` owns column
  `n`.
- **Reduce pass**: sums the `SPLIT` fp32 partials per group → bf16 `dBias[b]`.

### 3. `dDense` — split-reduction outer-product GEMM (`grad_dense_*`)

`dDense[b][k,n] = sum_m Jagged[m,k] · dOut[m,n]` also contracts over the
**dynamic** `m` axis, so it uses the same two-pass split-reduction idea, but the
per-block work is a full `(K, N)` accumulation:

- A 16×16 thread grid (256 threads) owns the `(K, N)=(128,128)` output tile;
  each thread accumulates a `(KPT, NPT) = (8, 8)` register sub-block.
- Rows are staged `DDENSE_BM = 64` at a time through LDS for both `Jagged`
  (`sJ`) and `dOut` (`sD`), then each thread does outer-product FMAs over those
  rows.
- The reduce pass sums the `SPLIT` fp32 `(K, N)` partials into bf16 `dDense[b]`.

## Cross-cutting details

- **Group resolution**: every kernel reads `SEQ_OFFSETS[b]` and
  `SEQ_OFFSETS[b+1]` to find its group's `[seq_start, seq_end)`, then
  `readfirstlane` scalarizes those so group-derived addresses stay uniform
  across the wave.
- **Jagged tail handling via hardware bounds**: rather than branchy masking,
  buffers are bounded to exactly `M_b` rows. On CDNA, out-of-bounds loads return
  0 and out-of-bounds stores are dropped — so tail rows of a tile that overrun a
  short group contribute 0 to contractions and don't corrupt the neighboring
  group.
- **Shared constants with the forward kernel**: it imports `BLOCK_M/N/K`, `K`,
  `N`, etc. from the sibling `jagged_dense_bmm` module so forward and backward
  stay in lockstep. `N == K == 128`.
- Targets AMD CDNA GPUs (gfx942 / gfx950), bf16 data with fp32 accumulation
  throughout.

## Summary

It's a hand-written FlyDSL backward kernel that splits the three gradients by
their contraction axis — `dJagged` as a forward-style MFMA GEMM over the static
N, and `dDense`/`dBias` as two-pass fp32 split-reductions over the dynamic
sequence axis — all while handling variable-length jagged groups using CDNA's
buffer-bounds semantics.
