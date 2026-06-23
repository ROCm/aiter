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

## Why one kernel for `dJagged` but two for `dDense` / `dBias`

The asymmetry comes down to **what axis each gradient reduces over**, and whether
that reduction can be owned by a single block without serializing.

### `dJagged`: no reduction over the jagged axis → one kernel

`dJagged[m,k] = sum_n dOut[m,n] · Dense[b][k,n]` sums over `n`, the **static** `N`
axis (= 128, known at compile time). Each output element depends only on **one
row** `m` of `dOut`, so different output rows are completely independent. The work
parallelizes cleanly across the `(m, k)` output grid, and each tile reduces over a
small, fixed `N` entirely **inside** its own MFMA accumulator. There's no
cross-block reduction to coordinate, so a single GEMM-style kernel writes the
final answer directly. The output `(L, K)` is the same size as the jagged input,
and every block owns a disjoint slice.

### `dDense` / `dBias`: reduction over the dynamic `m` axis → two kernels

Both contract over `m`, the **dynamic** (variable-length, per-group) sequence
axis:

```
dDense[b][k,n] = sum_m Jagged[m,k] · dOut[m,n]
dBias[b][n]    = sum_m dOut[m,n]
```

The defining difference: **every row `m` in the group contributes to the same
output element.** The output `dDense[b]` is only `(K, N)` per group and `dBias[b]`
only `(N,)` — tiny and fixed-size, independent of how many rows `M_b` the group
has (which can be large and varies per group at runtime). That creates a reduction
problem with two options:

1. **One block reduces the whole group serially.** Works, but throws away
   parallelism: with few groups and long sequences, most of the GPU sits idle
   while a handful of blocks grind through long serial reductions on the critical
   path.
2. **Split the reduction across blocks, then combine** (what the code does, with
   `SPLIT = 4`). The partials kernel launches `SPLIT` blocks per group that each
   sum a strided subset of rows into their own fp32 scratch slot in parallel; the
   reduce kernel then sums those partials into the final bf16 output.

The second kernel is needed precisely *because* the first produces multiple
partial results that must be combined — and they can't be combined within the
partials kernel, since they're computed by **different blocks** with no cheap
global synchronization (GPU blocks can't barrier with each other mid-kernel). A
separate kernel launch is the synchronization point.

Two reasons make the split worth it over the serial option:

- **Parallelism / occupancy.** The output is tiny and the reduction axis is huge
  and dynamic; splitting gives more independent blocks to fill the machine.
- **fp32 accuracy.** Partials accumulate and are stored in fp32 scratch; only the
  final reduce truncates to bf16. This keeps the long `m`-reduction numerically
  stable regardless of sequence length.

| Gradient | Contraction axis | Reduction across blocks? | Kernels |
|----------|------------------|--------------------------|---------|
| `dJagged` | static `N` | No — each output row independent, reduces inside one tile | 1 (GEMM) |
| `dDense`  | dynamic `m` | Yes — all rows hit the same `(K,N)` output | 2 (partials + reduce) |
| `dBias`   | dynamic `m` | Yes — all rows hit the same `(N,)` output | 2 (partials + reduce) |

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
