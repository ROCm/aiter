# Jagged-Dense BMM — Backward Pass Implementation Plan

Date: 2026-06-18
Status: Plan only (no implementation yet)

## Hard Constraints (binding on all implementation work)

- **Do NOT reference the development phases of this document in source code.**
  The phase numbers/names here exist only to track our internal development
  progress; they are meaningless to other developers (the intended audience of
  the code). Source files, comments, docstrings, identifiers, commit messages,
  and test names must NOT mention "Phase 0/1/2/3/4" or this plan's phase
  structure. Code comments should explain the kernel/math itself, not where we
  are in this plan.

## 0. Context & References

The forward kernel lives in
`aiter/aiter/ops/flydsl/kernels/jagged_dense_bmm.py` and its standalone
validation/benchmark harness in
`aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm.py`.

Forward math, per group `b` over its packed row slice `[s, e)` (`M_b = e - s`):

```
Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
  (M_b x N)     (M_b x K)      (K x N)     (1 x N broadcast)
```

Tensor conventions used by the forward kernel (we will reuse them):

- `A` = Jagged, packed `(L, K)` bf16, `L = sum_b M_b`.
- `B` = Dense, **host pre-transposed** to a tall `(n_groups * N, K)` bf16 matrix.
  Per group, `B_g` is `(N, K)` = `Dense[b].T`. The MFMA computes
  `Out[m,n] = sum_k A[m,k] * B_g[n,k]`.
- `BIAS` = flat `(n_groups * N,)` bf16.
- `SEQ_OFFSETS` = `(n_groups + 1,)` int32 prefix sums (on device).
- `C` = Out, `(L, N)` bf16 (host pads `L + BLOCK_M` rows for tail-tile safety).
- Tiling: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64`, `N=128, K=128`, fp32 accumulate.
- Grid: `(bm * N_BLOCKS, 1, n_groups)`, block `(256,1,1)`, one group per `block_idx.z`.

## 1. Backward Math (what we must compute)

Given the upstream gradient `dOut` (a.k.a. `gC`) of shape `(L, N)` bf16, we
produce three gradients, per group `b`:

```
dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)   = dOut @ B_g
dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
dBias[b]        = sum over rows m of dOut[s:e, :]   (N,)
```

Index form (helps pick MFMA operand orientation):

- `dJagged[m,k] = sum_n dOut[m,n] * Dense[b][k,n]`
  → reduction over `N`. Output is **per-row independent** → embarrassingly
  parallel over `(M-tile, K-tile)`, structurally identical to the forward GEMM.
  Note the reduction operand here is `Dense[b]` in its **original** `(K, N)`
  orientation (i.e. `B_g.T`), not the tall pre-transposed `B`.
- `dDense[b][k,n] = sum_m Jagged[m,k] * dOut[m,n]`
  → reduction over `m`, the **jagged/sequence axis**, which spans the whole
  group and may be split across many M-blocks. This is the hard one: it needs
  cross-block accumulation (atomics or split-reduction).
- `dBias[b][n] = sum_m dOut[m,n]`
  → also a reduction over the jagged axis. Cheap, but same cross-block concern.

### Key design consequence

The three outputs have **two different parallelization structures**:

1. `dJagged` — GEMM with the contraction over the static `N` axis. Clone of the
   forward kernel.
2. `dDense` + `dBias` — contraction over the dynamic `M_b` (sequence) axis,
   requiring accumulation across all row-tiles of a group.

I propose **separate kernels per output**, which keeps each kernel close to a
pattern that already works, is the most TDD-friendly (validate one thing at a
time), and matches the forward file's "clean prototype first" philosophy. A
fused kernel can be a later optimization, noted in §8.

Per A3, `dDense` and `dBias` are each computed as a **two-pass split-reduction**
(partials kernel + reduce kernel), and `dBias` partials are fused into the
`dDense` partials pass since both reduce over the same `m` axis.

## 2. Resolved Decisions (was: Open Questions / Assumptions)

All five items below were confirmed by the user on 2026-06-18.

- **A1. dtype of grad outputs — CONFIRMED.** `dJagged`, `dDense`, `dBias` are
  bf16 in/out with fp32 accumulate (mirrors forward). Cross-block accumulation
  for `dDense`/`dBias` is done in fp32 (see A3).
- **A2. Output orientation of `dDense` — START WITH (a).** Produce `dDense[b]` as
  `(K, N)` (natural math layout). Measure performance once implemented; only
  revisit the tall `(n_groups * N, K)` forward-compatible layout (b) if needed.
- **A3. Cross-block reduction strategy — USE (iii) SPLIT-REDUCTION.** Atomics
  (i) serialize and are "terribly inefficient"; the single-block-per-group loop
  (ii) is too slow for long sequences. We want the most performant solution, so
  `dDense` and `dBias` use a **two-pass split-reduction**:
  - **Pass 1 (partials):** a split-K-style grid with a split factor `S` over the
    `m` axis. Each block accumulates a slice of the group's M-tiles in fp32
    registers and writes one fp32 partial to a scratch buffer.
  - **Pass 2 (reduce):** a small kernel sums the `S` partials per group into the
    final `dDense[b]` `(K,N)` and `dBias[b]` `(N,)`, casting fp32 → bf16.
  - `S` is a tunable (start with e.g. `S=4`); see §3 Phase 3 for the grid/scratch
    sizing. No atomics; each scratch slot is written by exactly one block.
- **A4. Empty / partial groups — REUSE.** Reuse the forward kernel's runtime
  early-exit (`if start_m < M_b`) and bounded buffer-descriptor masking so empty
  groups and partial tail tiles are handled without OOB. Masked stores skip rows
  `>= M_b`; masked loads in the reduction zero-fill tail rows (see A5).
- **A5. Do bounded buffer-descriptor loads zero-fill OOB? — VERIFIED TRUE.**
  Empirically confirmed on the target arch **gfx950 (CDNA4)** with a standalone
  probe (16-element input, descriptor bounded to 8 elements, one load per
  thread): in-bounds lanes returned the real value, OOB lanes returned **exactly
  `0.0`**. So bounding the `Jagged`/`dOut` group views by `M_b` rows makes tail
  rows (`m >= M_b`) contribute 0 to the `(K,N)` contraction with no explicit
  masking arithmetic. (Caveat for future RDNA ports: `_get_buffer_flags` sets
  `OOB_SELECT=2` "no bounds checking" on RDNA, so this zero-fill guarantee is
  **CDNA-only**; the backward kernels target CDNA like the forward kernel.)

## 3. TDD Plan — Order of Work

Build the example/validation harness **first**, with CPU references and kernel
call stubs, then implement kernels one at a time until each validation passes.

### Phase 0 — Reference & harness scaffolding (no kernel yet)

File: new `example_jagged_dense_bmm_bwd.py` next to the forward example.

**Status: COMPLETE (2026-06-18).**

1. [x] Reuse `make_seq_offsets` / `make_inputs` from the forward example (import
   or copy the sibling helpers). Extend `make_inputs` to also create a random
   `dOut` `(L, N)` bf16 (the upstream gradient). *(Imports `make_seq_offsets`
   from the sibling example; local `make_inputs` adds `dOut`.)*
2. [x] Write **pure-torch references** for all three gradients, looping per group:
   - [x] `ref_dJagged[s:e] = dOut[s:e].float() @ Dense[b].float()` (`Dense[b]` is
     `(K,N)`; result `(M_b,K)`). *(`ref_grad_jagged`, uses `dense[b].t()`.)*
   - [x] `ref_dDense[b] = Jagged[s:e].float().T @ dOut[s:e].float()` (`(K,N)`).
     *(`ref_grad_dense`.)*
   - [x] `ref_dBias[b] = dOut[s:e].float().sum(dim=0)` (`(N,)`).
     *(`ref_grad_bias`.)*
   - [x] Validate this reference against `torch.autograd` on the forward
     (build forward with `requires_grad`, backprop a known `dOut`) so the
     hand-written references are themselves trusted. *(`autograd_grads`; harness
     asserts match at cosine > 0.99999 before any kernel runs.)*
3. [x] Add per-output validators (cosine > 0.999 + max-abs-err). Each gradient
   gets its own PASS/FAIL line. *(`cosine_maxabs` / `report`.)*
4. [x] Add a `run_flydsl_bwd(...)` that prepares device buffers and **calls
   not-yet-implemented launchers** (stubs raising `NotImplementedError`),
   defining the host-side contract (shapes, transposes, scratch buffers).
   *(Catches `NotImplementedError` per gradient and prints `[skip]`.)*
5. [x] Confirm the references pass against autograd with kernels stubbed out.
   *(Verified in both `uniform` and `skew` regimes; all three references
   cosine 1.000000.)*

### Phase 1 — `dJagged` kernel (the easy GEMM clone)

This is the forward kernel with operands re-pointed; do it first to bank a win.

**Status: COMPLETE (2026-06-18).** Implemented in `jagged_dense_bmm_bwd.py` as
`grad_jagged_kernel` + `grad_jagged`. Because `N == K == 128` the tiling numerics
match the forward exactly, so the full double-buffered pipeline was reused.

1. [x] New `@flyc.kernel grad_jagged_kernel` derived from `jdbba_kernel`:
   - [x] Output tile is `(BLOCK_M, BLOCK_K)` over `(M, K)` instead of `(M, N)`.
     *(Output column axis K tiled by `BLOCK_N`; see `KOUT_BLOCKS`.)*
   - [x] LHS = `dOut` group view `(M_b, N)`; contraction over `N`.
   - [x] RHS = `Dense[b]` in `(K, N)` orientation. **Host decision:** pass the
     original (non-transposed) dense reshaped to `(n_groups*K, N)`. *(Done in
     `run_flydsl_bwd` via `dense.reshape(n_groups*K, N)`.)*
   - [x] No bias term (bias load/add epilogue dropped).
   - [x] Grid tiling `(bm * KOUT_BLOCKS, 1, n_groups)`.
   - [x] Reuse per-group rebasing, `readfirstlane` scalarization, bounded `C`
     descriptor (`M_b * K * 2`), runtime early-exit, masked store.
2. [x] Add `@flyc.jit grad_jagged` launcher mirroring `jagged_dense_bmm`'s
   tiled-mma / tiled-copy setup (contraction dim is `N`).
3. [x] Wire into `run_flydsl_bwd`; `dJagged` validation passes (cosine 0.999999
   in uniform + skew regimes).

### Phase 2 — `dBias` split-reduction (de-risk the two-pass infra)

Do this before `dDense` because it's the simplest reduction over `m` and
de-risks the split-reduction scratch + reduce-kernel infrastructure (A3-iii)
before tackling the harder transposed GEMM. `dBias[b][n] = sum_m dOut[m,n]`.

**Status: COMPLETE (2026-06-18).** Implemented in `jagged_dense_bmm_bwd.py` as
`grad_bias_partials_kernel` + `grad_bias_reduce_kernel` + `grad_bias`. Validates
at cosine 0.999999 in uniform + skew regimes.

**Implementation note (deviation from the original sketch):** rather than tiling
rows into `(BLOCK_M, BLOCK_N)` tiles, the partials pass uses **row-level
striding** with one thread per output column `n` (block `= (N,1,1)`). Split `s`
sums local rows `r = s, s+SPLIT, ...` via a dynamic loop-carried `range`
(`scf.for`), so the loop bound `M_b` keeps every read in-range (no OOB / no
masking needed) and each per-row read is coalesced across the `N` column-threads.
This is simpler than a tiled load + intra-tile reduction and avoids any
cross-wave reduction.

1. [x] **Scratch buffer.** fp32 `bias_partials`, viewed as `(n_groups * SPLIT, N)`.
   Each `(group b, split s)` block owns exactly row `b*SPLIT + s` (no atomics).
2. [x] **Pass 1 — `grad_bias_partials_kernel`.** Grid `(1, SPLIT, n_groups)`,
   block `(N,1,1)`. Thread `n` accumulates split `s`'s strided rows in fp32 and
   writes the `(N,)` partial. *(Row-striding variant; see note above.)*
3. [x] **Pass 2 — `grad_bias_reduce_kernel`.** Grid `(n_groups,1,1)`, block
   `(N,1,1)`: sums the `SPLIT` fp32 partials per group and writes bf16 `dBias[b]`.
4. [x] Row-reduction mechanism decided: one thread per column + row-strided
   accumulation (no LDS / no warp reduction needed). Kept simple.
5. [x] Validate `dBias` against the reference. *(cosine 0.999999, uniform +
   skew; empty groups and splits with no rows correctly yield 0.)*

### Phase 3 — `dDense` split-reduction (the hard transposed GEMM)

`dDense[b][k,n] = sum_m Jagged[m,k] * dOut[m,n]` — contraction over the dynamic
`m` axis, fused with the `dBias` reduction structure (same split over `m`).

**Status: COMPLETE (2026-06-18).** Implemented in `jagged_dense_bmm_bwd.py` as
`grad_dense_partials_kernel` + `grad_dense_reduce_kernel` + `grad_dense`.
Validates at cosine 0.999999 in uniform + skew regimes.

**Implementation note (deviation from the MFMA sketch):** to get a correct
baseline without the risky MFMA double-transpose, the partials pass uses an
**LDS-tiled register reduction** rather than MFMA. A 16x16 thread grid owns the
`(K, N)` output tile; each thread accumulates a `(KPT, NPT) = (8, 8)` fp32
register sub-block. For each `DDENSE_BM = 64`-row m-tile assigned to the split,
the block cooperatively stages `Jagged` and `dOut` rows into LDS, then every
thread does the `sum_m J[m,k]*dOut[m,n]` FMAs from LDS. No MMA, no operand
transpose. **MFMA acceleration is deferred to §8** (the performance follow-up).

1. [x] **Scratch buffer.** fp32 `dense_partials`, viewed as
   `(n_groups * SPLIT * K, N)`. Each `(group b, split s)` owns the `K`-row block
   at `(b*SPLIT + s)*K` (no atomics). Sizing `n_groups=64, S=4, K=N=128, fp32`
   → 16 MiB.
2. [x] **Pass 1 — `grad_dense_partials_kernel`.** Grid `(1, SPLIT, n_groups)`,
   block `(256,1,1)`. Dynamic strided m-tile loop (`m_tile = s, s+SPLIT, ...`,
   count `ceil(M_b / DDENSE_BM)`); fp32 register accumulator persisted across the
   `scf.for` via an rmem fragment.
   - [x] Output `(K, N)` accumulated in fp32 register sub-blocks (no `mma_frag`).
   - [x] Dynamic m-tile loop over the split's assigned tiles.
   - [x] Contraction `C[k,n] += sum_m J[m,k]*dOut[m,n]` — done via **LDS-staged
     FMA** instead of MFMA (deviation; transpose risk avoided).
   - [x] Tail rows `m >= M_b` zero-fill via `M_b`-bounded buffer descriptors
     (A5), no explicit masking.
   - [x] Write the fp32 `(K, N)` partial to the split's scratch block.
3. [x] **Pass 2 — `grad_dense_reduce_kernel`.** Grid `(K, n_groups, 1)`, block
   `(N,1,1)`: sums the `SPLIT` fp32 partials per group/k-row and writes bf16
   `dDense[b]`. Shares the `_load_scalar`/`_store_scalar` reduce structure with
   the bias reduce.
4. [x] Validate `dDense` (uniform then skew). *(cosine 0.999999 both; empty
   groups and splits with no rows yield 0.)*

### Phase 4 — Integration & polish

1. Single `main()` in the bwd example runs all three validations + reports
   timing/TFLOPs per gradient (FLOPs: `dJagged` and `dDense` each ~`2*L*K*N`;
   `dBias` ~`L*N` adds).
2. Confirm `uniform` and `skew` regimes both pass (skew exercises empty/partial
   groups → validates masking and early-exit).
3. Run the style gate: `bash scripts/check_python_style.sh --fix --include-local`.

## 4. Files to Create / Touch

- **New** `aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm_bwd.py`
  — references, validators, host wiring, benchmark (Phase 0 first).
- **New** `aiter/aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`
  — `grad_jagged_kernel`/`grad_jagged`; the two-pass split-reduction kernels
  `grad_bias_partials_kernel` + `grad_bias_reduce_kernel`/`grad_bias` and
  `grad_dense_partials_kernel` + `grad_dense_reduce_kernel`/`grad_dense`
  (+ shared helpers; reuse `make_bounded_buffer_tensor` from the forward module
  via import). Each `grad_*` launcher allocates its own fp32 scratch partials
  buffer (or accepts a caller-provided one) sized by the split factor `S`.
- Do **not** modify the forward kernel files except to export reusable helpers
  if cleaner than duplication (decide during Phase 0; prefer import over copy).

## 5. Validation Strategy (success criteria)

For each gradient, against the torch reference (itself checked vs autograd):

- cosine similarity > 0.999 over the flattened tensor, and
- bounded max-abs-error (bf16 tolerance, e.g. matching the forward example).
- Both `uniform` and `skew` regimes pass.
- Empty groups (`M_b = 0`) produce exactly-zero `dDense[b]`/`dBias[b]` and write
  nothing to `dJagged` for those rows.

## 6. Numerical / Correctness Risks

- **Tail-row masking in reductions** (A5): RESOLVED — verified that the
  `M_b`-bounded buffer descriptor zero-fills OOB loads on gfx950 (CDNA4), so
  `m >= M_b` rows contribute 0 to the contraction automatically. Re-check if the
  target arch ever changes (RDNA does NOT zero-fill; see A5 caveat).
- **Operand transpose for `dDense`**: getting `(K,N)` out of `(M,K)` and `(M,N)`
  inputs likely needs an LDS-staged transpose; this is the most error-prone
  step. Validate with a tiny shape (1 group, `M_b < BLOCK_M`, `K=N=128`) first.
- **Partials accumulation dtype**: scratch `dDense_partials`/`dBias_partials`
  MUST be fp32; bf16 partials would lose precision and likely fail the cosine
  bar. Only the Pass-2 reduce output is cast to bf16.
- **Scratch lifetime / init**: each split slot is written by exactly one Pass-1
  block, so no pre-zeroing of scratch is required for *written* slots; but slots
  for empty groups / unused splits must either be skipped by Pass 2 or
  zero-initialized so they don't pollute the sum. Decide explicitly (prefer
  zero-init scratch for safety in the first version).
- **Split factor `S` vs `ceil(M_b/BLOCK_M)`**: when a group has fewer M-tiles
  than `S`, some splits do zero work — their partials must be 0 (covered by
  zero-init + early-exit).

## 7. Build / Run Commands

Use the existing `flydsl_venv` for all testing. **Do not modify the environment**
(no installs/upgrades); if something is missing or broken in the venv, stop and
ask for guidance rather than changing it. The venv + a gfx950 (CDNA4) GPU were
confirmed working during A5 verification.

```bash
source flydsl_venv/bin/activate
python aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm_bwd.py
# subset while iterating, e.g. add a --only {djagged,dbias,ddense} flag
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm_bwd.py
```

Disable the JIT cache (`FLYDSL_RUNTIME_ENABLE_CACHE=0`) while iterating on kernel
source to avoid stale artifacts.

## 8. Future Optimizations (out of scope for first correct version)

- **Accelerate `dDense` with MFMA.** The current `dDense` partials pass is an
  LDS-tiled register-FMA reduction (correctness baseline). Replace the per-thread
  FMA inner loop with an MFMA-based transposed GEMM (`C[k,n] = sum_m J[m,k]
  dOut[m,n]`), feeding MFMA fragments from the LDS-staged tiles (CDNA4 supports
  LDS-read-transpose). This is the highest-value perf follow-up.
- Fully fuse the `dBias` partials into the `dDense` partials kernel (both reduce
  over `m` with the same split structure) to share `dOut` loads.
- Autotune the split factor `S` (and a possible 2-level reduction tree) against
  the sequence-length distribution; revisit A2 (tall `dDense` layout) if a
  forward-compatible output avoids a host transpose on the hot path.
- Port `dJagged` onto the double-buffered software pipeline already in the
  forward kernel for throughput.
- Autotune `BLOCK_M/N/K` per gradient.

## 9. Step-by-Step Checklist (execution order)

1. [x] Phase 0: bwd example with torch references validated against autograd;
   `run_flydsl_bwd` stubs in place. (Done 2026-06-18: references match autograd
   in uniform + skew regimes; FlyDSL launchers stubbed.)
2. [x] Phase 1: `grad_jagged` kernel + launcher; `dJagged` validation passes.
   (Done 2026-06-18: forward-pipeline clone with operands re-pointed and bias
   epilogue dropped; cosine 0.999999 vs reference in uniform + skew regimes.)
3. [x] Phase 2: `grad_bias` split-reduction (partials + reduce kernels + fp32
   scratch); `dBias` passes. (Done 2026-06-18: row-strided per-column
   accumulation; cosine 0.999999 in uniform + skew.)
4. [x] Phase 3: `grad_dense` split-reduction (partials + reduce kernels + fp32
   scratch); `dDense` passes (uniform then skew). (Done 2026-06-18: LDS-tiled
   register reduction baseline, no MFMA; cosine 0.999999 in both regimes. MFMA
   acceleration deferred to §8.)
5. [ ] Phase 4: unified `main()`, timing, both regimes green, style gate clean.
