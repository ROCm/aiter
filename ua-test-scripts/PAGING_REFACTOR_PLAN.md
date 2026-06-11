# UA KV Paging — Simplification + Contiguous-Evaluation Plan

Status: **planning** (no code changes yet). Owner discussion: keep existing
scalar-promote tiers until the SRD rebase is *proven* fast **and** the tiers
are *proven* redundant by measurement.

---

## 0. Guiding principles

1. **Addressing is not the bottleneck.** The canonical-shape ATT profile
   (`addr_baseline_b16_sq10000`) is dominated by `memwait` (27%) + `softmax`
   (22%); the `addr` phase is ~17.8% busy and largely latency-hidden behind
   `memwait`. Wider FA literature treats KV address math as negligible. So the
   goal is **fewer, clearer paths**, not more micro-optimized tiers.
2. **No "maybe-helps" code.** Every distinct addressing path must justify its
   existence with a *measured* win above a fixed threshold on a *named* shape.
   If it can't, it gets deleted.
3. **One mental model.** A reader should be able to hold the addressing in
   their head: "single page → rebase the base; otherwise → per-lane scatter."

### Decision rule (used throughout)

> A code path stays **iff** removing it regresses its target shape by
> **≥ 2%** wall-clock (median of `@perftest`), measured A/B with everything
> else fixed. Below 2% → delete it.

(2% is a starting threshold; adjust once we see the spread. The point is the
rule is *fixed in advance* and data-driven, not vibes.)

---

## 1. Where we are today (the complexity we want to remove)

`unified_attention_pipeline.hpp :: refresh_k_offsets` (lines ~983–1086) and the
symmetric `refresh_v_offsets` currently have **four** branches:

| # | gate | what it does | regime |
|---|------|--------------|--------|
| 1 | `kRebaseKSrd` (new) | wave-uniform SRD base + constant per-lane offset | single-page |
| 2 | `kScalarPromoteKPageIdx && kDedupPages` | `grp` dedup, per-issue page offsets | single- & multi-page |
| 3 | `kScalarPromoteKPageIdx` (no dedup) | per-issue scalar-promote | exotic page sizes |
| 4 | `else` | byte-identical per-lane scatter | everything / fallback |

The rebase (1) was **added on top** of (2)-(4), so right now the file is
*more* complex, not less. That is the thing this plan fixes.

### The three irreducible regimes

| regime | compile-time predicate | example shapes | addressing |
|--------|------------------------|----------------|------------|
| **single-page** (tile ⊆ one physical page) | `kHasCePageSize && kPageSize % kPageBlockSize == 0` | d128 ps32/64/128, d64-fp8 ps64/128, d64-bf16 ps32/64 | **rebase** SRD base (uniform) |
| **multi-page** (tile spans pages) | `kHasCePageSize && kPageBlockSize % kPageSize == 0 && kPageSize < kPageBlockSize` | d128 ps16, **d64-fp8 ps32**, d64-bf16 ps16 | **per-lane scatter** (pages non-contiguous) |
| **runtime page size** | `!kHasCePageSize` | no host dispatch on page_size | per-lane scatter fallback |

`kPageBlockSize` (= `BlockTile[2]`) is **32** for d128 (fp8/bf16) and d64-bf16,
**64** for d64-fp8.

### Why we can't "just copy the legacy method" everywhere

- The legacy `PageBlockNavigator` *is* a uniform-SRD-base rebase — which is
  exactly what branch (1) now does. We did copy it; we just scoped it to where
  it is valid.
- It is valid **only for single-page tiles.** For multi-page tiles the tile's
  tokens live in several *logical* pages, and `block_tables[]` maps those to
  physical blocks that can be **anywhere in the pool (non-contiguous)**. No
  single wave-uniform base can address them → per-lane scatter is **required**.
  This is the real blocker.
- **It is not the int32 overflow.** Overflow is why the per-lane path has a
  `_long` (per-lane 64-bit base) variant for >4 GB pools. The rebase actually
  *removes* overflow for single-page: the 48-bit SRD base absorbs the large
  part and the per-lane voffset stays within one tile (always int32-safe). So
  the rebase path never needs `_long`.

---

## 2. Target structure (the simplification)

Collapse to **two intents**:

```text
refresh_k_offsets(tile_idx):
    if constexpr (kRebaseSrd):           // single page
        k_srd_base_offset = uniform_base(tile_idx)   // 1 readfirstlane, wave-uniform
        k_page_offsets    = const_lane_offsets()     // loop-invariant
    else:                                 // multi-page OR runtime page size
        per-lane k_page_offsets(i) = scatter_offset(tile_idx, i)
```

Concrete edits (after the rebase is proven — see §4 gates):

1. **Broaden the rebase gate** to `kRebaseSrd = kHasCePageSize && (kPageSize %
   kPageBlockSize == 0)` (drop the `kScalarPromote` dependency). Single-page
   then *always* takes the simple rebase, including `KNRepeat==1` shapes
   (e.g. bf16-d128 prefill) that currently fall into branch (4).
2. **Delete the single-page sub-cases** of branches (2)/(3)/(4): the `grp`
   logic only ever needs the multi-page form afterwards.
3. **Unify the structural plumbing**: drop the special-case "advance the
   `k_view` base for tile 0" and route tile 0 through the same
   `rebase_buffer_base()` call as every other tile (one mechanism, in
   `K_mem_load`/`V_mem_load`).
4. **Keep exactly one scatter path** for multi-page + runtime. Whether the
   scalar-promote *sub-path* survives inside it is decided per §4.

Target: **4 branches → 2** (single-page rebase + scatter), with an optional
scalar-promote sub-branch that survives only if §4 says so.

---

## 3. Evaluation infrastructure — `--paged` / contiguous mode + Triton SAGE

This is the *tool* that lets us answer "what to keep." Without a clean
contiguous baseline, perf numbers are confounded by the `randperm`
(non-contiguous) block table (cache/TLB effects). Build this **first**.

### 3.1 Two levels of "no paging"

| level | what it removes | cost | answers |
|-------|-----------------|------|---------|
| **L1 — contiguous data** | non-contiguous *memory* (identity block table + sequential page fill) | test-only, ~1 day | "how much of the cost is non-contiguous KV access vs the kernel itself?" |
| **L2 — constexpr non-paged kernel** | the `block_tables` walk + rebase/scatter *instructions* entirely | kernel change, constexpr `kIsPaged=false` | "speed-of-light with zero paging addressing" |

Do **L1 first** (cheap, high signal). Only do **L2** if L1 shows the address
instructions (not the memory pattern) are material.

### 3.2 L1 — contiguous data (test-only)

- Add CLI flag `--paged` / `--no-paged` to `op_tests/test_unified_attention_ck.py`
  (argparse at line ~1644). Default `--paged` (current behavior).
- In `_make_inputs` (lines ~518–632), when `--no-paged`:
  - Fill physical block `i` with logical tokens `[i*page_size:(i+1)*page_size)`
    (sequential), and build an **identity** `block_tables` (logical page `j` →
    physical block `j`), instead of the `randperm` (lines ~542–585).
  - `ref_paged_attn` (lines 209–313) already gathers via `block_tables`, so it
    needs **no change** — identity table just makes the gather contiguous.
- This is exactly what CK `42_unified_attention/example_unified_attention.cpp`
  does for `debug_probe==2` (sequential fill + identity table). Low risk.

### 3.3 Triton SAGE contiguous leg

- Wire `aiter.ops.triton.attention.fav3_sage.fav3_sage_wrapper_func` as a new
  backend `run_sage` behind `--sage` / `--no-sage` (mirror `run_triton`,
  lines ~1072).
- Inputs: dense `bshd` `[batch, seqlen, heads, dim]`, **bf16/fp16/fp32** Q/K/V
  (it quantizes internally to int8-QK + fp8-V), `causal`, `window_size`,
  GQA-aware. Returns bf16 `out`.
- **Caveats to document in the report:**
  - SAGE does its *own* quant (int8 QK + fp8 V) — it is **not** numerically
    identical to our per-tensor-fp8 path. Treat it as a **throughput**
    reference, not a correctness oracle; check SAGE against `ref_paged_attn`
    with a *loose* tolerance (or skip its allclose and only time it).
  - It is **dense/contiguous** (`IS_VARLEN=False`), so the apples-to-apples
    comparison is the `--no-paged` (contiguous) UA run, equal-length seqs.
  - bshd vs our `[total_q, h, d]` (varlen) layout: for the contiguous leg use
    equal seqlens so a simple reshape `[b, s, h, d]` is exact.

### 3.4 Contiguous reference

- For `--no-paged` + equal seqlens, `ref_paged_attn` with the identity table is
  already a correct contiguous reference (no new reference needed).
- Optional nicety: a tiny dense `einsum` reference (same math, `[b,s,h,d]`) to
  cross-check, but not required.

### 3.5 L2 — constexpr non-paged kernel path (optional, later)

Only if L1 says the *instructions* matter. Design sketch:
- Add a compile-time `kIsPaged` (template/trait) to the UA pipeline. When
  `false`:
  - skip `block_tables_lds` population and the `refresh_*_offsets` page lookup;
  - the K/V window base is `pool_base + seq_offset*row_stride` (plain
    contiguous strides), advanced by a fixed `kPageBlockSize*row_stride` per
    tile — i.e. the rebase with `phys_page == logical_page` and no table.
  - This makes "rebase optional (constexpr-guarded)" literal: paged → rebase to
    `phys_page`; non-paged → rebase to `logical_page` (a trivial add, no LDS).
- Host side: a `is_paged` arg → `dispatch_variant` picks the `kIsPaged` instance.
- Cost: new instances (kernel-size/build-time), new wrapper arg, new test leg.
  Defer until justified.

---

## 4. Test matrix + keep/delete gates

### 4.1 Shapes (each exercises a specific regime/tier)

| label | shape | regime | tier under test |
|-------|-------|--------|-----------------|
| `sp_d128_ps64` | b16 sq10000 fp8 d128 ps64 (canonical) | single-page | rebase |
| `sp_d128_ps32` | prefill fp8 d128 ps32 | single-page | rebase vs scalar-promote |
| `mp_d64fp8_ps32` | prefill fp8 d64 ps32 | **multi-page** | scalar-promote dedup (the −15.8% claim) |
| `mp_d128_ps16` | prefill bf16 d128 ps16 | multi-page | scatter fallback |
| `rt_pagesize` | `!kHasCePageSize` variant | runtime | scatter fallback |

### 4.2 Gates (run in order; each must pass before the next)

- **G0 — rebase correctness**: `rebuild_and_test.sh` ALL GREEN. *(DONE: 6/6 +
  regression.)*
- **G1 — rebase is not slower**: A/B wall-clock on `sp_d128_ps64` (+ ps32),
  rebase vs current `main`. Require rebase ≤ baseline (within noise). Also
  capture ATT `addr` % (target: < 17.8%) and SGPR/VGPR/occupancy (per-tile
  `init_raw` must not drop occupancy).
- **G2 — rebase subsumes single-page tiers**: with the broadened gate (§2.1),
  re-run G0/G1 on all single-page shapes. If green and ≤ baseline → delete the
  single-page sub-cases of branches (2)-(4).
- **G3 — re-justify each surviving tier**: for the multi-page scalar-promote
  sub-path, A/B on `mp_d64fp8_ps32` with it ON vs OFF. Keep iff ≥ 2% (per the
  decision rule). The earlier comments cite −15.8% — verify it still holds on
  the current FA4 pipeline; if it's now < 2%, delete it too.
- **G4 — fallback minimalism**: confirm branches (3) (exotic, `!kDedupPages`)
  and (4) can be merged into a single scatter path without regressing
  `mp_d128_ps16` / `rt_pagesize`.

### 4.3 What "keep" looks like at the end

Best case: **2 paths** — `rebase` (single-page) and `scatter` (everything
else). Likely case: **3 paths** if `mp_d64fp8_ps32` still needs scalar-promote
(then it's a clearly-commented sub-branch of `scatter`, with the measured
number in the comment so it justifies itself).

---

## 5. Phased execution

| phase | work | GPU? | gate |
|-------|------|------|------|
| **P0** | Prove rebase: wall-clock A/B (`sp_d128_ps64`), ATT overlay, occupancy | yes | G1 |
| **P1** | Build eval harness: `--no-paged` (L1) + `--sage` Triton leg | build only | — |
| **P2** | Collapse single-page tiers into rebase (broaden gate, delete sub-cases) | yes | G2 |
| **P3** | Re-justify scalar-promote + fallback merge; delete what fails 2% | yes | G3/G4 |
| **P4** *(optional)* | constexpr `kIsPaged` non-paged kernel path (L2) | yes | only if P1 shows instructions matter |

P0 is in progress (rebase build is correct; wall-clock ≈ **5.47 ms** on
`sp_d128_ps64`; baseline A/B pending GPU availability).

---

## 6. Risks / notes

- **Identity block table changes cache behavior**: `--no-paged` measures
  *contiguous-memory* perf, not "zero addressing instructions." Be explicit
  about which question each level (L1 vs L2) answers.
- **SAGE quant ≠ our fp8**: use as a throughput reference only; loose/again-its-
  own-ref correctness.
- **Don't delete on a single shape**: a tier that loses on one shape may win on
  another — the matrix (§4.1) must cover each tier's *best* case before delete.
- **Build-time/kernel-size**: L2 adds instances; only pay that if justified.
