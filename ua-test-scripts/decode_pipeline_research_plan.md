# Decode Attention Pipeline — Research Brief & Project Plan

> Audience: the implementation agent that will write the code.
> Status: research + plan only. No code was changed to produce this.
> Scope: the **memory-bound decode** case of the CK unified attention pipeline
> (`3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/`), i.e.
> small batch × long context where K/V HBM loads are the bottleneck.
> Goal: an "innovative pipeline" for decode analogous to what FA4 did for
> the compute-bound prefill case (which already hits ~1900 TFLOP/s).

---

## 0. TL;DR for the implementor

- **Prefill FA4** overlaps two *on-chip compute pipes* (matrix ‖ softmax) to
  approach peak MFMA issue. **Decode is different**: it is **HBM-bandwidth
  bound on K/V**, so the innovation must overlap *HBM streaming* with a tiny
  compute body and keep **many KV tiles in flight at once** (Little's law:
  bytes-in-flight = BW × latency).
- The current serial decode loop keeps **exactly one KV tile in flight** and
  does a **full memory drain (`s_waitcnt_vmcnt<0>`) every iteration** → it
  exposes HBM latency and underutilizes the 5.3 TB/s bus on the
  long-context/small-batch shapes.
- Five stacking levers, in recommended order:
  1. **Design A — deep multi-stage async ring** (raise in-flight tiles 1→N,
     staged `vmcnt` waits). Lowest risk, likely biggest cheap win.
  2. **Design C — Stream-K / Lean partition + fused reduction** (replace
     fixed-split + separate combine kernel). Orthogonal, big at long context.
  3. **XCD/L2 locality swizzle** (MI300 8 XCDs ≈ 8 KV heads). Cheap, stacks.
  4. **Reduced-byte KV** (INT4/MXFP4 + SoA/preshuffled layout). Biggest raw
     bandwidth lever where numerics allow.
  5. **Design B — warp-specialized producer/consumer decode** (the FA4-analog
     headline). Highest effort/risk; only if A leaves the consumer load-stalled.
- **North-star metric: % of peak HBM bandwidth** (and CU occupancy) per shape,
  NOT TFLOP/s. That is decode's analog of the 1900 TFLOP/s prefill number.

---

## 1. What already exists (do not re-invent)

Read of `unified_attention_pipeline.hpp`, `unified_attention_kernel.hpp`, and
`aiter/ops/unified_attention.py`. The decode path today is already strong:

- **GQA heads packed into `M`, fed to MFMA.** `kBlockM = q_seqlen ×
  num_queries_per_kv`, with `decode_m16/32/64/128` tiers (16×16×32 and
  32×32×16 MFMA). This is the FlashInfer "use tensor-core/prefill kernels for
  GQA decode" trick — **already done**. Don't redo the MFMA mapping.
- **FlashDecoding-style fixed split-KV.** Grid `z = num_splits`;
  `_pick_num_splits` targets `num_cus * 4` CTAs, `next_pow2`, clamp 128. Each
  split writes fp32 `o_acc`/`lse_acc`; a **separate Triton `_reduce_segments`
  combine kernel** merges them.
- **Single-warp-group serial inner loop** for decode (`kFA4 == false`). The FA4
  2-warp-group matrix‖softmax overlap and the "WG0 loads V / WG1 loads K"
  cooperative split are **prefill-only**.
- **Paging optimizations:** Tier-0/2 LDS-resident page-table cache, constexpr
  page-size strength reduction, scalar-promoted page indices, per-split page
  window load. Leave these intact.

Net: decode ≈ *FlashDecoding + GQA-packed MFMA + a hand-tuned single-WG core
loop + strong paging*. Headroom is in **(a) how bytes move from HBM**,
**(b) how work is balanced across CUs**, **(c) overlap of load vs compute** —
not in the MFMA math.

---

## 2. Anatomy of the current serial decode loop

The decode path is the `NumWarpGroups == 1` branch (in `operator()`):

Per KV tile it does:
1. `s_waitcnt_vmcnt<0>()` — **full drain** of all outstanding global loads,
   then `__builtin_amdgcn_s_barrier()`.
2. Issue async prefetch of the **next** K and V tile (HBM → LDS).
3. `V_lds_load` + `s_waitcnt_lgkmcnt<0>` → `fmha_alu1` (exp) → PV gemm.
4. `K_lds_load` + `s_waitcnt_lgkmcnt<0>` → QK gemm → mask → `fmha_alu0` →
   `fmha_alu_D_upd`.

So: a **2-stage, 1-tile-ahead, single-warp-group** software pipeline with a
**full memory barrier every iteration**. Correct and well-tuned for
compute-heavy decode tiles (large GQA pack, fp8), but the wrong shape for the
bandwidth-bound regime.

Key code anchors (line numbers approximate; the file is actively edited):
- Serial loop body: the `if constexpr(NumWarpGroups == 1)` block (~L2685-2768).
- `K_mem_load` / `V_mem_load` (async HBM→LDS) and `K_lds_load` /
  `V_lds_load` lambdas (~L1610-1666).
- 2-entry LDS ring: `k_lds_window_store/load`, `v_lds_window_store/load`
  declared with `number<2>` (~L635-669).
- LDS budget: `GetSmemSize()` (~L385-409), Tier-2 page cache 16 KiB
  (`kPageTableLdsEntries`, ~L347).
- `cache_ptr_int32_overflow_possible` long-load path changes the per-load
  instruction count (`K_mem_su_ld_insts` / `V_mem_su_ld_insts`, ~L2677).
- LSE side-output for the combine (natural-log `m + log(l)`), ~L2891-2929.

---

## 3. Why it can't saturate HBM (Little's Law)

Achievable BW is capped by **bytes-in-flight = BW × latency**. On MI300/MI355,
HBM is ~5.3 TB/s with ~hundreds of ns latency → you must keep on the order of
**hundreds of KB of loads in flight per CU** to saturate.

The current loop keeps **one KV tile in flight** (2 buffers: one consumed, one
loading). A decode tile (`kPageBlockSize × kHeadDimPadded`), e.g. page=16,
d=128, bf16, is ~4 KB for K and ~4 KB for V. One tile in flight is far below
BW×latency, so `s_waitcnt_vmcnt<0>()` stalls with the memory system mostly
idle, and the tiny per-tile compute (one GQA-packed QK + softmax + PV) cannot
refill the pipe.

Two compounding factors:
- **Full drain (`vmcnt<0>`)** waits for the slowest outstanding load before
  computing — no decoupling of "consume tile k" from "tiles k+1…k+N arriving".
- **Single warp group (4 waves), ~1 WG/CU (LDS-bound, see the note at
  ~L674-679)** structurally caps memory-level parallelism.

This is the decode analog of the resource-saturation problem FA4 solved for
prefill — except the resource is HBM bandwidth, not MFMA issue.

---

## 4. Design A — Deep multi-stage async ring (do this first)

**Idea:** raise in-flight tiles from 1 to **N** (N ≈ 3–6) and replace the
`vmcnt<0>` full drain with **staged partial waits** (`vmcnt<(N-1)·insts>`), so
the consumer only waits on the *oldest* outstanding tile while N-1 newer tiles
keep streaming.

**Changes:**
- Generalize the 2-entry LDS ring (`k_lds_window_store/load`,
  `v_lds_window_store/load`, currently `number<2>`) to a compile-time
  `kDecodeStages` ring with modulo indexing. Buffers are already separate K/V
  regions, so this is a width + indexing change.
- Prologue: issue loads for tiles `0..N-1`.
- Steady state for tile `k`: consume after `s_waitcnt_vmcnt<(N-1)·per_tile_insts>()`
  (not `<0>`), then immediately issue the load for tile `k+N`. The hard
  per-iteration `s_barrier` becomes a per-buffer dependency.
- Keep the `fmha_alu1 → PV → QK → alu0 → D_upd` body unchanged.

**LDS budget:** N tiles of K+V. Verify against `GetSmemSize()` and the Tier-2
page-table cache (16 KiB). On d=128/page=16 the per-tile cost is small so N=4 is
plausible; large pages may force `kBlockPerCu` down. **Sweep N per decode tier**
(`decode_m16/32/64/128` × d64/d128 × page size) and pick the largest N that
does not cost occupancy.

**Risk:** low–medium. No numerics change, no new warp-group logic. The subtlety
is `vmcnt` accounting across the `cache_ptr_int32_overflow_possible` long-load
path (different instruction counts).

**Gate:** `UA_DECODE_STAGES` (default 2 = bit-identical to today), A/B-able like
the existing `UA_FA4_*` experiments.

**Expected:** moves achieved HBM BW from exposed-latency toward
bandwidth-limited on long-context/small-batch shapes. Cheapest big win.

---

## 5. Design C — Stream-K / Lean partition + fused reduction (orthogonal)

Attacks the cross-CTA axis (independent of A/B).

**Today** (`aiter/ops/unified_attention.py`): `_pick_num_splits` →
`target_ctas = num_cus * 4`, `clamp(next_pow2(raw_splits), 1, 128)` →
**fixed-split** equal `blocks_per_split` (kernel ~L494-499); partials in fp32
`o_acc`/`lse_acc`, merged by a **separate Triton `_reduce_segments` kernel**.
This is exactly the load-balancing / quantization inefficiency Lean Attention
targets, plus a second kernel launch + workspace + HBM round-trip.

**Plan:**
1. **Stream-K planner:** replace `_pick_num_splits` with a `plan()` assigning
   each CTA an equal number of LeanTiles (KV tiles) across (head, split) so
   every CU finishes together regardless of batch×heads×ctx and `num_cus`. Keep
   it static-shape / CUDA-graph friendly (FlashInfer plan-then-run style).
2. **Fused reduction:** softmax rescale is associative; `lse_acc` is already
   natural-log `m+log(l)` and `o_acc` is pre-normalized (see combine notes in
   `unified_attention.py` header). Fold the merge into the attention kernel via
   an atomic/semaphore handoff to the last-arriving CTA per (token, head),
   removing the second kernel. Stage it: keep the Triton combine first (low
   risk), then fuse.
3. **Preserve the split-KV correctness invariant:** the kernel partitions over
   the causal-INDEPENDENT full block count and clamps only the END by the
   per-tile causal horizon (kernel ~L469-504). This fixes a co-owned-token race
   under split-KV + causal + non-dividing GQA. The Stream-K partition MUST keep
   this invariant. Regression fixtures live in `op_tests/test_unified_attention_ck.py`
   (search `regr splitkv`, `_REGRESSION_FIXTURES`).

**Expected:** Lean Attention reports 1.7–2.2× over FlashDecoding at long
context, mostly from tail occupancy + removing combine overhead.

---

## 6. XCD/L2 locality swizzle (cheap accelerator)

Remap the decode grid (`GridSizeDecode`, and the `blockIdx → (kv_head, seq,
split)` decode in the kernel ~L320-367) so all CTAs of a GQA **Attention
Compute Cluster** (query heads sharing one KV head + its splits) land on the
**same XCD**. MI300 has 8 XCDs ≈ 8 KV heads, so KV is fetched into that XCD's L2
once and reused — directly reduces HBM traffic (the bound itself). Pure index
remap, no math change. See arXiv 2511.02132 (swizzled head-first / block-first).

---

## 7. Reduced-byte KV (biggest raw bandwidth lever)

On a bandwidth-bound loop, reading fewer bytes ≈ linear speedup.
- FP8 KV is partly present (`v_descale` fused into the final normalization,
  ~L2870).
- Add INT4 / MXFP4 KV with a **SoA / preshuffled CDNA-aligned layout** (AITER
  `pa_fwd_asm` style), dequantized in-register before MFMA, with V staged in
  LDS once and reused across all GQA query heads sharing a KV head (TurboQuant
  pattern). Co-design with the Tier-0/2 paging cache.
- Numerics-changing → gate + validate like `UA_FA4_EXP2_APPROX`.

---

## 8. Design B — Warp-specialized producer/consumer decode (FA4-analog headline)

**Idea:** repurpose the FA4 2-warp-group machinery, but split into
**memory ‖ compute** instead of matrix ‖ softmax. WG1 = pure **producer**
streaming K/V HBM→LDS as fast as the memory system allows; WG0 = **consumer**
running QK+softmax+PV from the LDS ring. Decouple via the ring + lightweight
buffer-ready flags, not a lockstep per-tile barrier.

Why beyond Design A: doubles the load-issuing waves without the consumer's
compute serializing them; reuses existing cooperative-load infra (`kFA4WG0LoadsV`
/ `kFA4WG1LoadsK`, `v_load_active`/`k_load_active` gating ~L585-589); consumer
compute hides fully under producer streaming.

**Shape:**
- New `kDecodePipelined` policy flag enabling a 2-WG decode instance for the
  memory-bound tiers (mirror the `kFA4` static gate ~L1464).
- Producer: `for k: wait ring-slot k%N free; issue async load; mark in-flight`.
  Consumer: `for k: wait slot k%N filled; QK+softmax+PV`. Cross-WG handoff via a
  small LDS flag array; cheaper than a full block barrier per tile.
- Reuse `core_loop_fa4`'s phase scaffolding, drop the softmax half from the
  producer.

**Critical caveat:** decode is already LDS-bound at ~1 WG/CU. Pursue the
**shared-ring** variant (producer fills, consumer drains the SAME ring — no
duplication) so LDS cost ≈ Design A. Avoid duplicating buffers per WG.

**Risk:** high (occupancy, flag/barrier correctness, register pressure). Do A
first; B only if the consumer still stalls on loads after A.

---

## 9. Recommended sequencing

1. **Design A (deep ring)** — highest ROI, low risk; quantifies the MLP gap.
2. **Design C (Stream-K + fused reduce)** — orthogonal, big at long context.
3. **XCD swizzle** — cheap, stacks with everything.
4. **Reduced-byte KV** — biggest raw win where numerics allow.
5. **Design B (producer/consumer)** — deep headline pipeline, last.

---

## 10. Validation methodology (every phase)

- **Metric:** achieved HBM BW as % of peak (5.3 TB/s) and CU occupancy, per
  shape. Build a decode roofline. This is the north-star, not TFLOP/s.
- **Sweep:** small batch × long context — b ∈ {1, 8, 64}, sk ∈ {4k, 16k, 64k,
  256k}, MHA + GQA ratios {1, 6, 8, 32}, d ∈ {64, 128}, bf16 + fp8. Reuse
  `op_tests/test_unified_attention_ck.py` fixtures + `ua-test-scripts/att_analysis`.
- **A/B discipline:** every lever behind a compile macro; 3-run same-GPU
  medians (the code comments document cross-GPU confounds — e.g. GPU0 vs GPU2
  readings differing by several %). Default macros to bit-identical behavior.
- **Correctness:** keep all existing regression fixtures green, especially
  split-KV + causal + non-dividing-GQA and the SWA decode tiers.

---

## 11. References

- Flash-Decoding — Stanford/Together (2023): split-KV + LSE combine (baseline).
- FlashDecoding++ — arXiv 2311.01282: unified/global max, async double-buffer.
- **Lean Attention** — arXiv 2405.10480: online-softmax rescale as associative
  reduction; Stream-K balanced partition + fused reduction; ~100% occupancy.
- **FlashInfer** — arXiv 2501.01005 (MLSys 2025): GQA→tensor-core decode;
  load-balanced plan-then-run scheduler; block-sparse unified KV.
- **MI300X spatial locality** — arXiv 2511.02132: Attention Compute Cluster,
  XCD/L2 co-location, swizzled head/block-first mapping.
- AITER / vLLM ROCm decode (vLLM blog, 2026): preshuffled SoA KV layout,
  `pa_fwd_asm` / `mla_decode_fwd`, zero-conversion coalesced loads.
- ROCm TurboQuant blog: 4-bit KV dequant, GQA-aware tiling, transposed-V LDS,
  native MFMA dispatch.
