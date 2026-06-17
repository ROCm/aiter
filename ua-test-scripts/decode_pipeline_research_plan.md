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

---

## 12. Design E — Single-warp, many-CTA-per-CU occupancy attack (2026-06-17)

> Status: planned; E0 diagnostic in progress. Distinct lever from Design A.
> Target: **fp8 long-context decode** (the production `b4 sq1 sk196608 GQA-6 d128
> page64` shape — CK ~4.56 TB/s vs Triton ~4.76, peak ~8).

### Premise / why it's not Design A

Design A (§4, findings §5) deepened the K/V ring **inside one warpgroup** and was
perf-neutral: 2-buffering already issues loads at the loop's natural rate, so more
depth = more LDS, no BW. That is an *issue-rate* lever and it's spent.

Design E attacks the axis the brief itself names as the structural cap (§3):
**"~1 WG/CU (LDS-bound) caps memory-level parallelism."** Raise the count of
**independent** single-warp CTAs resident per CU so 4 staggered (non-lockstep) K/V
streams overlap → more bytes-in-flight → closer to peak HBM. Untested.

### Key grounding: the single-warp pipeline ALREADY exists

The "1 warp / BLOCKM=16 / no LDS roundtrip for S,P" design is the existing
**TinyDecode m16 tier**, so this is *tuning occupancy of an existing path*, not a
new pipeline:
- `decode_d128_m16` / `decode_d64_m16`: `BlockWarps=sequence<1,1,1>` →
  `kBlockSize=64` (one 64-lane warp), `kBlockM=16`, 16×16×32 MFMA, `NumWarpGroups==1`
  (`unified_attention_impl.hpp:191-207,256-263`; `TinyDecodePolicy`
  `NumWarpPerGroup=1`, `default_policy.hpp:788-792`).
- For **bf16/fp16** S (`sp_compute`) and P are already register-resident — P is a
  plain register cast, no LDS roundtrip (`pipeline.hpp:1332-1349`). So LDS is
  essentially *all* K/V ring, exactly the intuition.
- **fp8 m16 caveat:** the 16×16×32 PV gemm forces the **LDS-roundtrip P relayout
  (strategy B, with two `s_barrier`s)** — `pipeline.hpp:1189-1199,1284-1324`. On a
  single-warp CTA the `s_barrier` is a near-no-op but the lgkm drain remains. So for
  the fp8 target, "keep P in registers" is NOT free — closing that (a 16×16 within-
  wave permute instead of the LDS trip) is itself a sub-experiment.

### The knobs fight each other (what E0/E2 must resolve)

| Knob (user intuition) | Helps | Hurts |
|---|---|---|
| Large BLOCKN (= page_size; wide loads, fast SRD-rebase) | per-load coalescing, fewer iters/syncs | LDS+VGPR per CTA scale with BLOCKN → **lower** occupancy |
| Many tiles in flight (deep ring) | MLP within a warp | shown neutral (Design A); costs LDS |
| 4 WGs/CU (independent warps) | **the untested win** (aggregate MLP) | needs *small* per-CTA LDS+VGPR |

"page_size vs BLOCKN synergy" = set **BLOCKN = page_size** so each KV tile is exactly
one page → the existing single-page SRD-rebase fast path fires (`kRebaseKSrd`, needs
`kPageSize % kPageBlockSize == 0`), widest coalesced load, no multi-page math.

### LDS / occupancy facts (gfx950)

- LDS/CU = **160 KiB** (`arch.hpp:1107-1113`). Policy K/V ring = `4·GetSmemSizeKV`
  (2K+2V double buffer, `default_policy.hpp:773-777`).
- m16 d128 **bf16** total LDS ≈ 20 KiB (probe comment: *"LDS pressure alone caps
  decode_d128_m16 at 1 CTA/CU"*, `unified_attention_impl.hpp:301-303`) — but 20 KiB
  on a 160 KiB CU is NOT a hard 1-CTA cap, so the real binder is likely **VGPR=256**
  (bf16 saturating). **E0 must confirm VGPR vs LDS.**
- `kBlockPerCu` is the default `-1`→2 hint for all decode instances
  (`unified_attention_impl.hpp:376-378`, `pipeline.hpp:102-108`); effective residency
  is `min(VGPR, LDS)` bound, not the hint.

### Experiment ladder (each gated, default bit-identical; `--full` must stay green)

- **E0 — diagnose (no code):** for `unified_attention_d128_fp8_mask_decode_t`
  measure VGPR/lane, LDS/CTA (`measure_vgpr.sh`), resident CTAs/CU, achieved BW
  %peak + vmcnt-wait fraction (`att_analysis` on the production shape). Decides the
  binding resource and whether E1 has headroom.
- **E1 — maximize m16 occupancy:** cut the binding resource (single-buffer K/V to
  halve ring LDS; trim persistent VGPR via single-sp / narrower o_acc; raise the
  `kBlockPerCu` bound) and confirm CTAs/CU rises 1→4 and BW follows.
- **E2 — BLOCKN = page_size wide-load sweep:** `BLOCKN ∈ {16,32,64}` aligned to page,
  single-page rebase active; trade load width vs occupancy.
- **E3 — strip per-iter sync:** single-warp loop still does full `vmcnt<0>` drain +
  `s_barrier` each iter (`pipeline.hpp:1693-1694…`); lighten to a staged per-warp
  wait (no cross-warp ordering needed at 1 warp).
- **E4 (stretch) — feed occupancy:** ensure `_pick_num_splits`
  (`unified_attention.py:202-325`) launches enough independent CTAs to fill the new
  residency (ties into Design C Stream-K).

### Risks
- **VGPR, not LDS, may be the wall** → may only fit 2 CTAs/CU regardless of LDS cuts
  (E0 gates this).
- More resident CTAs ⇒ more splits ⇒ more combine overhead (loss is worst at
  b=64/splits=16 — Design C territory).
- fp8 m16 P relayout LDS roundtrip must be made single-warp-cheap or it caps the win.

---

## 13. Design E results + HipKittens 4-wave interleave borrow (2026-06-17)

### E0 diagnostic — DONE (via the torch-free standalone, not the JIT harness)

Tooling: extended `ua-test-scripts/analysis/standalone/ua_trace` to drive the real
decode tier (`SK` decouples context from `sq`; `NUM_SEQS` replicates batch; the
production split-KV heuristic + fp32 `o_acc/lse_acc` workspaces + LSE combine are
wired so CHECK validates). `rocprof_standalone.sh` runs the 4-phase PMC tree in
~40-75 s vs ~10 min on the Python path. Target instance
`unified_attention_d128_fp8_nmask_decode_t` = `decode_d128_m16`, paged64.

Measured (production `sq=1 sk=196608 GQA-8 d128 page64`, fp8):

| batch | grid CTAs (2·b·128 splits) | lat | achieved BW |
|---|---|---|---|
| b=1 | 256  | 65 µs  | 1.54 TB/s |
| b=4 | 1024 | 88 µs  | **4.55 TB/s** |
| b=8 | 2048 | 166 µs | 4.86 TB/s |

PMC (per-dispatch, b=4): `SQ_WAVES=1024`, `VGPR=132`, `LDS≈35 KB`, WG=64 (single
wave). Memory-wait dominates: `TCP_PENDING_STALL_CYCLES≈33M` and `SQ_WAIT_ANY≈19M`
vs `GRBM_GUI_ACTIVE≈1.7M`.

**Verdict (updates §12 risks):**
- **It is genuinely bandwidth-bound at production batch.** BW saturates at b=4 and
  barely moves to b=8 (4.55→4.86) — more resident CTAs no longer help. So the lever
  is **bytes-per-second efficiency per CTA**, not raw occupancy. (Caveat: `rotate=1`
  may let L2 serve some KV; bump `ROTATE` for the true HBM ceiling — the *flat
  plateau* shape holds regardless.)
- **VGPR=132, LDS≈35 KB → the m16 tier is NOT pinned at 1 CTA/CU** (the §12 bf16
  worry). fp8 m16 already runs ~4 waves' worth of work; pure occupancy-raising (E1)
  is therefore *not* the headline win. The headline is **latency-hiding inside the
  streaming loop**: keep enough KV loads in flight to cover HBM latency while the
  MFMA runs.

### HipKittens study (arXiv 2511.08083 + repo `HazyResearch/HipKittens`)

HK has two overlap schedules:
- **8-wave ping-pong** (2 waves/SIMD, leader/follower swap compute↔memory via
  `s_barrier` + shared-LDS double buffer) — *balanced* work (their GEMM + attn
  **forward**, `kernels/attn/gqa/kernel.cpp`, `NUM_WARPS 8`). **This is our prefill
  FA4 analog — leave it alone.**
- **4-wave interleave** (1 wave/SIMD, each wave issues *both* compute & memory,
  finely staggered) — *imbalanced* work (compute- or **memory-**heavy). This is the
  decode model.

Two separable ideas inside "4-wave":
1. **1 wave/SIMD ⇒ full 256 VGPR + 256 AGPR to that wave** (registers are split by
   resident-wave-count per SIMD; paper §3.2 fn). More registers ⇒ deeper in-flight
   load pipeline ⇒ more HBM latency hidden. We already are single-wave; we are
   *under-using* the register file (VGPR=132) to hold in-flight KV.
2. **The interleave itself** (the part to copy), from
   `kernels/attn/gqa_backwards/archive/GQA_bkwd_4warps.cpp` +
   `kernels/attn/gqa/kernel.cpp`:
   - **Double/triple-buffered loads issued ahead** — next tile's `buffer_load`
     issued at the top of the cluster (`tic`/`toc`), MFMA consumes the resident tile.
   - **`__builtin_amdgcn_sched_group_barrier(mask,cnt,group)`** to *force* a fixed
     MFMA:VALU issue ratio (`sched_barrier_pairs<P,N>` = repeat "1 MFMA then N VALU"),
     so the matrix and vector/transcendental (exp2) pipes run in parallel instead of
     serial bursts.
   - **`__builtin_amdgcn_s_setprio(1/0)`** around MFMA clusters so address/memory
     issue doesn't steal slots mid-MFMA.
   - Small register tiles, no LDS round-trip for S/P.
- **Pure-memory-bound floor** (rotary/layernorm, `NUM_WORKERS=1`): when there is
  *no* compute to hide, HK just does 1 wave/block + huge grid + wide
  `buffer_load_dwordx4` + minimal address math. Decode has MFMA to hide, so it wants
  the interleave on top of this.

**1 WG×4 waves vs 4 WGs×1 wave (the user's question):** equivalent *for the 4-wave
interleave* — the waves are independent (no cross-wave handoff), and the per-SIMD
register split depends only on resident-wave-count, so either gives each wave the
full file **iff the launch lands one wave per SIMD** (size to ~4 resident WGs/CU).
Difference is LDS: LDS is per-WG, so 4 separate WGs split the 160 KB into ~40 KB
each (matches the brief), while 1 WG-of-4 pools it. Decode splits are independent
KV ranges that don't share LDS → **4 separate single-warp WGs is the clean choice**
(what we already launch). The ping-pong's leader/follower handoff is the *only*
thing that needs a single WG — and that's prefill, which we keep.

### Implementation plan (decode pipeline rewrite; prefill FA4 untouched)

Gated behind a new `kDecodeInterleave` policy flag on the TinyDecode tiers only;
prefill (`prefill_d128/d64`, FA4 8-warp ping-pong) is a different policy/variant and
is not touched. Default OFF → bit-identical until proven.

1. **Deepen the in-flight KV window** beyond the current 2-buffer: software-pipeline
   N tiles (N from the register/LDS budget, target ~4-8) with `buffer_load` issued
   ahead — this is the actual latency-hider E0 says we lack.
2. **Interleave QK/PV MFMA with the KV loads + softmax** via `sched_group_barrier`
   ratios (port `sched_barrier_pairs`/`_exp_pairs`), and `s_setprio` around MFMA.
3. **Keep S/P register-resident**; for fp8 m16 replace the strategy-B LDS P-relayout
   with an in-wave 16×16 permute (the §12 caveat) so the loop never round-trips LDS.
4. **Strip the per-iter full `vmcnt<0>` drain + `s_barrier`** to a staged per-wave
   wait (single warp ⇒ no cross-warp ordering needed).

Validate every step with the standalone CHECK (b=1/4, paged, split-KV combine) and
the b=1/4/8 BW table above; north-star = achieved HBM %peak, not TFLOP/s.

### Results (2026-06-17) — interleave is neutral; the split CAP is the lever

Implemented step 1 of the plan: `UA_DECODE_INTERLEAVE` (pipeline.hpp, default 0,
decode-only `NumWarpGroups==1`). It issues the next tile's K+V async prefetch
*before* the consume-wait and relaxes the per-iter `s_waitcnt_vmcnt<0>` full drain
to a partial `vmcnt<K_insts+V_insts>`, so ~2 KV tiles stream in flight with **no
extra LDS** (still the 2-buffer ring). CHECK passes on every shape (sk multiple &
non-multiple {8192,8200,4097}, tiny {130,200}, b=4 multi-seq).

**Perf A/B (standalone, fp8 decode_t, sk=196608 GQA-8 d128 page64, rotate=2):**

| shape | OFF | ON |
|---|---|---|
| b=1 | 65.3 µs / 0.77 TB/s | 66.1 µs / 0.76 TB/s |
| b=4 | 69.7 µs / 2.89 TB/s | 70.1 µs / 2.87 TB/s |

→ **Neutral (even slightly worse).** In-loop KV-prefetch depth is NOT the decode
binder — confirms the Design A finding at the source-schedule level too (the backend
already pipelines the `buffer_load_lds`; 2 LDS buffers cap residency regardless). The
macro is left **OFF** (harmless gated infra for a future *deep* ring, where it would
need 4+ LDS buffers to actually raise residency).

**The actual win — the split cap.** E0 said decode is occupancy/MLP-bound at low
batch. The binder is the host `_pick_num_splits` (`unified_attention.py:202-325`):
for b=1 GQA-8 it computes `raw_splits = ceil(num_cus·4 / base_ctas) = ceil(1024/1)
= 1024` then **clamps `min(128, …)`**, launching only 256 CTAs on 256 CUs (~1 of 4
SIMDs busy). Forcing more splits (standalone `NUM_SPLITS`, main-kernel time only):

| batch | splits 128 | splits 512 | splits 1024 |
|---|---|---|---|
| b=1 | 65.3 µs / 0.77 | 21.5 µs / 2.34 | **16.1 µs / 3.13 TB/s** |
| b=2 | 65.1 µs / 1.55 | **24.1 µs / 4.18** | — |

CHECK passes at 512/1024 splits (combine correct). At **high** batch the cap is
already inert (`raw_splits < 128` once `base_ctas` grows), so this only starves
small-batch long-context decode — exactly the production target shape. The cap=128
was set citing a prior MI300X A/B ("gap was per-split work, not occupancy"); this
MI355X (256-CU) data contradicts that for this shape.

**Caveats before changing the default heuristic:**
- Standalone perf times the **main kernel only** — combine excluded. At b=1/1024
  splits combine reads `o_acc[hq,splits,1,d]` fp32 ≈ 4 MiB (~1 µs) vs a ~49 µs
  main-kernel saving, so the net win largely holds, but must be confirmed.
### End-to-end split A/B (2026-06-17, incl. the REAL combine kernel)

`AITER_UA_FORCE_SPLITS` on `test_unified_attention_ck.py` single-shape decode
(b, sq=1, sk, GQA-8, d128, page64, fp8 → bf16; CK time = main + combine, median):

| shape | splits 128 | 256 | 512 | 1024 | best | win |
|---|---|---|---|---|---|---|
| b=1 sk=196608 | 76.9 µs | 46.9 | **40.8** | 108.6 | 512 | **1.88×** |
| b=2 sk=196608 | 77.5 | 49.3 | **45.7** | — | 512 | 1.70× |
| b=4 sk=196608 | 81.2 | **54.1** | 68.1 | — | 256 | 1.50× |
| b=8 sk=196608 | **88.6** | 94.9 | 117 | — | 128 | 1.0× |
| b=1 sk=65536 | 35.5 | **24.1** | 30.5 | — | 256 | 1.47× |
| b=1 sk=16384 | 17.9 | **15.3** | 25.3 | — | 256 | 1.17× |

**The win is real but bounded and shape-dependent** (the standalone's 4× was inflated
by excluding combine). Two findings:
1. Optimal splits **decrease with batch** (b=1→512, b=4→256, b=8→128) — `base_ctas`
   grows with batch so less fan-out is needed; over-splitting pays pure combine cost.
   The current `min(128,…)` cap truncates only the **low-batch** end (the production
   target), where it leaves 1.5–1.9× on the table.
2. Optimal splits **increase with sk** (b=1: 16K→256, 196K→512) — longer per-split
   KV work amortizes the combine, so longer context tolerates more splits.

**The blocker: `_pick_num_splits` cannot see `sk`** — by design it does NOT read
`seq_lens` off-device (CUDA-graph capture safety, docstring L245). So it can't be made
sk-aware without giving that up, and no single static constant wins across the sk×batch
grid: `target=num_cus·2` fixes long-context low-batch but **regresses** sk=16K b=1
(25.3 vs 128's 17.9); `cap=256` regresses b=8 (94.9 vs 88.6). Correctness (CK vs torch)
stays PASS at every split count tested (combine is correct).

### IMPLEMENTED (2026-06-17): sk-aware split heuristic

`_pick_num_splits` (`unified_attention.py`) rewritten — the launch heuristic, NOT the
kernel pipeline, is the highest-ROI decode lever:

```python
hard_cap    = num_cus * 2                              # was num_cus * 4
occ_splits  = ceil(hard_cap / base_ctas)               # fill ~2 waves/CU
sk_ub       = block_tables.shape[1] * key_cache.shape[1]   # capture-safe sk bound
work_splits = sk_ub // _UA_MIN_SPLIT_KV_TOKENS         # =128, env-overridable
num_splits  = clamp(next_pow2(min(occ_splits, work_splits)), 1, hard_cap)
```

Key enabler: `sk_ub = max_num_blocks_per_seq · block_size` is read from tensor SHAPES
only (no device sync) → stays CUDA-graph-capture safe, solving the "heuristic can't see
sk" blocker. `min_split_size = 128` tokens (2 pages) is the best single constant for the
linear work cap (the true optimum grows ~√sk; 128 is optimal at mid/long sk, ≤2 µs miss
at very short sk). The standalone C++ port (`ua_trace_main.cpp:pick_num_splits`) mirrors
it. `AITER_UA_MIN_SPLIT_KV_TOKENS` env overrides for tuning.

**Validated end-to-end (CK incl. combine, no FORCE_SPLITS; heuristic picks):**

| shape | old (cap128) | new | picked splits | net |
|---|---|---|---|---|
| b=1 sk=196608 | 76.9 µs | **41.2** | 512 | **1.87×** |
| b=2 sk=196608 | 77.5 | **45.4** | 512 | 1.71× |
| b=4 sk=196608 | 81.2 | **53.7** | 256 | 1.51× |
| b=8 sk=196608 | 88.6 | 89.4 | 128 | ~neutral |
| b=1 sk=65536 | 35.5 | **31.1** | 512 | 1.14× |
| b=1 sk=16384 | 17.9 | 16.6 | 128 | 1.08× |
| b=1 sk=4096 | 14.3 | 15.0 | 32 | 0.95× (−0.7 µs) |

Correctness PASS at every shape; **regression fixtures all green** (prefill guard still
returns 1; GQA-3/5/6 decode + prefill fixtures unaffected). Only the predicted, negligible
−5% at very short context. The kernel `UA_DECODE_INTERLEAVE` macro stays OFF (neutral).

**Remaining:** A/B vs Triton across the full 640-shape production trace to confirm the net
distribution win; consider the √sk form (C≈1.4) if the mid-sk residual (b=1 sk=65536 picks
512 vs optimal 256) matters.
