# Decode Pipeline — code-grounded implementation findings

> Companion to `decode_pipeline_research_plan.md`. That doc is the strategy;
> this is the first pass of "look into the code and see how the plans map."
> All line numbers are against
> `3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp`
> at the `jukorhon/fa4-k-preread` HEAD that includes the sliding-window
> page-table fix (`be398c224`). Lines drift as the file is edited — grep the
> quoted tokens.

---

## 0. The problem is real and measured (today's sweep)

`sweep_amir_shapes.py` (FP8, GQA-6, d128, page64, MI355) confirms the plan's
premise. CK-UA decode **wins short context and loses long context**, exactly the
HBM-bandwidth-bound regime:

| batch | Sk=1k | Sk=5k | Sk=10k | Sk=50k | Sk=131k | Sk=196k |
|---|---|---|---|---|---|---|
| 8   | 1.40× | 1.05× | 1.05× | 0.93× | 0.90× | 0.91× |
| 64  | 0.91× | 0.95× | 0.95× | 0.92× | 0.83× | 0.86× |
| 256 | 1.01× | 0.94× | 0.96× | 0.89× | 0.88× | 0.92× |

(× = CK vs Triton; <1 means Triton wins.) Prefill, by contrast, wins every
cell (1.08–1.44×). So the gap is **decode-structural, long-context**, not a
dtype-pipeline issue — the loss grows with Sk (more KV-tile HBM streaming) and is
worst at the batch tier where each CTA owns the most tiles (b=64, splits=16).
This is the Little's-law shortfall from plan §3.

North-star to track (plan §10): **achieved HBM BW as % of 5.3 TB/s**, per shape.
The `att_analysis` overlay + `memwait.py` already break out the vmcnt-wait
fraction; that is the decode bottleneck signal.

---

## 1. Design A (deep multi-stage async ring) — concrete mapping

This is the recommended first lever (lowest risk, biggest cheap win). The decode
loop today is a **2-stage, 1-tile-ahead** software pipeline with a **full memory
drain every iteration**. Raising in-flight tiles 1→N and replacing the full drain
with staged partial waits is a width + indexing + waitcnt change, no new math.

### 1a. The LDS ring is hard-coded to depth 2

- **K/V LDS store windows** — `number<2>{}`:
  - `k_lds_window_store` L670-678, `v_lds_window_store` L680-686.
- **K/V LDS load windows** — `statically_indexed_array<..., 2>`:
  - `k_lds_window_load` L688-695, `v_lds_window_load` L697-704; initialized at
    L788-796.

Generalize both to a compile-time `kDecodeStages` (`generate_tuple(...,
number<kDecodeStages>{})` and `statically_indexed_array<..., kDecodeStages>`),
and change the buffer index from the current `pi / 1-pi / number<0|1>` parity to
modulo-`kDecodeStages` indexing. The store/load descriptors
(`MakeKLdsStoreBlockDescriptor` taking `i_buf`) already parameterize on the
buffer index, so this is mostly a width change — verify `GetSmemSize` /
`Policy::GetSmemSize` lay out `kDecodeStages` K + V regions, not 2.

### 1b. The register tiles are NOT the ring — leave them at depth 1

L709-725: `kv_tile.k_tile` / `v_tile` are **single** register tiles, deliberately
*de*-unioned (the comment explains: separating them lets the K `ds_read` overlap
the PV MFMA). Design A deepens the **LDS** ring (the HBM landing buffers), not the
register tiles — each tile is still consumed into registers immediately after its
`vmcnt` clears. Don't try to deepen `kv_tile`; that's a different (register-
pressure-bound) axis and would re-trigger the kv128 spill problem
(`kv128_vgpr_findings.md`).

### 1c. The full drain to replace

Single-WG decode loop (`NumWarpGroups == 1`, L2968-3051). Every iteration opens
with:
```
s_waitcnt_vmcnt<0>();         // L2980, L3003, L3030 — FULL drain
__builtin_amdgcn_s_barrier();
slide_page_table();           // (new, from the page-table fix)
... prefetch next tile ...
```
`vmcnt<0>` waits for *all* outstanding loads → only ever 1 tile in flight.
Replace with a **staged** wait that only blocks on the oldest of N in-flight
tiles: `s_waitcnt_vmcnt<(kDecodeStages-1) * per_tile_insts>()`.

The per-tile instruction count already exists:
- `K_mem_su_ld_insts` / `V_mem_su_ld_insts` = `*_dram_window.get_num_of_access()`
  (L1829-1830; example staged wait already used at L3027
  `s_waitcnt_vmcnt<K_mem_su_ld_insts + V_mem_su_ld_insts>()`).
- **Subtlety (plan §4 risk):** the `cache_ptr_int32_overflow_possible` long-load
  path (`async_load_tile_raw_long`, K_mem_load L1888-1891 / V_mem_load
  L1914-1917) emits a *different* instruction count per load. The staged
  `vmcnt` constant must be computed from the path actually taken, or the wait
  under/over-shoots. Gate the constant on the same `cache_ptr_int32_overflow_possible`.

### 1d. Prologue

The pre-stage (ends ~L2964) issues loads for tiles 0..1. Extend it to issue
0..kDecodeStages-1 so steady state starts with the pipe full.

### 1e. Gate + budget

- Gate `kDecodeStages` behind `UA_DECODE_STAGES` (default **2** = bit-identical to
  today), A/B-able like the existing `UA_FA4_*` / `UA_K_*` policy macros in
  `unified_attention_pipeline_default_policy.hpp`.
- LDS budget: `GetSmemSize()` L439-444 + `GetPageTableLdsBytes()`. Decode is
  **LDS-bound at ~1 WG/CU** (L709-714 note), so each extra stage of K+V LDS may
  cost occupancy or force `kBlockPerCu` down — **sweep `kDecodeStages` per decode
  tier** (`decode_m16/32/64/128` × d64/d128 × page) and pick the largest N that
  doesn't lose a CTA/CU. The decode K/V tile is small (`kPageBlockSize ×
  kHeadDimPadded`), so N=3–4 is plausible at d128/page≤32.

**Synergy with the page-table fix just landed:** the sliding-window page-table
cache (`be398c224`) decoupled max-KV-length from LDS, so long-context decode no
longer asserts — which is precisely the regime Design A targets. The two stack:
deeper KV ring + unbounded context.

---

## 2. Design C (Stream-K + fused reduce) — where it lives

Orthogonal, big at long context (the plan's 1.7–2.2× claim). Host side, not the
kernel inner loop:
- `aiter/ops/unified_attention.py` `_pick_num_splits` → `target_ctas = num_cus*4`,
  `clamp(next_pow2(...), 1, 128)` — the fixed-split heuristic. Replace with a
  Stream-K `plan()` assigning equal LeanTiles per CTA.
- The separate Triton `_reduce_segments` combine kernel is the second-launch /
  HBM round-trip to fuse away. `o_acc`/`lse_acc` are already fp32 + natural-log
  `m+log(l)` (combine notes in the `unified_attention.py` header), so the merge
  is associative — fold into the attention kernel via a last-CTA atomic/semaphore
  handoff. Stage it: keep the Triton combine first, then fuse.
- **MUST preserve** the split-KV correctness invariant (partition over the
  causal-independent full block count, clamp only the END by the per-tile causal
  horizon — kernel ~L469-504). Regression fixtures: `op_tests/test_unified_attention_ck.py`
  search `regr splitkv` / `_REGRESSION_FIXTURES` (all green in today's `--full`).

Note: the sweep shows the loss is *worst at b=64 (splits=16)*, not the few-CTA
tier — consistent with tail-occupancy imbalance from fixed equal splits, which is
exactly what Stream-K fixes. So C may matter as much as A for the mid-batch band.

---

## 3. XCD/L2 swizzle — cheap, stacks (plan §6)

Pure index remap in the decode grid `blockIdx → (kv_head, seq, split)` decode
(kernel ~L320-367) + `GridSizeDecode`, so all CTAs sharing one KV head land on the
same XCD (MI300/MI355 = 8 XCDs ≈ 8 KV heads) → KV fetched into that XCD's L2 once.
No math change. Do after A.

---

## 4. Recommended next step

1. Prototype Design A behind `UA_DECODE_STAGES` (N=3, then 4). Validate with
   `--full` (correctness) + the decode bands of `sweep_amir_shapes.py --phase
   decode` (perf) + an `att_analysis` overlay on `b8/b64 Sk=131072` to confirm
   the vmcnt-wait fraction drops. Target: move the 0.83–0.91× long-context cells
   toward / past 1.0×.
2. If A leaves the consumer still load-stalled, it quantifies the gap that
   Design C (mid-batch tail occupancy) and B (producer/consumer) would close.

Validation discipline (plan §10): every lever behind a compile macro defaulting
to bit-identical; 3-run same-GPU medians (cross-GPU readings differ several %);
keep all regression fixtures green.

---

## 5. Design A — IMPLEMENTED + MEASURED (2026-06-16): perf-neutral ⇒ wrong lever

Built the N-deep async ring behind `UA_DECODE_STAGES` (default 2 = original,
bit-identical; even N ≥ 2). Touch points:

- `unified_attention_pipeline_default_policy.hpp`: `kDecodeStages` macro +
  `GetRingStages<Problem>()` (decode = N, FA4 = 2); `MakeVLdsStoreBlockDescriptor`
  K-buffer base parameterized (`+2` → `+KBufCount`); `GetSmemSize` = `2·N·KV`.
- `unified_attention_pipeline.hpp`: K/V LDS store+load rings sized `kRingStages`;
  decode loop forked — `kRingStages==2` keeps the exact original 2-buffer
  even/odd pipeline, `>2` runs a new N-unrolled ring: fill K_r1..K_r(N-1) /
  V_r0..V_r(N-2) in flight, then a constant **staged wait
  `vmcnt<(N-2)·(K+V)_insts>`** (leaves N-2 of each in flight) with the freed
  slots refilled at the top (no WAR; overlaps PV+QK). Deferred-PV `sp` stays
  2-deep (parity compile-time since N even); the last tile's PV is finalized
  inline (its V is in ring slot r%N, not the parity slot the shared
  post-process assumes) → `decode_ring_final_pv_done` suppresses that
  post-process. Past-end `K/V_mem_load` harmlessly reload the last tile
  (window stops advancing at split end) → no bound guards, constant threshold.

**Correctness:** `--full` at N=4 = **263 PASS / 0 FAIL / 32 skip**, identical to
N=2 (incl. every split-KV / causal / GQA / prefill fixture).

**Perf (A/B, same GPU, bf16, `sweep_amir_shapes.py --phase decode`, 56 cells):**
N=4 vs N=2 CK time is a **wash — geomean 1.0001×**, every cell within ±2% noise
(long-ctx sk≥10k: 14/32 nominally faster, all noise). No movement on the
0.82–0.85× long-context band.

**Why (HBM BW, sk≥50k):** CK ≈ **5.3–5.9 TB/s**, Triton ≈ **6.0–6.5 TB/s**, peak
≈ 8 TB/s. Both below peak, and N=4 lands the *same* BW as N=2. The decode inner
loop is near-pure memory streaming with tiny `BlockM` (16/32) compute, so
**2-buffering already issues loads at the loop's natural rate** — a deeper ring
adds no issue-rate/BW, just LDS pressure (`2·N·KV`). The ~1.2× achieved-BW gap to
Triton is **load/access efficiency, not prefetch depth.**

**Conclusion / pivot:** Design A does not move decode here. The gap is aggregate
HBM efficiency, so the promising levers are the ones that raise achieved BW or
cut traffic, not pipeline depth:
- **Design C (Stream-K / tail-occupancy)** — loss is worst at b=64/splits=16
  (§2 note), the classic fixed-equal-split imbalance.
- **Reduced-byte KV / better load coalescing** — close the 5.4→6.4 TB/s gap.
- **XCD/L2 swizzle (§3)** — cheap, stacks.

The `UA_DECODE_STAGES` scaffolding is correct + zero-regression (default N=2) and
stays as a dormant, A/B-ready lever; it should **not** be shipped enabled.
