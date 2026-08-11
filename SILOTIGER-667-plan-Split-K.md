# SILOTIGER-667 — Split-K (Phase D / G5) implementation plan

**Scope:** this document tracks **only** the Split-K occupancy-lever work for the
warp-decode MoE kernels. The overall convergence plan remains
[`SILOTIGER-667-plan-10082026.md`](./SILOTIGER-667-plan-10082026.md); this file is
the Phase D / G5 sub-tracker referenced from there.

**Inherited constraints — the main plan's §2 "Locked decisions" still apply in
full.** In particular:
- **Test env:** `flydsl_venv`, e.g.
  `HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python -m pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`.
- **GPU 6** (`HIP_VISIBLE_DEVICES=6`) for clean cold-HBM numbers.
- **Kernel location:** `aiter/ops/flydsl/kernels/warp_decode_moe.py` (+ entry points in
  `aiter/ops/flydsl/warp_decode_moe.py`); reference `*_bart.py` are read-only.
- **Methodology:** one combined op_test; perf only via `run_perftest` (IQR-trimmed device
  time); cold-HBM rotation; `cos ≥ 0.999` + `checkAllclose`; markdown tables.
- **Output-semantics locked decision (§1.2):** keep the **BF16 direct-store fast path** as
  default; the **FP32 `atomicAdd` into a caller-zeroed buffer** epilogue is added **only under
  the split-K variant** (non-deterministic, needs zero-init). This is exactly the G5 surface.

---

## 1. Goal & when it pays

Warp-decode is one-wave-per-output-scalar; small grids (Qwen short-INTER, low B) under-fill
the CU array, leaving CUs idle. Split-K multiplies the grid by `k_batch` (each wave covers
`1/k_batch` of the contraction), filling idle CUs and cutting per-wave iteration count →
lower latency **only in the occupancy-bound regime**. It adds atomic + zero-init overhead, so
it is **default-off** and gated on `grid * k_batch <= CuCount`.

## 2. Design (two structurally different cases)

**down — single-phase (linear reduction).** `y[b,out_j] = Σ_k rw_k · Σ_i inter·w` is linear in
the INTER contraction:
- Split INTER into `k_batch` chunks; grid → `B·(HIDDEN/kh)·k_batch`.
- Each `(output, kb)` wave computes its INTER sub-range partial over all TOPK experts
  (rw folded), then **atomic-fadds** into an FP32 `y` accumulator.
- Finalize: FP32 `y` → bf16 (fold the cast into a finalize wave, or a tiny cast pass).

**gate_up — two-phase (nonlinear silu).** `out = silu(gate_acc)·up_acc` is nonlinear, so the
final output cannot be split-K'd directly. Mirror `moe_gemm_2stage.py`'s 2-phase:
- Phase 1: split HIDDEN into `k_batch`; each wave atomic-fadds partial `gate_acc`/`up_acc`
  into FP32 scratch `[B,TOPK,INTER]` ×2. (PerTensor scale applied in phase 2; Block2D may be
  folded per-partial since it is per-K-block.)
- Phase 2: elementwise kernel reads the two scratch buffers, applies the weight scale +
  `silu(gate)·up`, writes bf16 `out` (this is the existing gate_up epilogue split off).

**Zero-init (the "free" part, vLLM `blockscale_splitk_zero_init`).**
- v1 (correctness-first): caller `torch.zeros_` the FP32 accumulator/scratch (as
  `moe_gemm_2stage` does today).
- v2 (fold away): `kb==0` wave plain-stores (inits), `kb>0` atomic-adds — needs a launch-order
  guarantee or a fused zero-init prologue. Do this only if the memset shows in the profile.

**Reuse / feasibility (already in-tree).** Atomic-fadd is a solved problem here:
`splitk_hgemm.py` / `small_m_hgemm.py` use `llvm.AtomicRMWOp(fadd, …, syncscope="agent")`;
`moe_gemm_2stage.py` implements gate/up split-K partials with caller pre-zero and gfx950
`buffer_atomic_pk_add_bf16` (bf16 atomics halve atomic bandwidth). CU count / arch via the
`flydsl.runtime.device` helpers already imported there.

---

## 3. Steps (tracked)

### Step 1 — down split-K, v1 zero-init  [x]  (2026-08-11)
- [x] Added `k_batch` to `build_down_reduce_fp8_module`. Kernel derives `kb = bid % k_batch`
      (grid `B·(HIDDEN/kh)·k_batch`), restricts each wave to `iters_per_kb = num_iter/k_batch`
      INTER iterations (`k_base = (kb*iters_per_kb + i)*ktile_n + lane*kvector`), and swaps the
      lane-0 bf16 store for an **FP32 atomic-fadd** under `const_expr(split_k)`. `k_batch==1` is
      the unchanged bf16 direct-store path (kb≡0). Guard: `num_iter % k_batch == 0`.
- [x] Added module-level `atomic_add_f32(ptr, elem_off, val_f32)` helper (the
      `llvm.AtomicRMWOp(fadd, …, syncscope="agent", alignment=4)` pattern from the split-K GEMMs).
- [x] Entry point `flydsl_warp_decode_down_reduce(..., split_k=1)`: for `split_k>1` allocates a
      `torch.zeros` **FP32 accumulator**, launches with the `k_batch`-scaled grid, then
      `out.copy_(accum)` finalizes → bf16 (v1; fold later per Step 6). Cache getter keys on
      `k_batch`.
- [x] Correctness op_test `test_down_reduce_split_k` (`DOWN_SPLITK_CASES`): split_k=2 (INTER=2048,
      kv16→num_iter=2) and split_k=4 (INTER=4096→num_iter=4), each **cos 1.000000** vs both the
      fp32 ref and the non-split baseline. **38/38 suite.**
- Note: split granularity is the **iteration** (`ktile_n = 64·kvector`), so split_k needs
      `num_iter = INTER/(64·kvector) ≥ k_batch`. The occupancy target is thus small-**HIDDEN**
      (small grid) with INTER large enough to split (e.g. HIDDEN≤512, INTER≥2048), not the
      INTER=512 shapes — refine the Step 3 shape list accordingly.

### Step 2 — trigger gate + autotune  [x]  (2026-08-11)
- [x] `_cu_count(device_index)` (lru-cached `torch.cuda.get_device_properties().multi_processor_count`;
      gfx950 → **256 CU**) + `_auto_split_k_down(B, HIDDEN, kh_per_warp, INTER, kvector, dev)`: picks the
      largest `k ∈ {8,4,2}` with `num_iter % k == 0` **and** `base_grid·k ≤ CuCount`, else **1**
      (`base_grid = B·(HIDDEN/kh)`, `num_iter = INTER/(64·kVector)`).
- [x] Entry point `flydsl_warp_decode_down_reduce(..., split_k="auto")` resolves the factor after
      `kvector` is known (before alloc/launch); the early assert accepts `"auto"`. Explicit int
      `split_k` still overrides. `k_batch==1` keeps the bf16 direct-store fast path (no atomic/scratch).
- [x] Tests: `test_down_reduce_split_k_auto` (auto path cos 1.000000 vs base) +
      `test_auto_split_k_gate_logic` (divisibility + `base_grid·k ≤ CuCount`, saturated→1).
      On gfx950 the small grid (base_grid=64, num_iter=2) auto-picks **k=2** (128 ≤ 256); a
      DeepSeek-scale `base_grid` (HIDDEN=7168) stays at **1**. **40/40 suite.**

### Step 3 — down perf A/B (prove the occupancy win)  [ ]
- [ ] `@benchmark()` A/B `split_k` on/off across `k_batch∈{1,2,4,8}` on **Qwen3Next-TP1 B=1**
      (INTER=512/256/128) via `run_perftest`; expect a win only in the under-occupied regime.
- [ ] Confirm **neutral/negative on the big-grid DeepSeek** shape (guards the gate). Record the
      Qwen small-grid rows.

### Step 4 — gate_up 2-phase split-K  [ ]
- [ ] Phase-1 builder: `k_batch` split over HIDDEN; atomic-fadd partial gate/up into FP32
      scratch `[B,TOPK,INTER]`×2 (reuse `moe_gemm_2stage` scratch/atomic patterns).
- [ ] Phase-2 elementwise kernel: read scratch, apply scale + `silu(gate)·up`, write bf16.
- [ ] Entry point orchestration (alloc + zero scratch, launch p1, launch p2); correctness vs
      non-split gate_up; A/B on the Qwen small-grid shapes.

### Step 5 — FP4 split-K (optional, after FP8 proven)  [ ]
- [ ] Port `k_batch` to `build_down_reduce_fp4_module` / `build_gate_up_fp4_module` (offset math
      already K3-Tier-1-safe). Same atomic epilogue; measure whether the FP4 win + split-K stack
      on the small-grid shapes.

### Step 6 — fold zero-init (v2), only if profiled as costly  [ ]
- [ ] Replace the `torch.zeros_` with the `kb==0`-inits / `kb>0`-atomic-adds trick or a fused
      prologue; re-measure to confirm the memset overhead is gone.

### Step 7 — record + close  [ ]
- [ ] Add the Qwen small-grid split-K/LDS rows to the main plan's Phase D perf table; check off
      G5 in the main plan's summary table; summarize the win/regime here.

---

## 4. Risks / watch-items
- **Atomic non-determinism** → tolerance-based correctness (`cos`, `checkAllclose`), not
  bit-exact.
- **gate_up 2-phase HBM cost:** scratch `[B,TOPK,INTER]×2` FP32 read+write; for large INTER the
  extra traffic can offset the occupancy gain — measure; prefer **bf16 scratch** on gfx950 if
  precision allows (`buffer_atomic_pk_add_bf16`).
- **Zero-init cost** can eat the win if not eventually folded (Step 6).
- **down finalize cast** adds a pass; fold into a finalize wave to avoid a separate launch.
- **Regime discipline:** split-K must stay default-off on saturated grids — Step 2's gate is
  load-bearing.

## 5. Status log
- 2026-08-11 — plan drafted; feasibility confirmed (atomic-fadd + split-K patterns already exist
  in `moe_gemm_2stage.py` / `splitk_hgemm.py` / `small_m_hgemm.py`).
- 2026-08-11 — **Step 1 done:** down FP8 split-K (iteration-granularity, FP32 atomic-add,
  caller-zeroed accum, bf16 finalize). split_k=2/4 bit-faithful (cos 1.0 vs ref + non-split);
  38/38 suite. Surfaced the `num_iter ≥ k_batch` granularity constraint → adjusted the Step 3
  target shapes to small-HIDDEN / large-INTER.
- 2026-08-11 — **Step 2 done:** CU-count trigger + `k_batch` autotune (`split_k="auto"`).
  gfx950 CU=256; gate picks largest `k∈{8,4,2}` with `num_iter%k==0` and `base_grid·k≤CuCount`,
  else 1. Small grid → k=2, DeepSeek-scale → 1 (fast path). 40/40 suite (2 new tests).
