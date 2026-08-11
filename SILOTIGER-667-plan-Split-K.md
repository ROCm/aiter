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

### Step 1 — down split-K, v1 zero-init  [ ]
- [ ] Add `k_batch` to `build_down_reduce_fp8_module` (start with FP8; the simplest, most-used
      path). Kernel derives `kb` from a grid dim, restricts its INTER range to
      `[kb*chunk, (kb+1)*chunk)` (`num_iter/k_batch` iters), and swaps the lane-0 bf16 store for
      an **FP32 atomic-fadd** into a caller-zeroed accumulator (reuse the `llvm.AtomicRMWOp`
      pattern from `small_m_hgemm.py`).
- [ ] Add an `atomic_add_f32` helper to `buffer_ops` (or a local kernel helper) if none is
      directly reusable.
- [ ] Entry point `flydsl_warp_decode_down_reduce(..., split_k=1)`: allocate FP32 `y` accum,
      `zeros_` it, launch phase-1, then cast → bf16 `out` (finalize wave or small cast pass).
- [ ] Correctness op_test: split_k∈{2,4} vs non-split on a Qwen short-INTER B=1 shape; assert
      `cos ≥ 0.999` (tolerance for atomic non-determinism, not bit-exact).

### Step 2 — trigger gate + autotune  [ ]
- [ ] Query CU count (device props / `runtime.device`). Pick largest `k_batch ∈ {1,2,4,8}` with
      `grid * k_batch <= CuCount`; default **1 (off)** for saturated grids (DeepSeek).
- [ ] Wire the auto-pick into the down entry point behind the existing shape logic; keep the
      bf16 direct-store path when `k_batch==1` (no atomic, no scratch).

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
- 2026-08-11 — plan drafted; no code yet. Feasibility confirmed (atomic-fadd + split-K patterns
  already exist in `moe_gemm_2stage.py` / `splitk_hgemm.py` / `small_m_hgemm.py`).
