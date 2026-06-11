# Project Plan — Conditional Softmax Rescaling for CK Unified Attention

_Audience: an implementing agent with no prior context on this investigation.
Read this top-to-bottom before touching code. Pair it with `ua-test-scripts/STATUS.md`
(the running work-log) for the kernel's broader state._

---

## 0. TL;DR

The CK unified-attention kernel is **softmax-VALU-bound** in its compute phases:
the online-softmax inner loop issues so many VALU ops (exp + the per-iteration
output rescale) that the SIMD VALU pipe saturates and the warp-specialized
pingpong stalls on barriers. This is the same class of problem
**FlashAttention-4** addresses with two tricks; the more portable one is
**conditional (skipped) online-softmax rescaling**: don't rescale the output
accumulator every iteration, only when the running max actually moves enough to
matter, and fold the deferred scaling into a final correction.

This plan has **two parts**:

- **Part 1 (do first, low-risk, no kernel changes):** make the test harness
  generate tensors with a *realistic* softmax distribution, and add an
  instrument that **predicts the rescale-skip headroom per shape** purely in
  Python. This produces a trustworthy benchmark **and** is the go/no-go gate for
  Part 2.
- **Part 2 (kernel change, gated, higher-risk):** implement conditional
  rescaling in the kernel, benchmarked with Part 1's tooling.

**Do not skip Part 1.** Random `N(0,1)` tensors give a near-uniform softmax whose
running max barely moves, which makes the optimization look like it skips ~100%
of rescales (unrepresentative) while never exercising the rescale path (so
correctness of the deferred-stat bookkeeping goes untested). See §3.

---

## 1. Background: why softmax VALUs are the bottleneck

On modern accelerators, matrix-engine (MFMA/Tensor-Core) throughput keeps
doubling while the units that compute `exp` and elementwise rescales do not. So
softmax's non-matmul work becomes the gate. Evidence in this kernel
(from `STATUS.md`'s rocprof analysis on FP8 `prefill_d128`, `b=16 sq=sk=10000`):

- The warp group running the compute phases (`W0-3`, `core_loop(0)`) **gates the
  barriers** because "its 4 waves keep the SIMD VALU pipe busy and lose
  arbitration" → `ARBITER_NOT_WIN 35.3%`, **VALU samples 48.8%** vs the other
  group's 35.8%.
- The two VALU hotspots are:
  1. `fmha_alu1`'s `exp2` over the score tile (plus, on FP8, a
     `cvt_pk_fp8 + ds_bpermute` re-layout cluster), and
  2. the **`fmha_alu_D_upd` rescale tail**: ~16× `v_pk_mul_f32` rescaling the
     128-VGPR PV accumulator by `exp2(scale_s·(m_old − m_new))` every iteration.
- Hotspot (2) is the `T_D` term that `STATUS.md`'s warp-group-imbalance analysis
  identified as the dominant driver of barriers B2/B3.

**This plan targets hotspot (2): the per-iteration output rescale.** (Hotspot (1),
polynomial `exp` emulation à la FA4, is a *separate, weaker* lever on AMD —
see §6 — and is out of scope here.)

### Research references (read these for the technique)
- **FlashAttention-4** (arXiv:2603.05451), §3.1.4 "Skipping online softmax
  rescaling" — the exact technique. Conditional rescale when
  `m_j − m_{j-1} > τ` with `τ = log₂(256) = 8`; track total deferred scaling and
  apply the true denominator at the end; "rescale if any thread in the warp needs
  it" to avoid divergence. Reports ~10× fewer corrections.
- **FlashAttention-3** (NeurIPS 2024, arXiv:2407.08608), §3 — frames the
  matmul-vs-exp throughput disparity and the warp-specialized pingpong this
  kernel already mirrors.
- Modal, "We reverse-engineered Flash Attention 4" — readable companion that
  walks the conditional-rescale logic.

---

## 2. Where the relevant code lives

| Concept | File / location |
|---|---|
| Kernel pipeline (all `fmha_alu*` lambdas, the core loop) | `3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp` |
| `fmha_alu0`: computes running max `m`, saves `m_old`, shifts scores by **`m`** | ~line 1232; score-shift `fma_impl_vsv(..., scale_s, -scale_s*m)` at ~line 1264 |
| `fmha_alu1`: `exp2(sp_delta)`, rowsum, `l` update, **partial** O rescale (first 6 regs) | ~line 1270; partial rescale `o_acc *= o_acc_scale` at ~line 1322; `tmp = exp2(scale_s*(m_old−m))` at ~line 1315 |
| `o_acc_scale` field | declared ~line 1228 (comment: "rescale o_acc in fmha_alu1() & fmha_alu_D_upd()") |
| `fmha_alu_D_upd`: computes `o_acc_scale = exp2(scale_s*(m_old−m))`, rescales the rest of the 128-VGPR accumulator via `v_pk_mul_f32` | ~line 1640 (factor at 1641, packed mul at ~1651/1669) |
| `fmha_alu_D_reg_cnt = 6` (how many O regs get rescaled in alu1 vs the D_upd tail) | ~line 540 |
| Per-phase schedule + the three barriers (B1/B2/B3) + `i_total_loops` | core loop ~lines 1615–1742; `fmha_alu_D_upd()` call sites ~1808/1912/1968/2015 |
| UA-owned scheduler (fork of `CoreLoopScheduler` for tuning `sched_group_barrier` hints without touching the FMHA pipeline) | `unified_attention_core_loop_scheduler.hpp` (`UAcoreLoopScheduler`) — created during STATUS.md "Experiment 1.5" |
| Test harness (tensor synthesis + correctness + perf) | `op_tests/test_unified_attention_ck.py` |
| `_make_inputs` (current random-tensor synthesis) | `op_tests/test_unified_attention_ck.py` ~line 197 |
| `ref_paged_attn` (full-fp32 softmax oracle — valid for ANY input distribution) | same file ~line 146 |
| Production-shape sweep driver / analyzer | `ua-test-scripts/sweep_amir_shapes.py`, `analyze_sweep.py` |
| Decode regression sweep | `ua-test-scripts/regression_decode.sh` |

### The current online-softmax math (what you're changing)
Per K/V block iteration `j`, the kernel does:
1. `m_old = m`; `m = max(m, rowmax(S_j))`  (`fmha_alu0`)
2. `P_j = exp2(scale_s·S_j − scale_s·m)`; `rowsum_j = sum(P_j)`  (`fmha_alu1`)
3. `o_acc_scale = exp2(scale_s·(m_old − m))`  (the rescale factor)
4. `l = o_acc_scale·l + rowsum_j`  (denominator)
5. `o_acc *= o_acc_scale`  (the **expensive** 128-VGPR rescale, split across
   `fmha_alu1` + `fmha_alu_D_upd`)
6. `o_acc += P_j · V_j`  (the PV gemm)

When `m` doesn't change, `o_acc_scale == 1.0` and step 5 is wasted work — but the
kernel does it unconditionally every iteration. That is the target.

---

## 3. Why the test tensors must be made realistic (READ THIS)

The harness currently builds Q, K, V with `torch.randn` (`_make_inputs`,
~line 208–215). For `head_size=128` and scale `1/√128`, each logit
`S = scale·Σ qₖkₖ ≈ N(0,1)`. Softmax over thousands of `N(0,1)` logits is nearly
**flat** (top weight ~0.004 over `sk=10000`). Consequences for *this*
optimization:

- **Running max barely moves.** Block maxima of `N(0,1)` are all ≈ 3–4; across
  blocks the running max grows by record-statistics (~`ln(num_blocks)` tiny
  increments, each ≪ τ=8). So at any sane τ, random tensors **skip ≈100% of
  rescales** → benchmark *overstates* the benefit and **never fires** the rescale
  path (the risky deferred-stat code goes untested).
- **Real attention** has large, structured logit spread (trained Q/K alignment,
  attention-sink/initial-token spikes, local-window peaks); effective logit std
  is more like 4–12, and high-scoring keys can appear *late* in stream order,
  genuinely pushing the running max past `m_commit + τ`. That gives a
  *representative* skip ratio **and** exercises the rescale path.

The realism levers that move rescale frequency: **(a) logit magnitude/spread,
(b) peakedness (a few dominant keys), (c) where peaks fall in block/streaming
order** (sinks-at-front vs scattered). V's distribution is irrelevant to rescale
frequency (only affects output magnitude/tolerance).

---

## 4. Part 1 — Realistic tensors + rescale-headroom instrument (no kernel changes)

All in `op_tests/test_unified_attention_ck.py` (+ small sweep additions).
**Default behavior must stay bit-identical** so existing CI numbers don't move.

### 4A. Gated realistic generator in `_make_inputs`
Add CLI `--logit-dist {uniform,realistic}` (**default `uniform`** = today's
`randn`, bit-identical). The `realistic` path post-processes the `randn` Q/K
**before** FP8 quantization (so quantization sees realistic activations, matching
production):
- `--logit-std G` — scale Q so per-row logit std ≈ `G` (base ≈ 1, so gain ≈ `G`).
- `--peak-frac F --peak-gain P` — add a shared low-rank direction `u` to a
  fraction `F` of keys (and to Q) so a few keys get systematically high scores
  → peaked softmax.
- `--sink-tokens N` — bias the first `N` keys of each sequence upward along the
  Q-mean direction (StreamingLLM-style sink). This concentrates the max **early**
  in stream order (skip-friendly, realistic for decode).
- Must interact correctly with the existing causal mask (`mask_type=2`) and the
  paged `block_tables` (sinks map to the correct physical pages).

### 4B. Rescale-headroom instrument (the key de-risker)
Pure-Python, runs from the same generated Q/K, **no kernel involvement**. Replay
the online-softmax running-max trajectory **in the kernel's block/streaming
order** and report, per shape:
- softmax entropy / effective #attended-keys (perplexity), per-row max-logit and
  logit-std — confirms the distribution is realistic and non-degenerate;
- **rescale-trigger count vs. always-rescale count at τ ∈ {0, 4, 8, 12}** — i.e.
  the predicted skip ratio. `τ=0` reproduces today's always-rescale; the curve is
  the headroom estimate **and the τ-sensitivity, computed before any kernel
  code.**

This table is the **go/no-go gate**: if realistic tensors still skip ~100% at
τ=8, Part 2 is "free skip everywhere" (low risk, do it); if they rescale often,
you've quantified the accuracy/perf knee and which shapes benefit.

### 4C. Calibration (don't hardcode "realistic")
Sweep `--logit-std ∈ {1,4,8,12}` (1 = current random baseline) and report 4B's
skip-ratio + entropy for each. Add a couple of representative cells to
`ua-test-scripts/sweep_amir_shapes.py` so this rides existing sweep tooling.
*(Optional, higher-fidelity:)* a `--qk-from-file` path replaying captured real
logit stats from one small-model forward pass — nice-to-have oracle, not required.

### Part 1 acceptance criteria
- `--logit-dist uniform` (default): existing `--quick`/grid/`regression_decode.sh`
  numbers bit-identical.
- `--logit-dist realistic`: `ref_paged_attn` still PASSES at all swept stds (it's
  a full-fp32 oracle, so it must — if it doesn't, the generator is buggy).
- 4B prints a skip-ratio table that is ≈100% for `uniform` (this *confirms* the
  "random overstates the benefit" hypothesis) and clearly non-degenerate for
  realistic settings.

---

## 5. Part 2 — Conditional rescaling in the kernel (gated; do after Part 1)

Implement in `unified_attention_pipeline.hpp`, behind
`#define CONDITIONAL_RESCALE 0` (default off — follow the parked-lever convention
used by `MOVE_FMHA_MASK_TO_GEMM1` etc. in STATUS.md).

### 5A. Algorithm
Introduce a **committed max** `m_commit` and threshold `τ` (start `τ = 8`,
i.e. `log₂256`):
1. Keep computing the true running max `m` in `fmha_alu0`, but **shift scores by
   `m_commit`**, not `m` (change the `-scale_s*m` term at ~line 1264 to
   `-scale_s*m_commit`). Scores stay bounded: `S − m_commit ≤ τ` ⇒
   `exp2 ≤ 256`, safe in fp32.
2. Compute a **wave-uniform** predicate `need_rescale = any_lane(m − m_commit > τ)`
   (ballot/reduce; the `permlane32_swap` reductions already in `fmha_alu0/1` for
   `kWarpGemmM==32` show the cross-lane idiom).
3. Only when `need_rescale`: set `o_acc_scale = exp2(scale_s·(m_commit_old −
   m_commit_new))`, rescale `o_acc` (the `fmha_alu1` partial at ~1322 **and** the
   `fmha_alu_D_upd` tail at ~1651/1669) and `l`, then advance `m_commit`.
   Otherwise skip steps 5 (and the corresponding `l` rescale).
4. **Final correction** at loop exit (near the `fmha_alu_D_upd()` epilogue call
   sites ~1968/2015): fold the total accumulated deferred scaling into `l`/output
   so the normalized result matches the always-rescale path.

### 5B. Scheduler-hint integration (the known trap — see STATUS.md Experiments 1/1.5/2)
A conditional rescale means **variable instruction counts per phase**, but the
`sched_group_barrier` hints are statically sized for the always-rescale path.
STATUS.md showed mismatched hints get **reabsorbed into WAITCNT/ARBITER stalls**
(no net win) and, on FP8, **break correctness** by disrupting the
`cvt_pk_fp8 + ds_bpermute` cluster. Therefore:
- Do the hint changes in `UAcoreLoopScheduler` (the UA-owned fork), **not** the
  shared FMHA scheduler.
- Expect **bf16 and FP8 to behave differently**; treat bf16 as the leading
  indicator and validate FP8 separately. A **bf16-only landing is an acceptable
  fallback** if FP8 hint interaction proves too costly.

### 5C. Benchmark protocol (use Part 1)
- Benchmark with `--logit-dist realistic` (calibrated std from 4C), **not** random
  tensors. Report the measured speedup **against the 4B-predicted headroom**.
- Report **bf16 and FP8 separately** on the canonical `prefill_d128`
  `b=16 sq=sk=10000` cell and a decode cell (e.g. `b=128 sq=1 sk=16384`).
- Use `ua-test-scripts/rocprof_warpgroup_balance.py` to confirm the `T_D`/B3
  barrier wait actually drops.

### Part 2 acceptance criteria
- `CONDITIONAL_RESCALE=0`: codegen bit-identical to baseline (verify a couple of
  shapes).
- `CONDITIONAL_RESCALE=1`: `regression_decode.sh` (8 configs) PASS, no perf
  regression; the 68-cell `sweep_amir_shapes.py` correctness PASS (within the
  existing `2·atol` catastrophic cap + `1e-5` outlier guards described in
  STATUS.md "Test tolerance fix").
- Measured perf delta on the canonical cells consistent with the 4B headroom; if
  FP8 regresses (hint reabsorption), land bf16-only and park FP8 behind the macro
  with a note.

---

## 6. Scope, risks, and decisions left to the implementer

- **Out of scope:** FA4's polynomial `exp` emulation (hotspot 1). On AMD/CDNA
  there is no separate SFU — `v_exp_f32` is a quarter-rate transcendental on the
  *same* VALU as FMAs, so replacing one `v_exp_f32` with a 3-FMA Horner chain is
  roughly an issue-slot wash, not the free parallel win it is on NVIDIA. Revisit
  only if Part 2 lands and exp remains the gate.
- **Bounded ceiling:** the rescale tail is the `T_D` share of the ~12% total
  barrier-wait budget (STATUS.md), so expect a low-single-digit % kernel win, not
  a step change. Part 1's 4B number sets the realistic expectation per shape.
- **Decisions for the implementer:** the τ value (start 8); whether to advance
  `m_commit` straight to `m` or in τ-quanta; exact placement of the
  `need_rescale` branch relative to the existing B1/B2/B3 barriers; and whether to
  land bf16-only.
- **Numerical guardrail:** with `τ=8`, intermediate `exp2` is bounded by 256 in
  fp32 — fine. Do **not** raise τ so high that `exp2(τ)` threatens fp32 range for
  the accumulated `l`.

---

## 7. Suggested order of work
1. Part 1A + 1B + 1C (harness). Land it; it's independently useful and low-risk.
2. Run 4B/4C, decide go/no-go and pick τ + calibrated logit-std.
3. Part 2A behind `CONDITIONAL_RESCALE`, bf16 first, validate correctness.
4. Part 2B scheduler hints in `UAcoreLoopScheduler`; re-validate bf16, then FP8.
5. Part 2C benchmark vs. predicted headroom; land or park per acceptance criteria.
6. Update `ua-test-scripts/STATUS.md` with results (follow its existing
   experiment-log format).
