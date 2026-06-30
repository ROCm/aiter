# Jagged-Dense BMM Backward — Optimization Plan du Jour (2026-06-30, MI300X)

Successor to `2026-06-29_final_optimization_plan_mi300x.md`. That plan's day of
experiments converged on a single mechanistic verdict (EXP-2026-06-29d/e):

> **`grad_dense_partials` is MFMA-pipeline-bound.** Doubling occupancy (128×128→64×64:
> VGPR 224→98, LDS 32→16 KB, ~2→~4 WG/CU) and ±2× HBM traffic both left kernel time
> **flat**; vectorizing the staging loads is **neutral**. The only invariant across the
> whole sweep is the MFMA instruction count → the kernel sits at its matrix-core
> **issue/dependency** limit, with loads/LDS/occupancy already hidden behind the feed.

So memory, occupancy, LDS, and staging-vectorization are **exhausted** as levers for that
kernel. This plan attacks the **MFMA feed itself**, primarily by **software-pipelining
(double-buffering) the partials m-tile loop**, which is the user's primary interest.

Phase numbers are local to this doc — **do not reference them in source** (same rule as
the other logs). Each phase has a measurement gate; re-run the §5 configs and append a
dated EXP block to the profiling log on every gate. Correctness gate everywhere:
`example_jagged_dense_bmm_bwd.py --dim {256,512}` cosine > 0.999, **uniform + skew**, no
regression at the repo default.

Kernels: `aiter/ops/flydsl/kernels/jagged_dense_bmm_bwd.py`.

> ## ⚠️ MANDATORY TESTING PROTOCOL (STRICT — applies to every phase) ⚠️
>
> **1. GPU pinning — non-negotiable.** **ALL** testing, profiling, benchmarking, and
> correctness runs in this plan **MUST** be invoked with the environment setting
> **`HIP_VISIBLE_DEVICES=6`**. No run, of any kind, is valid without it. Every command
> in this document is prefixed accordingly; do not strip it.
>
> **2. Benchmark harness — non-negotiable.** All benchmarking **MUST** be run with
> **`op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py`**, in the **`genrec`**
> regime, with **`-mi 16384`**, **`--warmup 25`**, and **`--rep 50`**. The canonical
> invocation is:
>
> ```bash
> HIP_VISIBLE_DEVICES=6 \
> PYTHONPATH=/workspaces/aiter:/workspaces/generative-recommenders:$PYTHONPATH \
> flydsl_venv/bin/python op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py \
>   --regime genrec -mi 16384 --warmup 25 --rep 50 --component dense_bias -test
> ```
>
> (`--component dense_bias` scores the `grad_dense_partials`-driven path that every phase
> below targets; use `--component all` for the full-backward number. The FlyDSL kernels
> take `K`/`N` as compile-time constants — set `K=N=D` in
> `aiter/ops/flydsl/kernels/jagged_dense_bmm.py` (or use the example/profile `--dim`) to
> match the shape under test, e.g. D=256 then D=512.)

---

## 0. Why this is the right lever (the prize)

At D=256 the partials kernel does `2·L·K·N = 2·7.86M·256²` ≈ **1.03 PFLOP**, which at the
measured **473 TF/s** bf16 MFMA ceiling is a **~2180 µs** floor. It currently runs at
**5236 µs = 41.7% of peak**, i.e. **~2.4× of headroom**, and EXP-29d attributes all of it
to MFMA-feed efficiency (26–28% MFMA util is feed latency, not throughput headroom). This
is a far bigger ceiling than the reduce/traffic levers (≤2.4% of the backward, already near
their HBM floor), so the re-prioritization onto the MFMA feed is correct.

`grad_dense_partials` is **51% of the backward at D=256** and **54% at D=512** — the single
biggest kernel; any feed win helps both stars.

---

## 1. Root cause in the current source — the m-tile loop is not pipelined

The inner reduction loop over m-tiles (current source, `grad_dense_partials_kernel`)
stages an m-tile into LDS, barriers, runs the MFMAs, barriers, and repeats — **no
prefetch, two barriers per iteration**, a hard `global-load → ds_write → barrier →
ds_read → mfma → barrier` dependency chain every tile:

```text
for m_tile in range(off_s, num_tiles, SPLIT):
    # transpose-stage J  -> sJ   (32 scalar bf16 buffer_loads + ds_write)
    # transpose-stage dOut-> sD   (32 scalar bf16 buffer_loads + ds_write) + bias add
    barrier()
    for block_k_iter in range(DDENSE_BM // 32):
        copy sJ -> mma_frag_A   (ds_read)
        copy sD -> mma_frag_B   (ds_read)
        gemm(mma_frag_C, A, B, mma_frag_C)
    barrier()
```

Because the stall is a **within-wave feed/sync serialization**, occupancy cannot hide it
(exactly what EXP-29d observed). But **software pipelining can**: overlap m-tile *t+1*'s
global loads + LDS restage with m-tile *t*'s MFMAs.

We already have the target pattern **in the same file**: `grad_jagged_kernel`'s
`run_pipeline_stage` double-buffers the next K-tile (prefetch into the alternate stage
while the current stage's MFMAs run). Phase A ports that idea to the partials m-loop.

EXP-29e shipped the **64×64** sub-tile (LDS 32→16 KB). Its real value (re-framed): the
16 KB headroom is exactly what lets us **double-buffer the staging to 2×16 = 32 KB** while
staying at ≥2 WG/CU. That is the enabling change for Phase A, not "occupancy" per se.

---

## 2. Phased plan

### Phase A — Double-buffer (software-pipeline) the partials m-tile loop  ← PRIMARY
- **Why:** §1 — the m-loop serializes load→stage→barrier→mma→barrier with no overlap; the
  kernel is MFMA-feed-latency-bound (26–28% util at 41.7% of peak). This is the one lever
  consistent with EXP-29d's "MFMA-pipeline-bound" verdict.
- **Do:**
  - Allocate **two LDS staging buffers** `sJ[2]`, `sD[2]` (ping/pong). With the shipped
    64×64 tile that is `2·(64+64)·64·2 B = 32 KB` — same as the original 128×128 single
    buffer, so occupancy is unchanged vs the pre-29e baseline and ≥2 WG/CU holds.
  - **Prologue:** stage m-tile `t0` into buffer 0.
  - **Steady state:** for each m-tile `t`, issue the global loads + `ds_write` for tile
    `t+1` into the *alternate* buffer, then run the `block_k_iter` MFMAs on tile `t`'s
    buffer. One barrier per iteration (after the writes, before the reads of the just-
    written buffer) instead of two; structure it like `grad_jagged`'s `run_pipeline_stage`
    (`read_stage` / `write_stage = read_stage ^ 1`).
  - **Epilogue:** drain the last staged tile's MFMAs with no prefetch.
  - **Keep the fused dBias** accumulation as-is (per-thread fp32 iter-arg summed from the
    dOut it already stages; the LDS combine at the end is unchanged). Do **not** re-do the
    vec8 dBias rework that caused the EXP-29a regression — it is unrelated to pipelining.
  - The dynamic, split-strided trip count (`range(off_s, num_tiles, SPLIT)`) means the
    prologue/steady/epilogue split must handle `num_tiles == 0` (empty/short skew groups,
    already an early-out path) and a 1-tile group (prologue == epilogue, no steady state).
- **Risk / watch:** the loop bound is runtime (`num_tiles` from `M_b`), so the pipeline is
  a runtime `scf.for`, not a constexpr unroll — confirm FlyDSL lowers the ping/pong
  iter-args (or in-place alternating buffers) without re-introducing the barriers. Verify
  no VGPR spill from the prefetch fragments (64×64 has VGPR headroom: 98 → budget to 256).
- **Target metrics:** MFMA Utilization ↑ (toward `grad_jagged`'s 36%, ideally past it);
  Issue-Wait + Dependency-Wait ↓; **partials µs ↓**. Barriers/iteration 2 → 1.
- **Gate:** **≥1.15× on `grad_dense_partials` µs (rocprof kernel-trace) at both D=256 and
  D=512**, correctness green (cos 0.999999, uniform + skew, no D=128 regression).
- [ ] double-buffered m-loop  [ ] correctness  [ ] re-profile + EXP block.

### Phase B — Re-sweep the output sub-tile on the *pipelined* loop
- **Why:** EXP-29e's 64×64 measured neutral **only on the un-pipelined loop**, where the
  barrier/feed stall dominates and masks the within-wave accumulator-ILP difference.
  128×128 gives 64 fp32 C-elems/thread ≈ 16 independent 16×16 accumulator chains; 64×64
  gives 16 elems ≈ 4 chains. Once Phase A removes the sync stall, **accumulator-chain count
  starts to matter**, and the optimum may move back up (or to a rectangular tile).
- **Do:** with Phase A in place, re-sweep `{64×64, 64×128, 128×64, 128×128}` (the env-
  override hook from EXP-29d). For tiles whose double-buffered LDS exceeds the 64 KB cap
  (128×128 ⇒ 2·(128+128)·64·2 = 64 KB, right at the limit; larger ⇒ over), fall back to
  single-buffer or a smaller `DDENSE_BM`. Balance accumulator chains vs occupancy vs LDS.
- **Gate:** pick the per-D optimum; **partials µs ↓** vs Phase A alone, correctness green.
- [ ] tile re-sweep (pipelined)  [ ] re-profile + EXP block.

### Phase C — Widen / vary the MMA atom & K-fragment
- **Why:** even fully pipelined, a 16×16×16 atom amortizes per-MFMA issue/dependency
  overhead over little work. A wider-K or larger atom (if supported) raises work-per-issue,
  directly attacking the 26–28% util.
- **Do:**
  - **First verify hardware support** with a build probe: does gfx942/CDNA3 expose a
    wider bf16 atom (`fx.rocdl.MFMA(16,16,32,bf16)` / `MFMA(32,32,8,bf16)`)? CDNA3's
    native bf16 atom is 16×16×16; the deeper-K variants are **not guaranteed** — do not
    build around an atom that doesn't compile.
  - If available, A/B the atom and re-tune the `(4,4,2)` K-fragment ordering and
    `traversal_order` (currently `GemmTraversalOrder.KNM`) for the pipelined loop.
- **Gate:** MFMA util ↑, partials µs ↓ at both D, correctness green. (If no wider atom
  is supported, this phase is N/A — record the probe result and stop.)
- [ ] atom support probe  [ ] atom/fragment/traversal A/B  [ ] re-profile + EXP block.

### Phase D — Accumulator-ILP without enlarging the tile (fallback / complement)
- **Why:** if Phase B says small tiles win on occupancy but cost accumulator ILP, we can
  recover ILP by carrying **multiple independent C sub-tiles per wave** (split the MFMA
  K-fragment into independent accumulator groups) so consecutive `gemm`s don't chain on the
  same accumulator — lifting MFMA util without growing the output tile or LDS.
- **Do:** restructure the `block_k_iter` MFMA sequence into ≥2 independent accumulator
  chains; measure util vs the single-chain pipelined baseline.
- **Gate:** MFMA util ↑, partials µs ↓, no occupancy regression, correctness green.
- [ ] multi-chain accumulators  [ ] re-profile + EXP block.

---

## 3. Suggested order & expected payoff
**A is the headline** (double-buffered partials m-loop) and the user's primary target; it
is the change that matches the MFMA-feed-bound diagnosis and is enabled by the shipped
64×64 LDS headroom. **B and C/D are the tuning that the pipelined loop unlocks** — the
2026-06-29 plan correctly identified atom/fragment/traversal/accumulator-ILP as "the next
lever", but those were never measured on a *pipelined* loop, so sequence them after A.
Treat tile size as **open**, not settled at 64×64. Re-baseline **both stars (D=256, D=512)
and both regimes** after each phase.

## 4. Explicitly out of scope / disproven (do not re-litigate)
- **Occupancy as a lever** for `grad_dense_partials` — EXP-29d: ~2× occupancy, zero time
  change. (64×64 is kept only as LDS/VGPR *headroom* for Phase A, not for occupancy.)
- **Staging-load vectorization as a standalone win** — EXP-29d: neutral (loads already
  hidden). It may still be folded into Phase A's prefetch for instruction-count tidiness,
  but it is not expected to move time on its own, and the dOut+dBias vec8 rework
  (EXP-29a regressor) must **not** be reintroduced.
- **HBM-traffic / reduce-vectorization levers** for the partials path — low Amdahl and not
  the bottleneck (EXP-29d). Phase C of the 2026-06-29 plan (D=512 reduce-launch removal)
  is the one shipped structural win and stays as-is.
- **`grad_jagged` K-coarsening** — EXP-29b gate-failed (occupancy/latency-bound; extra
  accumulator drove VGPR 36→268).

## 5. Reproduce (STRICT — see the mandatory protocol at the top)

MI300X gfx942, **`HIP_VISIBLE_DEVICES=6`** (mandatory on **every** command), `flydsl_venv`,
rocprof-compute 3.4.0.

**Headline benchmark gate (the number every phase is judged on)** —
`bench_jagged_dense_bmm_bwd_perf.py`, **genrec regime, `-mi 16384`, `--warmup 25`,
`--rep 50`**:

```bash
# D=256 (set K=N=256 in jagged_dense_bmm.py first), then repeat for D=512:
HIP_VISIBLE_DEVICES=6 \
PYTHONPATH=/workspaces/aiter:/workspaces/generative-recommenders:$PYTHONPATH \
flydsl_venv/bin/python op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py \
  --regime genrec -mi 16384 --warmup 25 --rep 50 --component dense_bias -test

# full backward (all components):
HIP_VISIBLE_DEVICES=6 \
PYTHONPATH=/workspaces/aiter:/workspaces/generative-recommenders:$PYTHONPATH \
flydsl_venv/bin/python op_tests/flydsl_tests/bench_jagged_dense_bmm_bwd_perf.py \
  --regime genrec -mi 16384 --warmup 25 --rep 50 --component all -test
```

**Per-kernel attribution (secondary, for mechanism only):** `rocprofv3 --kernel-trace`
(mean/call) and rocprof-compute full counters, also under `HIP_VISIBLE_DEVICES=6`
(see the 2026-06-29 plan §7 for the `rocprof-compute profile` form).

**Always A/B back-to-back in a single run** (shared box; single off readings are
untrustworthy, per EXP-29e's contention caveat).

---

## EXP-30a — Phase A: double-buffered (software-pipelined) `grad_dense_partials` — GATE FAILED (neutral), REVERTED

**Date:** 2026-06-30. **Box/protocol:** MI300X gfx942, `HIP_VISIBLE_DEVICES=6`, `flydsl_venv`,
genrec regime, `-mi 16384`, `--warmup 25`, `--rep 50`, `--component dense_bias --flydsl-only`,
`-b 1024 -d 256 -kout 256` (D=256, SPLIT=2). A/B back-to-back via `git stash` of
`jagged_dense_bmm_bwd.py` only (shape edit `K=N=256` held constant for both legs).

**What was implemented.** Reconciled the in-tree partials tile to **64×64** (the tree was
still at 128×128 — EXP-29e's "shipped 64×64" was never committed; line 533's hardcoded `[64]`
C-fragment confirmed 128×128) and software-pipelined the m-tile loop with **DDENSE_STAGES=2**
LDS double buffering. 64×64 double-buffer LDS = 2·(64·64+64·64)·2 B = **32 KB == the old
128×128 single buffer**, so occupancy is held at 2 WG/CU and the *only* perf-relevant new
variable is the pipelining. Staging views became 3D `(out_dim, m, stage)`; the loop processes
two m-tiles per iteration (constexpr stage 0/1 ping-pong), computing the current buffer while
prefetching the next, OOB-safe over-prefetch via the bounded descriptors (zero-fill ⇒ harmless
to both the MFMA accumulator and the fused dBias). Fixed the SPLIT==1 epilogue `[64]` → computed
`_DDENSE_CFRAG`. **Barriers: 2 per iteration = 1 per computed tile (down from 2).**

**Correctness:** `example_jagged_dense_bmm_bwd.py -d {256,512} --regime {uniform,skew}` all
**PASS** (dDense & dBias cosine 0.999999), SPLIT=2 and SPLIT=1 paths.

**Result — neutral, gate FAILED (target ≥ 1.15×):**

| variant | `grad_dense_partials` (rocprofv3 kernel-trace, µs) | end-to-end `dense_bias` (ms) |
|---|---|---|
| baseline HEAD — 128×128, single-buffer | 5752 | 5.963 |
| this exp — 64×64, **double-buffered** | 5708 | 5.978 |

≈ −0.8 % on the kernel (within run-to-run noise), end-to-end flat. Combined with **EXP-29d**
(64×64-single ≈ 128×128-single), this isolates the pipelining variable: **double-buffering the
m-loop feed is ≈neutral.**

**Why (mechanism).** Confirms EXP-29d: the partials kernel is **MFMA-dependency-bound, not
LDS-feed-latency bound**. Each `fx.gemm` accumulates in place into the same `mma_frag_C`, so the
MFMA pipe is gated by the **accumulator dependency chain** (low MFMA util, issue/dep-wait), which
software-pipelining the *feed* does not touch. Hiding the global-load latency behind the MFMAs
buys nothing when the MFMAs themselves are the critical path.

**Decision:** **Reverted** `jagged_dense_bmm_bwd.py` and the `K=N` shape edit to HEAD (neutral +
added complexity should not ship, per the same discipline that reverted EXP-29b/Phase B). The
implementation is preserved in chat if we want to reuse 64×64 + the staged-LDS scaffold as a
*substrate* for the real lever.

**Next (the real lever, unchanged from §0/§4):** attack the **MFMA pipeline** itself —
**accumulator ILP** (multiple independent C-fragments / split-K-in-registers to break the
dependency chain) and **MMA atom / fragment width / traversal-order** tuning. Double-buffering is
only expected to matter *after* the accumulator chain is broken (once the kernel actually becomes
feed-bound).
