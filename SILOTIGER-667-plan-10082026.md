# SILOTIGER-667 — Gap-Closing Implementation Plan (Living Document, 2026-08-10)

**Ticket:** [SILOTIGER-667] MoE decode warp-decode kernels (small-M): FP8 + MXFP4 gate_up/down
**Goal of this doc:** Drive the WIP FlyDSL warp-decode MoE implementation
(`aiter/ops/flydsl/kernels/warp_decode_moe.py` + `aiter/ops/flydsl/warp_decode_moe.py`,
branch `samaario/warp-decode-moe`) up to the functional scope of the reference
"bartified" implementation (`kernels/moe_warp_decode_bart.py` +
`warp_decode_moe_bart.py`, branch `samaario/silotiger-667-warp-decode-bartified`)
and the CK-Tile reference, **without regressing** what the WIP already does better.
This is a *living document* — update the status boxes and notes as work progresses.
Supersedes `SILOTIGER-667-plan.md` (which tracked the FP8 baseline); that plan's
Phases 0–4 are the starting point (all [x] except the CK comparison).

---

## 1. Where we are (delta vs the reference, 2026-08-10)

The WIP shipped an FP8 correctness/perf baseline for **both** kernels (gate_up +
down_reduce) with PerTensor/PerToken/**Block2D** scales, H2 down, a production op_test,
and near-peak BW at B=1 on E=8 shapes. Comparing it against the reference surfaced the
following **gaps** (things the reference has that the WIP does not) and **divergences**
(things the two do differently, where the WIP is sometimes ahead).

### 1.1 Missing in the WIP (present in the reference and/or ticket scope)

| # | Capability | Reference has | WIP has | Ticket priority |
|---|---|---|---|---|
| G1 | **i64-safe weight/activation offsets** (large E) | ✅ per-row i64 base | ❌ single i32 element offset | **correctness blocker** |
| G2 | **MXFP4 / FP4 weights** (down + gate_up) | ✅ (down full; gate_up fp4) | ❌ | #1 remaining (ticket) |
| G3 | **BF16-weight path** (`w_dtype="bf16"`) | ✅ (dot2 + scalar) | ❌ FP8-only | scaffold / gfx942 |
| G4 | **gfx942 / scalar-f32 fallback** (`use_dot2=False`) | ✅ auto-arch | ❌ gfx950-only | portability |
| G5 | **Split-K** (`k_batch`) + zero-init fusion | ✅ (gate_up 2-phase, down) | ❌ | ticket lever |
| G6 | **LDS cooperative caching** (`n_waves`) | ✅ (down; gate_up dead) | ❌ | ticket lever |
| G7 | **s_nop-free / independent-accumulator dot2 (ILP)** | ✅ (fp8/fp4) | ❌ serialized `s_nop 2` | perf (deferred) |
| G8 | **Software-pipelined weight prefetch** | ✅ (bf16 dot2) | ❌ | perf (B=1 fp4) |
| G9 | **CK-Tile cross-benchmark harness** | ✅ (`ck_bench_*.cpp` + compare) | ❌ | validation |
| G10 | **Package public API registration** (`__init__.py`) | ✅ | ❌ | integration |

FP8/MXFP8 **activation** input is missing from *both* (both take BF16 activations).
Not a WIP-specific gap, but an unaddressed ticket datatype target — tracked as follow-on.

### 1.2 Divergences (WIP is ahead — do not regress)

- **Scale layouts:** the WIP supports **PerTensor + PerToken + Block2D** FP8 scales on
  *both* stages (folded exactly into the f32 accumulator). The reference FP8 path only
  supports a **single PerTensor scalar** (no PerToken, no Block2D for FP8; only MXFP4
  e8m0). Neither is a superset — the convergence target must keep the WIP's scale
  coverage *and* add the reference's datatype/perf coverage.
- **Output semantics:** the WIP `down` writes **BF16 directly** to `y[B,HIDDEN]`
  (self-contained, matches the ticket's BF16 output). The reference writes **FP32 via
  `atomicAdd`** into a caller-zeroed buffer (required by split-K, non-deterministic,
  needs a zero-init). Keep the BF16 direct-store fast path; add the atomic/scratch
  epilogue only under the split-K variant (G5).
- **`kh_per_warp` generality:** WIP generalizes down over arbitrary `kh`; reference
  hardcodes `{1,2}`. `kh=2` is optimal, so this is a WIP nicety to preserve.
- **Testing rigor:** WIP op_test uses `run_perftest` (IQR-trimmed device time), cold-read
  rotation, `cos ≥ 0.999` + `checkAllclose`, markdown tables, pytest-collectable. The
  reference bench uses a hand-rolled `time.perf_counter` loop (the under-warmed/warm-cache
  pattern the methodology forbids) but adds the CK C++ comparison the WIP lacks (G9).
- **Style / surface:** WIP uses the current `fx.*` surface and the ROCDL
  `cvt_scalef32_pk_bf16_fp8` op (no inline asm for the convert); reference uses raw
  `arith`/`scf`/`llvm` dialects + inline-asm converts with `op_sel`. Keep the WIP surface;
  it aligns with `flydsl-kernel-authoring` / `flydsl-kernel-code-cleanup`.

---

## 2. Locked decisions

The first four are **carried over verbatim (adapted) from `SILOTIGER-667-plan.md` §2**
per the ticket owner's instruction; the remainder are the other original §2 decisions
(plus the resolved E8M0 scale rule) that are **still valid and recommended for retention**.
All remain in force.

- **Test environment:** run all tests in **`flydsl_venv`** (has the correct deps, incl.
  triton 3.6.0):
  `./flydsl_venv/bin/python -m pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`
  (or `./flydsl_venv/bin/python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`). The
  default env's triton 3.3.1 < gluon's 3.6.0 requirement, which blocks `import aiter`.
- **Kernel location:** `aiter/ops/flydsl/kernels/warp_decode_moe.py` (+ a Python
  wrapper/entry point in `aiter/ops/flydsl/warp_decode_moe.py`), matching the existing
  MoE FlyDSL layout. New datatype/perf variants extend these files (or a sibling in the
  same `kernels/` dir); the reference `*_bart.py` files are **read-only references**, not
  the deliverable.
- **`v_dot2_f32_bf16` primitive:** implement as a **local helper inside the kernel
  module** via `llvm.inline_asm` — do **not** add a dependency by editing the installed
  FlyDSL package. (Pattern reference only: `flydsl/expr/rocdl/inline_asm.py`.)
- **Benchmarking & testing methodology (production-representative):** we target production
  use-cases, so all perf numbers must come from the shared harness — never ad-hoc
  `time.perf_counter` loops (those are under-warmed / warm-cache and misreport BW):
  - **One combined op_test, not separate scripts.** Correctness *and* perf live in the same
    `test_flydsl_warp_decode_moe.py`, per the `aiter-op-test` skill: `@benchmark()` fn +
    `run_perftest` candidate loop + `checkAllclose(ref.to(fp32), out.to(fp32), ...)` +
    a final markdown table with `us` / `TFLOPS` / `TB/s` / `err` per candidate. The torch
    reference is computed and compared but **never timed / never in the table**.
  - **Always time via `run_perftest`** (`aiter.test_common`). It does the warmup+repeat and
    reports IQR-trimmed torch-profiler **device** time (pure kernel). Any published TB/s for
    this ticket must be a `run_perftest` number.
  - **Warmup + iters for these tiny B=1 decode kernels:** use at least **`num_warmup=5`,
    `num_iters>=100`**. Pure-correctness-only checks may use the small `num_iters=2,
    num_warmup=1` convention since perf is not being measured there.
  - **Cache handling = cold HBM reads.** Keep `num_rotate_args` at its **default (auto
    L2-fill)** so each timed iter streams weights cold from HBM. Do **not** force
    `num_rotate_args=1` (warm-cache) except to dodge OOM on very large inputs, and if so,
    label the number as warm-cache. (See the op_test's `_rotate_for` helper.)
  - **Timing modes:** report the default **device** time as the headline BW; additionally use
    **`use_cuda_event=True`** (wall-clock, includes host dispatch) when characterizing the
    Python entry point's per-call `ptr_arg(...)` + `current_stream()` overhead, and
    `testGraph=True` for the low-host-overhead graph-replay figure.
  - **Roofline:** compute `TB/s` from the weight bytes actually streamed (FP8 = 1 B/elt,
    FP4 = 0.5 B/elt — the dominant term) and quote it against gfx950 HBM peak.
- **`kVector` default:** `kVector=16` (one 128-bit FP8 transaction) when
  `HIDDEN % 1024 == 0` (gate_up) / `INTER % 1024 == 0` (down); fall back to `kVector=8`
  otherwise. For the **MXFP4** path add `kVector=8` as the FP4 fast-path default and
  evaluate the wide `kVector=32` single-transaction FP4 variant (§6).
- **dot2 inner-loop form:** the FP8 baseline keeps the **serialized `s_nop 2`** dot2
  (`dot2_f32_bf16(..., serialize=True)`). The **s_nop-free + independent-accumulator +
  single-drain** ILP scheme (reference `_dot2_batched` + one `rocdl.s_nop(2)`) is
  introduced **with the MXFP4 work** (G7) and then A/B-tested back onto FP8.
- **`cvt_scalef32_pk_bf16_fp8/fp4` via the ROCDL op** (not inline asm) — the WIP already
  does this for fp8; use the analogous `cvt_scalef32_pk_bf16_fp4` ROCDL op for FP4.
- **Exponent-only (E8M0) scale semantics of the convert:** the convert applies only the
  **exponent** of its f32 scale operand. So for arbitrary PerTensor/PerToken/Block2D FP8
  scales, pass `scale=1.0` to the convert and fold the real f32 scale into the accumulator
  after dot2 (what the WIP does). Only **MXFP4 e8m0 microscales** are fed through the
  convert's scale operand (a power-of-two, so exact). This is the key rule that makes the
  MXFP4 scale application (G2) correct.

---

## 3. Feasibility (verified / to verify)

| Item | Status | Note |
|---|---|---|
| FP8→BF16 convert (ROCDL op) | ✅ shipped | `cvt_scalef32_pk_bf16_fp8` in WIP. |
| FP4→BF16 convert (ROCDL op) | ⏳ verify | `cvt_scalef32_pk_bf16_fp4(src, scale, sel_index)` — confirm the 4-`sel` op form and e8m0 scale path on gfx950 (Phase A). |
| `v_dot2_f32_bf16` ILP (no s_nop, 1 drain) | ⏳ verify | reference proves the pattern; re-validate exact vs torch in the WIP surface. |
| i64 offset addressing in FlyDSL | ⏳ verify | reference uses `create_buffer_resource_from_addr(..., num_records_bytes=...)` with i64 base per row; confirm the WIP `buffer_ops` path (it accepts i64 base + i32 in-row offset). |
| e8m0 → f32 decode (`bitcast(shli(byte,23))`) | ⏳ verify | reference pattern; validate vs `aiter.utility.fp4_utils`. |

---

## 4. Phased plan & status

Status legend: [ ] todo · [~] in progress · [x] done

### Phase A — i64-safe addressing (G1)  [ ]  ← correctness blocker, do first
- [ ] Reproduce the overflow: run the op_test with a **real expert count** (E=256,
      HIDDEN=7168, INTER=2048) and confirm FAIL / garbage (E=8 masks it today).
- [ ] Convert the weight/activation offset math to **i64**, matching the reference:
      either (a) per-row i64 base resources (`create_buffer_resource_from_addr(base_i64 +
      row_byte_off_i64, num_records_bytes=row_nb)`) with in-row i32 offsets, or (b) keep the
      whole-tensor resource but compute the element/byte offset in i64.
      Root cause: `w_row * hidden` is i32; DeepSeek `(255*2048+2047)*7168 ≈ 3.76e9 > INT32_MAX`.
- [ ] Add the **E=256 / E=512 correctness cases** to the op_test so this can't regress
      (these are the first real ticket-shape tests; see the coverage matrix §8.2).
- **Where:** both `build_gate_up_fp8_module` and `build_down_reduce_fp8_module`
  (`x_word0`/`w_word0`/`a_word0`/`w_row` and the `_ptr_rsrc` calls).

### Phase B — MXFP4 / FP4 (G2)  [ ]  ← ticket #1 win
- [ ] **down FP4** first (the ticket's shipped best; beats FP8 down at B≥2): raw packed
      128-bit FP4 load → `cvt_scalef32_pk_bf16_fp4` (4 `sel` per i32) → dot2. **e8m0 per-block
      scale** (`block_k=32` covers the lane's 8-elt chunk) applied after the dot; router_wt
      folded per expert. Reuse the H2 two-outputs/wave structure.
- [ ] **gate_up FP4** (apply the down recipe; gate on accuracy).
- [ ] Adopt the **s_nop-free 4-accumulator + single-drain** dot2 here (G7) — it's the
      natural home for the ILP scheme (plan §2).
- [ ] Scale layout: MXFP4 uses **Block2D<1,32> e8m0**; keep the WIP's existing exact-f32
      fold for FP8 PerTensor/PerToken/Block2D.
- [ ] Correctness vs torch (MXFP4 dequant via `aiter.utility.fp4_utils`); perf A/B FP4-vs-FP8
      down at B∈{1,2,4,8} (expect the ticket's 1.2–1.5× at B≥2, neutral at B=1).
- **Where:** new `build_down_reduce_fp4_module` / `build_gate_up_fp4_module` (or a `w_dtype`
  switch inside the existing builders); entry-point `w_dtype` arg.

### Phase C — BF16 weights + gfx942 fallback (G3, G4)  [ ]
- [ ] `w_dtype="bf16"` path (BF16×BF16 dot2) as a scaffold + non-fp8 correctness oracle.
- [ ] `use_dot2=False` scalar-f32 path (bitshift widen + FMA) for gfx942 portability;
      auto-select by arch (`get_gfx`), mirroring the reference's `_is_gfx950`.
- [ ] Extend the op_test arch guard to exercise the scalar path where available.

### Phase D — Occupancy levers: split-K + LDS (G5, G6)  [ ]
- [ ] **Split-K** (`k_batch`) on down (split INTER) and gate_up (split HIDDEN), triggered
      only when `grid * k_batch <= CuCount` (under-occupied: Qwen short-INTER, low B).
      Atomic-add epilogue into a **zeroed** buffer, with **zero-init folded** into the
      gate_up epilogue / a prologue (the vLLM `blockscale_splitk_zero_init` trick) so split-K
      is free. This is where the **FP32 atomic output** variant lives — keep the BF16
      direct-store as the default non-split path.
- [ ] **LDS `n_waves`** cooperative activation staging for down (and a real gate_up
      implementation, which the reference left as dead params). Guard: `inter %
      (n_waves*WAVE_SIZE*2) == 0`.
- [ ] Benchmark on small-grid Qwen (INTER=512/256/128, B=1) where these should pay.

### Phase E — Perf scheduling: ILP dot2 + prefetch (G7, G8)  [ ]
- [ ] Land the **s_nop-free independent-accumulator dot2 + single drain** as a selectable
      inner-loop form; A/B vs the serialized `s_nop 2` baseline on FP8 (methodology §2).
- [ ] **Software-pipelined weight prefetch** (issue next K-step loads while computing the
      current step via `scf.for` iter-args carrying loaded VGPRs) — evaluate for B=1 FP4
      down (MLP-bound) and the BF16 dot2 path; the reference found it *slower* for FP8, so
      gate it behind a variant flag and measure.

### Phase F — Validation + integration (G9, G10)  [ ]
- [ ] **CK-Tile side-by-side:** build the CK bench (`tickets/667/harness/build_ck_bench.sh`,
      `ck_bench_warp_decode.cpp`), run both, join via `compare_bart.py`, and record a
      FlyDSL/CK ratio table for DeepSeek-V3 / MiniMax / Qwen3Next at B∈{1,2,4,8}. This is the
      original plan's last open Phase-4 item.
- [ ] **Register** `flydsl_warp_decode_gate_up` / `flydsl_warp_decode_down_reduce` in
      `aiter/ops/flydsl/__init__.py` (behind `is_flydsl_available()`), add to `__all__`.
- [ ] Extend the op_test perf sweep to **B∈{1,2,4,8,32}** across MiniMax + Qwen3Next-TP1 and
      all shipped dtypes, closing the coverage matrix (§8.2); feed the same shapes to CK.

### Follow-on (out of scope for this convergence)
- [ ] FP8/MXFP8 **activation** input (fuse input-side BF16→FP8 quant into gate_up).
- [ ] Re-test XCD swizzle on small-grid Qwen after a cross-wave reuse tiling lands.
- [ ] K3-report techniques: lane-teams over disjoint expert subsets; offline weight
      permutation to cut runtime dequant (needs a versioned prepack layout contract).

---

## 5. Ordering & rationale

A (blocker) → B (biggest win) → C (portability/oracle) → D (occupancy) → E (scheduling)
→ F (validation). **Phase A gates everything**: without i64 offsets the kernels are wrong
at the ticket's real expert counts, and any perf number on E=8 is not representative of the
production grid. B is sequenced before C/D because MXFP4 is the ticket's #1 item and it's the
natural place to introduce the ILP dot2 (E/G7) once. Keep every phase behind the combined
op_test correctness gate before its perf A/B, and behind the **coverage gate** (§8.2): a
phase does not close until its coverage-matrix rows are ✅.

## 6. Tuning knobs (for later sweeps)

`kVector` 8/16/32 (16 = 128-bit FP8; 8 = FP4 fast path; 32 = wide FP4) · `kHPerWarp` (down)
1/2 (**2 best at B≥2**) · `kUseDot2` vs scalar · `kNPerWarp` (gate_up) 1/2 · `n_waves`
(LDS staging) · `k_batch` (split-K) · ILP-dot2 vs serialized `s_nop 2` · prefetch on/off ·
`kLanesPerOutput` (short-INTER subgroup / K3 lane-teams).

## 7. Design notes (carried from `SILOTIGER-667-plan.md` §7)

The reference math, tensor/stride/karg layout, gate_up and down mappings (incl. H2), the
primitive table, the correctness-harness fills/tolerances, and the divisibility constraints
in `SILOTIGER-667-plan.md` §7.1–§7.7 remain the source of truth and are not duplicated here.
**Additions for this plan:**

- **i64 addressing (Phase A):** weight-row linear index `w_row = e*INTER + neuron_j`
  (gate_up) / `e*HIDDEN + out_j` (down) must be widened to i64 *before* multiplying by the
  contraction dim; DeepSeek E=256 overflows i32.
- **MXFP4 (Phase B):** 1 i32 = 8 FP4 (E2M1) per lane per K-step; `cvt_scalef32_pk_bf16_fp4`
  `sel∈{0,1,2,3}` extracts the 4 BF16 pairs; e8m0 scale byte → f32 via
  `bitcast(shli(extui(byte), 23))`; `block_k=32 > kVector=8` ⇒ one scale per lane per K-step.
- **Split-K epilogue (Phase D):** atomic-add into a zeroed FP32 `y`; fold the zero-init into
  gate_up's epilogue/prologue; the deterministic scratch-reduce variant is the
  batch-invariant option.

## 8. Testing conventions & reuse

Same as `SILOTIGER-667-plan.md` §8 (CONTRIBUTE.md standalone scripts; `aiter.test_common`
`checkAllclose`/`run_perftest`; reuse `torch_moe_stage1/2`, `fused_topk`,
`aiter.utility.fp4_utils` for MXFP4 dequant/e8m0, and quant helpers; gfx950/FlyDSL skip
guard; black + ruff). Extend the existing `test_flydsl_warp_decode_moe.py` — do **not**
fork a second test file. New required cases: **real-E correctness** (E=256/512, Phase A),
**FP4** correctness + perf (Phase B), **scalar/gfx942** where available (Phase C).

### 8.1 Current coverage vs the ticket (as of 2026-08-10)

The WIP tests cover scale layouts well on **tiny synthetic shapes**, but **not one real
ticket configuration** is validated. The perf sweep borrows DeepSeek-V3's H/I/TOPK dims but
pins **E=8** (not E=256) — which is exactly why the G1 overflow is invisible and routing is
degenerate (E=8/TOPK=8 ⇒ every expert active).

- **Correctness (pytest):** gate_up + down, B∈{1,2}, H/I ≤ 1024/128, **E≤8**, TOPK≤2,
  {pertensor, pertoken, block2d}. FP8 weights × BF16 act only.
- **Perf sweep:** DeepSeek-*dimensioned* (H7168/I2048/TOPK8) + (4096/1024), **B∈{1,4}**,
  **E=8**, FP8 only.
- **Absent:** real E (256/512); MiniMax; Qwen3Next (any TP); B∈{2,8,32}; MXFP4; FP8 act.

### 8.2 Shape / batch / scale / dtype coverage matrix (target + status)

Legend: ✅ covered · ⏳ planned in the named phase · ⛔ unsupported by the kernel (reason).
Every phase's correctness cases **must extend this matrix**, and **no phase closes until its
rows are ✅** (the *coverage gate*). This is the single source of truth for "what must pass";
the phase bullets reference it rather than re-listing shapes.

| Model | H | I | TOPK | E | Runnable? | Status / owning phase |
|---|---|---|---|---|---|---|
| DeepSeek-V3 | 7168 | 2048 | 8 | **256** | ✅ (kv16) | ⏳ **A** (real-E correctness = overflow repro) |
| MiniMax | 3072 | 1536 | 8 | **256** | ✅ (kv16/kv8) | ⏳ **B/F** (add correctness + perf rows) |
| Qwen3Next TP1 | 2048 | 512 | 10 | **512** | ✅ (kv8) | ⏳ **B/F** (add correctness + perf rows) |
| Qwen3Next TP2 | 2048 | 256 | 10 | **512** | ⛔ | `INTER%512≠0`; needs short-INTER `kLanesPerOutput` path (see §6) |
| Qwen3Next TP4 | 2048 | 128 | 10 | **512** | ⛔ | `INTER%512≠0`; same short-INTER gap |

| Axis | Target | Status / owning phase |
|---|---|---|
| **Batch B** | 1, 2, 4, 8, 32 | ✅ 1,4 · ⏳ **F** adds 2, 8, 32 |
| **Scale layout** | pertensor, pertoken, block2d | ✅ all three (both stages) |
| **Weight dtype** | FP8, MXFP4, BF16 | ✅ FP8 · ⏳ MXFP4 **B** · ⏳ BF16 **C** |
| **Activation** | BF16, FP8 | ✅ BF16 · ⏳ FP8 (follow-on) |
| **Arch** | gfx950, gfx942 | ✅ gfx950 · ⏳ gfx942 scalar **C** |

**Explicit test deliverables per phase (extends the matrix above):**
- **Phase A:** add DeepSeek-V3 **E=256** and Qwen3Next-TP1 **E=512** correctness cases (the
  overflow repro *and* the first real-shape tests); keep the existing E=8 cases.
- **Phase B:** add **MXFP4** correctness + perf for DeepSeek-V3 / MiniMax / Qwen3Next-TP1.
- **Phase C:** add a **BF16-weight** oracle case and a **gfx942 scalar** case (arch-guarded).
- **Phase D:** add the small-grid **Qwen3Next-TP1 B=1** split-K/LDS rows.
- **Phase F:** widen the perf sweep to **B∈{1,2,4,8,32}** and add MiniMax + Qwen3Next-TP1 rows
  across all shipped dtypes; feed the same shapes to the CK side-by-side (§4 Phase F).
- **Deferred (kernel-support, not just test):** short-INTER **Qwen TP2/TP4** (I=256/128) —
  add the `kLanesPerOutput` subgroup path first, then the coverage rows.

## 9. Open questions / risks

- [blocker] **i32 offset overflow at real E** — assumed present from static analysis
  (single whole-tensor i32 element offset; `w_row*hidden` overflows for E≥~73 at
  H7168/I2048). Confirm empirically in Phase A before anything else.
- [open] Does the WIP `buffer_ops.buffer_load` element→byte multiply stay i32 internally
  even with an i64 base resource? If so, per-row i64 base resources (in-row offsets small)
  are the safer fix than a whole-tensor i64 element offset.
- [open] FP4 gate_up **accuracy** (ticket gates FP4 gate_up on accuracy; MXFP4 mantissa is
  tiny). Measure cos-sim vs BF16-weight reference before claiming the win.
- [open] Split-K / LDS pay only in the **occupancy-bound** regime (small-grid Qwen, low B);
  they may be neutral/regressing on large-grid DeepSeek (already near the HBM wall) — treat
  as regime-limited and isolate the small-grid case.
- [resolved, keep] Convert scale is **exponent-only (E8M0)** — see §2.

## 10. Changelog

- _init (2026-08-10)_ — plan created from the WIP-vs-reference comparison. Recorded the 10
  gaps (G1–G10) + the divergences where the WIP is ahead (scale layouts, BF16 output,
  `kh` generality, testing rigor, `fx.*` surface). Locked §2 (test env, kernel location,
  dot2 primitive, benchmarking methodology) carried from `SILOTIGER-667-plan.md`; also
  retained in §2 the kVector default, serialized-vs-ILP dot2 policy, ROCDL-op converts, and
  the exponent-only E8M0 scale rule. Phased A→F with Phase A (i64 addressing) as the
  correctness blocker to do first.
- _test coverage (2026-08-10)_ — added §8.1 (current coverage vs ticket) and §8.2 (the
  shape/batch/scale/dtype coverage matrix + per-phase test deliverables + the "coverage
  gate"). Recorded that no real ticket config is validated today (perf sweep pins E=8), that
  MiniMax and Qwen3Next are absent, and that Qwen TP2/TP4 short-INTER is a kernel-support gap
  (not just a missing test). Wired Phases A/F and §5 to reference the matrix.
