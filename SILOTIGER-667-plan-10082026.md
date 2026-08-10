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
| G1 | **i64-safe weight offsets** (only >8 GB tensors) | ✅ per-row i64 base | ✅ ticket shapes / ❌ >8 GB (K3) | **not a ticket blocker** (empirically verified 2026-08-10; K3-only) |
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
| FP4→BF16 convert (ROCDL op) | ✅ measured | `cvt_scalef32_pk_bf16_fp4(res, src, scale, src_sel_index)` — the 4-`sel` form (`sel∈{0,1,2,3}` selects one of the 4 bf16 pairs in an 8-FP4 i32). **Exact vs the MXFP4 codebook** on gfx950 (`max_abs_err=0`, cos=1.0, packing matches `fp4_utils` nibble order); `sel` must be a compile-time int (`I32Attr`) — unroll, don't loop. Repro: `/tmp/repro_fp4_primitive.py`. |
| `v_dot2_f32_bf16` ILP (no s_nop, 1 drain) | ⏳ verify | reference proves the pattern; re-validate exact vs torch in the WIP surface. |
| i64 offset addressing in FlyDSL | ✅ measured | WIP `fx.*` offsets are 64-bit-safe up to the **i32 dword-index limit** (`buffer_load` truncates offset to i32): correct for all ticket shapes (≤3.74 GB tensors), breaks only >8 GB (K3). Per-row i64 base is the K3 fix. |
| e8m0 → f32 decode (`bitcast(shli(byte,23))`) | ✅ measured | In-kernel `shl(byte,23)+bitcast` is **bit-exact vs `fp4_utils.e8m0_to_f32`** on the normal exponent range (bytes 1..254) on gfx950. The `0`/`0xFF` specials are never produced for real MXFP4 weights (out of scope). Same repro. |

---

## 4. Phased plan & status

Status legend: [ ] todo · [~] in progress · [x] done

### Phase A — real-E regression tests + K3-scale addressing (G1)  [~]  ← NOT a ticket blocker
**Empirically resolved (2026-08-10):** the WIP is **correct at real ticket expert counts**.
Verified on gfx950 via `/tmp/repro_g1*.py`: gate_up **and** down at DeepSeek-V3
(E=256, H7168/I2048), *with the max-offset expert 255 forced*, give **cos = 1.000000**
(gate_up per-expert all 1.0; down cos 0.999999). The original "overflows at E≥73" premise
was **wrong about the mechanism**: FlyDSL `fx.*` offset arithmetic is wider than i32 (the
reference needed i64 casts only because it works in explicit-i32 `arith`). The real limit is
the **i32 DWORD index** (`byte_offset/4`) that `buffer_load` truncates to i32 (`_to_i32_offset`),
which wraps only when a **weight tensor exceeds ~8 GB** (`byte_offset > 2^33`). Among ticket
shapes only DeepSeek passes 2 GB (3.74 GB) and its dword index (9.35e8) is still < 2^31 → safe.
**Kimi-K3 does break:** E=896/H3584/I3072 (dword index 2.46e9 > 2^31, 9.85 GB tensor) gives
**cos = 0.019** (`/tmp/repro_k3_addr.py`). So:
- [x] **Add real-E regression cases** (DeepSeek-V3 E=256; Qwen-TP1 E=512) to the op_test to
      lock in the verified-good behavior and give the first real ticket-shape tests (§8.2).
      *(This is the only Phase-A item on the ticket's critical path.)*
      `test_gate_up_fp8_real_expert_count` + `test_down_reduce_fp8_real_expert_count` are now
      **parametrized** over `REAL_E_CASES = {deepseek_v3_e256, qwen3next_tp1_e512}` (max-offset
      expert forced, compact per-(b,k) ref, HBM-guarded); all **4 pass** on gfx950 (5.99 s).
- [ ] **K3-scale addressing fix (deferred to the Kimi-K3 follow-on, not the ticket):** for
      weight tensors > ~8 GB, switch to **per-row i64 base resources**
      (`create_buffer_resource_from_addr(base_i64 + row_byte_off_i64, num_records_bytes=row_nb)`,
      in-row i32 offset small) — this fixes *both* the dword-index truncation *and* the 4 GB
      `num_records` clamp (`0xFFFFFFFF`) that would otherwise OOB-zero reads past 4 GB. The
      whole-tensor + i64-offset variant is insufficient (both the clamp and `buffer_load`'s
      i32 offset bite), so per-row is the required form at K3 scale.
- **Where:** `build_gate_up_fp8_module` / `build_down_reduce_fp8_module` (`_ptr_rsrc` +
  `w_word0`/`a_word0`); only the K3-scale item touches kernel addressing.

### Phase B — MXFP4 / FP4 (G2)  [ ]  ← ticket #1 win
- [x] **Primitives de-risked** (gfx950, `/tmp/repro_fp4_primitive.py`): the 4-`sel`
      `cvt_scalef32_pk_bf16_fp4` convert is **exact vs the MXFP4 codebook** and the e8m0
      `shl 23 + bitcast` decode is **bit-exact vs `fp4_utils.e8m0_to_f32`** (normal range).
      Gotcha: `src_sel_index` is an `I32Attr` → must be a compile-time constant (unroll the 4
      `sel` calls; a `for sel in range(4)` becomes an `scf.for` and fails with `bad_cast`).
- [~] **down FP4** first (the ticket's shipped best; beats FP8 down at B≥2): raw packed
      FP4 load → `cvt_scalef32_pk_bf16_fp4` (4 `sel` per i32) → dot2. **e8m0 per-block
      scale** (`block_k=32` covers the lane's `kVector`-elt chunk) applied **in the convert**
      (uniform over the chunk ⇒ ≡ scaling the partial dot); router_wt folded per expert.
      Reuses the H2 two-outputs/wave structure. **Builder landed** as a *separate*
      `build_down_reduce_fp4_module` (kVector=8 default: 1 i32 = 8 FP4 = one weight
      dword/lane/iter, `n_wwords = kVector/8`, `w_word = ipair//4`, `sel = ipair%4`; E8M0 byte
      loaded via `dtype=T.i8()` → `e8m0_byte_to_f32`). **Correctness PASS** on gfx950
      (`/tmp/repro_down_fp4.py`, cos 0.999999 at H1/H2, incl. max-offset expert).
      **Entry point wired**: `flydsl_warp_decode_down_reduce_fp4` (separate fn, not a
      `w_dtype` switch — cleaner given the FP8 scale-mode branches) takes `w_down` uint8
      `[E, HIDDEN, INTER//2]` + `w_down_scale` uint8 E8M0 `[(E*HIDDEN)//BN, INTER//BK]`,
      `scale_block=(1,32)` default, `kvector=8` via `pick_kvector_fp4`, cached via
      `_get_down_reduce_fp4`. End-to-end **PASS** (`/tmp/repro_down_fp4_entry.py`, cos 0.999998).
      **op_test landed**: `test_down_reduce_fp4` (parametrized `DOWN_FP4_CASES`, self-contained
      FP4-codebook + E8M0 gen, dequant reference, max-offset expert forced) — 2 cases pass;
      full file green (20 passed). Still TODO: A/B perf sweep (FP4 vs FP8 down at B∈{1,2,4,8}).
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
- [ ] **Plain FP8 activation** input (per-tensor / per-token): fuse input-side BF16→FP8 quant
      into gate_up; convert `x` via the same `cvt_scalef32_pk_bf16_fp8` op, fold the real
      scale after dot2 (exponent-only convert, §2).
- [ ] **MXFP8 (block-scaled) activation** input — *distinct from plain FP8*: microscaled FP8
      with per-block (e8m0) scales on the activation, mirroring the MXFP4 weight path but on
      `x`. Needs an activation-side per-block scale convert + fold; block granularity per the
      model's contract. Required for a faithful **Kimi-K3** op (MXFP8 activations, §8.2).
- [ ] **Parameterized gate_up activation — SiTU-GLU** (not just SiLU). Both kernels currently
      **hardcode `silu(gate)·up`**; make the epilogue activation selectable and add a
      **SiTU-GLU** path. Required for a faithful **Kimi-K3** op (K3's `hidden_act == "situ"`,
      SiTU-GLU per the public spec); SiLU stays the default for DeepSeek/MiniMax/Qwen.
- [ ] Re-test XCD swizzle on small-grid Qwen after a cross-wave reuse tiling lands.
- [ ] K3-report techniques: lane-teams over disjoint expert subsets; offline weight
      permutation to cut runtime dequant (needs a versioned prepack layout contract).

---

## 5. Ordering & rationale

**Revised after the 2026-08-10 empirical check.** Phase A is **no longer a blocker** — the
WIP is already correct at real ticket expert counts (verified), so Phase A shrinks to *adding
regression tests*. **Phase B (MXFP4) is now the first substantive work** and the ticket's #1
value item; it's also the natural place to introduce the ILP dot2 (E/G7) once. Order:
**B → (A regression tests, cheap, fold in alongside) → C (portability/oracle) → D (occupancy)
→ E (scheduling) → F (validation).** The K3-scale addressing fix (the surviving part of G1)
rides with the Kimi-K3 follow-on, not the ticket. Keep every phase behind the combined
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
| DeepSeek-V3 | 7168 | 2048 | 8 | **256** | ✅ (kv16) | ✅ **A** real-E correctness (both stages pass); ⏳ **B/F** perf |
| MiniMax | 3072 | 1536 | 8 | **256** | ✅ (kv16/kv8) | ⏳ **B/F** (add correctness + perf rows) |
| Qwen3Next TP1 | 2048 | 512 | 10 | **512** | ✅ (kv8) | ✅ **A** real-E correctness (both stages pass); ⏳ **B/F** perf |
| Qwen3Next TP2 | 2048 | 256 | 10 | **512** | ⛔ | `INTER%512≠0`; needs short-INTER `kLanesPerOutput` path (see §6) |
| Qwen3Next TP4 | 2048 | 128 | 10 | **512** | ⛔ | `INTER%512≠0`; same short-INTER gap |
| Kimi-K3 (routed, latent MoE) | **3584** | 3072 | **16** | **896** | ✅ (gate_up kv8 / down kv16) | ⏳ **follow-on** (needs MXFP4 + FP8/MXFP8 act; E=896 max-stresses G1) |

**Kimi-K3 mapping note (public spec, arXiv 2607.24653 / Moonshot model card).** K3 is a
**latent MoE**: the 7168 hidden is projected to a **3584-wide latent** and the routed
experts run *on the latent*, so the warp-decode routed-expert GEMM uses **HIDDEN(contract)
= 3584** (not 7168), INTER = 3072, E = 896, TOPK = 16. The dims are divisible for the
current kernels (gate_up `3584 % 512 == 0` ⇒ kv8; down `3072 % 1024 == 0` ⇒ kv16, H2 ok),
but K3 is **follow-on**: it needs MXFP4 weights (Phase B) + MXFP8 activations (follow-on),
and — unlike every ticket shape — it **actually hits the addressing limit**: its 9.85 GB
gate_up weight tensor gives a dword index 2.46e9 > 2^31, and the WIP is **measured broken**
here (cos 0.019, `/tmp/repro_k3_addr.py`, 2026-08-10). So K3 additionally needs the per-row
i64 base addressing fix (Phase A K3-scale item). The 2 **shared** experts run dense on the full 7168
and are out of scope for the routed warp-decode path. Suggested by a colleague as a
bench/tune target; verify against the shipped weights before publishing numbers.

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
- **Follow-on:** **Kimi-K3** (latent-MoE routed expert: H_contract=3584, I=3072, E=896,
  TOPK=16, MXFP4+MXFP8) — add once Phase B (MXFP4) + FP8/MXFP8 activation land; doubles as
  the harshest G1 overflow stress case.

## 9. Open questions / risks

- [resolved, 2026-08-10] **i32 offset overflow** — the original "overflows at real E"
  premise was **refuted empirically**. WIP `fx.*` offset math is 64-bit; the true limit is
  `buffer_load`'s **i32 dword-index truncation**, which bites only for weight tensors >~8 GB.
  All ticket shapes are safe (DeepSeek E=256 verified cos 1.0, max-offset expert forced);
  **Kimi-K3 breaks** (cos 0.019, 9.85 GB / dword index 2.46e9 > 2^31). Fix = per-row i64 base
  resources, deferred to the K3 follow-on. Repros: `/tmp/repro_g1.py`, `/tmp/repro_g1_down.py`,
  `/tmp/repro_k3_addr.py`.
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
- _Kimi-K3 shape (2026-08-10)_ — on a colleague's suggestion, confirmed K3's public dims
  (arXiv 2607.24653 / Moonshot model card): latent MoE, routed-expert contraction **3584**
  (not 7168), INTER 3072, E 896, TOPK 16, MXFP4+MXFP8. Added it to §8.2 as a **follow-on**
  coverage row with a mapping note; flagged E=896 as the harshest G1 overflow stress case.
  No `kimi`/`k3` shape preset existed in-repo; this is the first record of the dims.
- _K3 faithful-op gaps (2026-08-10)_ — folded in three items surfaced by the K3 analysis:
  (1) Phase A now notes the **`num_records` 4 GB clamp** ⇒ >4 GB tensors (e.g. K3) **mandate**
  per-row i64 base addressing (option (a)), not a whole-tensor i64 offset; (2) split the
  follow-on activation item into **plain FP8** vs **MXFP8 (block-scaled)**; (3) added a
  follow-on to **parameterize the gate_up activation and add SiTU-GLU** (kernels hardcode
  SiLU today) for a faithful K3 op. Bench/tune on K3 shapes still only needs Phases A+B; these
  three are for correctness-faithful K3.
- _G1 empirically resolved (2026-08-10)_ — ran the WIP gate_up + down at real expert counts
  on gfx950. **DeepSeek-V3 E=256 is correct** (cos 1.0 with max-offset expert 255 forced) —
  the "overflows at real E" premise was **wrong**: `fx.*` offsets are 64-bit; the true limit
  is `buffer_load`'s i32 dword-index truncation, hit only by >8 GB tensors. All ticket shapes
  are safe; **Kimi-K3 breaks** (cos 0.019, 9.85 GB). Downgraded G1 from "correctness blocker"
  to "not a ticket blocker; K3-only"; **re-ordered §5 so Phase B (MXFP4) is the first
  substantive work**, Phase A reduced to regression tests + the deferred K3-scale per-row i64
  fix. Updated §1/§3/§4/§8.2/§9. Repros in `/tmp/repro_g1.py`, `/tmp/repro_g1_down.py`,
  `/tmp/repro_k3_addr.py`.
- _Qwen-TP1 E=512 regression folded in (2026-08-10)_ — parametrized the real-E op_tests over
  `REAL_E_CASES = {deepseek_v3_e256, qwen3next_tp1_e512}` (both stages, max-offset expert
  forced); all 4 pass on gfx950. Closes the Phase A regression-test item; §8.2 matrix updated.
- _Phase B primitives de-risked (2026-08-10)_ — validated the two FP4 primitives on gfx950
  (`/tmp/repro_fp4_primitive.py`): the 4-`sel` `cvt_scalef32_pk_bf16_fp4` convert is exact vs
  the MXFP4 codebook (packing matches `fp4_utils` nibble order) and the e8m0 `shl 23 + bitcast`
  decode is bit-exact vs `fp4_utils.e8m0_to_f32` (normal range). Recorded the `src_sel_index`
  `I32Attr` compile-time-constant gotcha. Marked the §3 feasibility rows ✅ measured and added
  a Phase B "primitives de-risked" checkbox. **Next substantive step: `build_down_reduce_fp4_module`.**
