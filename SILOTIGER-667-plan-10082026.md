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
| G1 | **i64-safe weight offsets** (only >8 GB tensors) | ✅ per-row i64 base | ✅ ticket shapes / ✅ FP8 large (K3 Tier-2, B5 2026-08-17) | **resolved** — FP8 `down`/`gate_up` now use a guarded per-expert i64 base (repro E=896 cos 0.999999; DeepSeek E=256 cos 1.0) |
| G2 | **MXFP4 / FP4 weights** (down + gate_up) | ✅ (down full; gate_up fp4) | ✅ (down + gate_up FP4 landed, tested) | **landed** — ticket #1 win (cold-HBM 1.26–1.65×, cos 1.0) |
| G3 | **BF16-weight path** (`w_dtype="bf16"`) | ✅ (dot2 + scalar) | ❌ FP8-only | scaffold / gfx942 |
| G4 | **gfx942 / scalar-f32 fallback** (`use_dot2=False`) | ✅ auto-arch | ❌ gfx950-only | portability |
| G5 | **Split-K** (`k_batch`) + zero-init fusion | ✅ (gate_up 2-phase, down) | ⊘ evaluated — no-win, gated-off, not shipped (attic branch) | ticket lever (closed) |
| G6 | **LDS cooperative caching** (`n_waves`) | ✅ (down; gate_up dead) | ⊘ evaluated — no-win by feasibility (latency-floor bound), not built | ticket lever (closed) |
| G7 | **s_nop-free / independent-accumulator dot2 (ILP)** | ✅ (fp8/fp4) | ❌ serialized `s_nop 2` | perf (deferred) |
| G8 | **Software-pipelined weight prefetch** | ✅ (bf16 dot2; fp4 `down` flag) | ❌ default (knob) | **perf: ~5% B=1 fp4**, neutral B≥4 |
| G9 | **CK-Tile cross-benchmark harness** | ✅ (`ck_bench_*.cpp` + compare) | ✅ (`compare.py` + `g9_compare` of-record) | **done** — 0 n/a, all 15 cells (2026-08-18) |
| G10 | **Package public API registration** (`__init__.py`) | ✅ | ✅ (2026-08-11; all 6 entry points) | integration |

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
- **GPU selection:** run on **GPU 6** — prefix the command (or export first) with
  `HIP_VISIBLE_DEVICES=6` so the process sees a single device indexed as `cuda:0`, e.g.
  `HIP_VISIBLE_DEVICES=6 ./flydsl_venv/bin/python -m pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py`.
  (Isolating one device also keeps the cold-HBM-read perf numbers clean.)
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

### Phase A — real-E regression tests + K3-scale addressing (G1)  [x]  ← NOT a ticket blocker
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
- [x] **K3-scale addressing fix — landed as B5 (2026-08-17), K3 Tier-2.** The FP8
      `down`/`gate_up` builders now fold a **per-expert i64 base** into the buffer descriptor
      (`_ptr_rsrc_off` = `create_buffer_resource_from_addr(ptrtoint(ptr) + e*H*I)`) and address
      weights by the i32-safe in-expert dword offset. Guarded by a build-time `num_experts` so
      it only fires at `E*I*H >= 2^31` (i32-safe path + all FP4 shapes untouched). This is the
      *per-expert* form (coarser than the *per-row* sketch, but sufficient: an expert spans
      `H*I <= ~1.5e7 B`, well inside i32, and the default `num_records` clamp is never hit since
      each descriptor only reads its own expert). Empirically fixes the i32 **byte-offset** wrap
      (`w_row*INTER*4` — the real killer at 2^31 B, distinct from the 8 GB dword-index limit the
      2026-08-10 note assumed): DeepSeek E=256 FP8 `down`+`gate_up` cos 0.02→1.0, and the K3
      E=896 repro (9.85 GB tensor) cos 0.02→0.999999. Kimi-K3 FP4 (packed, `E*H*I < 2^32`) still
      rides the Tier-1 restructure.
- **Where:** `build_gate_up_fp8_module` / `build_down_reduce_fp8_module` (`_ptr_rsrc` +
  `w_word0`/`a_word0`); only the K3-scale item touches kernel addressing.

### Phase B — MXFP4 / FP4 (G2)  [x]  ← ticket #1 win
- [x] **Primitives de-risked** (gfx950, `/tmp/repro_fp4_primitive.py`): the 4-`sel`
      `cvt_scalef32_pk_bf16_fp4` convert is **exact vs the MXFP4 codebook** and the e8m0
      `shl 23 + bitcast` decode is **bit-exact vs `fp4_utils.e8m0_to_f32`** (normal range).
      Gotcha: `src_sel_index` is an `I32Attr` → must be a compile-time constant (unroll the 4
      `sel` calls; a `for sel in range(4)` becomes an `scf.for` and fails with `bad_cast`).
- [x] **down FP4** first (the ticket's shipped best; beats FP8 down at B≥2): raw packed
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
      full file green (20 passed).
- [x] **A/B perf sweep landed** (`bench_down_fp4` + `DOWN_FP4_PERF_SHAPES`, second markdown
      table; GPU 6, gfx950, device timing, 50 iters). **Finding: the first-cut FP4 `down` is
      NOT yet faster** — at DeepSeek E=8 I2048/H7168 it is **1.58× slower at B=1** (25.2 vs
      15.97 µs) and ~parity at B=4 (49.5 vs 47.2 µs); B2=27.3 µs, B8=96.4 µs. FP4 TB/s is
      2.5–5.2 vs FP8 7–10, i.e. the kernel is **convert/latency-bound, not BW-bound**, so
      halving the weight bytes buys nothing here. Root causes: (a) at **E=8** the weights fit
      in MALL (FP4 58 MB / FP8 117 MB) so reads aren't cold-HBM; (b) FP4 default **kVector=8**
      ⇒ 2× the iterations (⇒ 2× activation + per-`(h,iter)` sub-dword E8M0 byte loads + loop
      overhead) vs FP8 kVector=16; (c) the **serialized `s_nop 2`** dot2 (G7 ILP not yet
      applied) + no prefetch (G8). **FP4-win prerequisites (deferred to Phase D/E):** measure
      at **real E=256 cold HBM** (where 0.5 B/elt dominates), raise FP4 to **kVector=16/32**
      (§6, needs INTER%1024/%2048), hoist the E8M0 load, and land the **s_nop-free
      4-accumulator single-drain** dot2 (G7) + prefetch (G8). Correct + wired + tested now;
      the perf uplift is an optimization task, not a correctness gap.
- [x] **gate_up FP4** (applied the down recipe). **Builder** `build_gate_up_fp4_module`
      (separate; kVector=8 default, `n_wwords=kVector/8`, `w_word=ipair//4`, `sel=ipair%4`;
      gate/up have *separate* E8M0 scale tensors, each applied **in the convert**; single
      gate/up dot accumulator across iterations then one reduce each; `silu(gate)*up`
      epilogue unchanged). **Entry point** `flydsl_warp_decode_gate_up_fp4` (`w_gate`/`w_up`
      uint8 `[E, INTER, HIDDEN//2]` + two E8M0 scale tensors, `scale_block=(1,32)`,
      `kvector` via `pick_kvector_fp4(HIDDEN)`, cached via `_get_gate_up_fp4`). **op_test**
      `test_gate_up_fp4` (`GATE_UP_FP4_CASES`, dequant ref, max-offset expert) — 2 pass; full
      file **22 passed**. Accuracy: cos ≥ 0.99 holds through the SiLU on these shapes. Perf
      A/B deferred to Phase D/E with the `down` levers (kVector 16/32, G7 ILP, G8 prefetch).
- [x] **G7 s_nop-free multi-accumulator dot2 landed on FP4 `down`.** `dot2_f32_bf16_drain`
      round-robins the per-`(h,k)` pairs across `dot2_acc` (default **4**) independent f32
      accumulators; consecutive `v_dot2_f32_bf16` write different registers so the
      accumulator-RAW hazard is hidden by ILP (no `s_nop`), and only the **final write per
      accumulator** carries `s_nop 2` (the drain add reads it immediately). Restructured the
      k-loop to **collect all `(iter,pair)` contributions into one drain per `(h,k)`** (was a
      per-`(h,iter)` `dot_i` + `acc += dot_i*rw`), so the accumulators persist across the whole
      K-range and s_nop count drops from `num_iter*n_pairs` to `dot2_acc` per `(h,k)` (16→4 at
      DeepSeek I2048). Threaded `dot2_acc` through `build_down_reduce_fp4_module`,
      `_get_down_reduce_fp4`, and `flydsl_warp_decode_down_reduce_fp4` (`dot2_acc<=1` keeps the
      serialized `s_nop` chain for A/B). **Impl notes:** compile-time branch needs
      `const_expr(dot2_acc>1)` (a bare `if` lowers to `scf.if` and values don't escape); pair
      lists are built with comprehensions / `range_constexpr` index loops (a plain in-kernel
      `for` over a Python list is rewritten to a dynamic loop and fails to loop-carry).
      **A/B (GPU 6, gfx950, device timing, 100 iters, DeepSeek I2048/H7168/E8):** B1
      15.15→**13.54 µs (1.12×)**, B2 0.91×, B4 1.00×, B8 0.99×; cos **1.0000** at all B. G7
      helps exactly where decode is latency/serialization-bound (**B=1**) and is neutral at
      B≥2 (occupancy already hides the stall). Correct + wired + swept.
- [x] **G7 also wired on FP4 `gate_up`, but defaulted OFF (`dot2_acc=1`) — negative result.**
      Same `dot2_f32_bf16_drain` applied to both gate/up streams (collect all `(iter,pair)`
      pairs, drain each through `dot2_acc` accumulators); threaded through
      `build_gate_up_fp4_module` / `_get_gate_up_fp4` / `flydsl_warp_decode_gate_up_fp4`.
      **A/B (GPU 6, gfx950, 100 iters, DeepSeek-like H7168(contraction)/I2048/E8):** G7 is
      **~4% slower** (B1 0.94–0.96×, B2 0.97×, B4 0.98×, B8 ~1.0×); cos **1.0000**. Root cause:
      gate_up already runs **two interleaved dot2 streams** (gate+up) that mutually cover the
      accumulator hazard, and its B=1 grid `B*TOPK*INTER`=16384 waves (vs `down`'s
      `B*HIDDEN/kh`=3584) is **occupancy-bound not latency-bound**, so removing `s_nop` buys
      nothing while the extra accumulators/drain adds cost a little. **Decision:** default
      gate_up to the serialized path; keep the `dot2_acc` knob wired to re-test once kVector
      16/32 lengthens each stream's independent-pair count (may flip the balance).
- [x] **kVector 16/32 evaluated — NEGATIVE, kept default kVector=8.** Generalized
      `pick_kvector_fp4` (temporarily) to pick the largest that tiles the contraction and added
      a `kvector=` override to both FP4 entry points; the builders already handle kVector 16/32
      (`n_pairs=8/16`, `n_wwords=2/4`, one 64/128-bit weight load, `scale_bk=32` still a
      multiple). Correctness holds at 8/16/32 for both stages (added explicit-kVector op_test
      cases; **24 passed**). **A/B (GPU 6, gfx950, 100 iters, DeepSeek I2048/H7168/E8):**
      *down* (dot2_acc=4) µs B1 **kV8 13.96 / kV16 13.45 / kV32 14.61**, B2 20.4/21.0/24.4,
      B4 36.0/38.7/43.5, B8 70.9/74.6/85.9; *gate_up* (dot2_acc=1) B1 **kV8 20.2 / kV16 21.7**,
      B2 37.2/37.8, B4 70.0/71.5, B8 132/134. So **kVector=8 is best or tied-best everywhere**
      except a marginal ~4% at *down* B=1 (kV16); kV16/32 regress B≥2 and gate_up, kV32 is
      uniformly worse. **Root cause:** warp-decode is one-wave-per-output, so a lane's VGPR
      footprint gates occupancy; larger kVector inflates live activation/weight dwords (and
      pairs held through the G7 drain), and the lost occupancy outweighs the fewer-iterations
      saving. **Decision:** `pick_kvector_fp4` stays at **8** (reverted the auto-scale); the
      `kvector` override remains for tuning (e.g. a *down*-B=1-only deployment could pick 16).
      This kills the "raise kVector" lever from the original FP4-win hypothesis — the remaining
      lever is **real E=256 cold-HBM measurement** (where 0.5 B/elt finally dominates) + G8
      prefetch, not bigger tiles.
- [x] **Cold-HBM large-E harness landed — FP4 CONFIRMS the ticket's BW win (1.26–1.54×).**
      New `bench_down_cold` (behind `--cold`) allocates the *full* E-expert down pool once and
      **rotates a tiny router-id set over disjoint expert groups** (`_router_group_list`), so
      steady-state launches stream fresh weights from HBM instead of re-reading the TOPK experts
      warm in MALL (the artifact that made the earlier E=8 sweep misleading). Pools are built in
      packed form (`_gen_down_fp4_pool` / `_gen_down_fp8_pool`) and the correctness gate
      dequantizes only the *touched* experts (`_dequant_down_expert_fp4`) so the ~15 GB fp32
      E-pool is never materialized; router rotation drives content via a stateful closure
      (weights captured, not deep-copied → no OOM). **A/B (GPU 6, gfx950, 50 iters, DeepSeek
      I2048/H7168, E=128, cos 1.0):**
      | B | FP4 µs | FP8 µs | FP4/FP8 | speedup |
      |---|--------|--------|---------|---------|
      | 1 | 17.1 | 21.5 | 0.795 | **1.26×** |
      | 2 | 26.3 | 35.2 | 0.746 | **1.34×** |
      | 4 | 43.1 | 66.1 | 0.651 | **1.54×** |
      | 8 | 81.8 | 126.0 | 0.649 | **1.54×** |
      Complete reversal of the warm E=8 result (FP4 was 1.58× *slower* at B=1): once the pool
      (FP4 0.94 GB / FP8 1.88 GB) ≫ MALL and reads are cold, **halving the weight bytes wins,
      and the win grows with B** (more cold weight traffic per launch). FP8 still shows higher
      raw TB/s (5.5→7.5 vs FP4 3.6→6.1, FP4 pays convert overhead) but moves 2× the bytes, so
      FP4 finishes first. FP8 still shows higher raw TB/s (5.5→7.5 vs FP4 3.6→6.1) but moves 2×
      the bytes, so FP4 finishes first.
    - **K3 Tier-1 fix landed (2026-08-11) — FP4 now runs at the real E=256.** The i32 weight
      offset used to overflow because it computed `(w_row*DIM + k_base)//WPACK`, and the
      `w_row*DIM` element product wrapped 2^31 at E≈146. Restructured to the algebraic identity
      `w_row*(DIM//WPACK) + k_base//WPACK` (both terms divisible by WPACK, so bit-exact — 26/26
      op_tests still pass, incl. new E=256 correctness cases). This drops the intermediate product
      so FP4's limiting quantity becomes the hardware **byte** offset `w_row*INTER/2`, which fits
      i32 up to `E*H*I < 2^32`; E=256 (`3.76e9`, byte offset 1.88 GB) now runs. Cold E=256 FP4
      `down` (pool 1.88 GB ≫ MALL): B=1 17.5 µs / B=2 26.4 / B=4 42.9 / B=8 80.8, **cos 1.000, no
      fault** — matches the E=128 timings (per-launch reads are E-independent), so the 1.26–1.54×
      FP4-vs-FP8 ratios above hold at true E=256. **FP8 E=256 — RESOLVED (K3 Tier-2, B5, 2026-08-17).**
      Its byte offset is `w_row*INTER` (2× FP4) and wrapped i32 at 2^31; the fix folds a
      **per-expert i64 base** into the buffer descriptor (guarded to `E*I*H >= 2^31`) so the
      in-expert offset stays i32-safe. The cold harness no longer auto-skips FP8 (now gates on
      the in-expert `H*I < 2^31`); DeepSeek E=256 FP8 `down`+`gate_up` cos=1.0 and the of-record
      G9 artifact has 0 n/a cells. Remaining upside: G8 prefetch (Kimi-K3 9.85 GB addressing is
      also covered by Tier-2 — E=896 repro cos 0.999999).
    - **gate_up cold A/B (2026-08-11) — FP4 wins *bigger* than down.** `bench_gate_up_cold`
      mirrors the down harness (two-stream gate+up pools `_gen_gate_up_fp4_pool` /
      `_gen_gate_up_fp8_pool`, touched-expert dequant, same router rotation + dual i32 guard on
      `E*INTER*HIDDEN`). Paired E=128 (both legs, DeepSeek H7168/I2048, cos 1.0):
      | B | FP4 µs | FP8 µs | FP4/FP8 | speedup |
      |---|--------|--------|---------|---------|
      | 1 | 28.5 | 41.1 | 0.695 | **1.44×** |
      | 2 | 45.9 | 72.2 | 0.635 | **1.57×** |
      | 4 | 81.0 | 131.2 | 0.618 | **1.62×** |
      | 8 | 152.4 | 251.2 | 0.607 | **1.65×** |
      This **overturns the prior** that gate_up (occupancy-bound per G7) would show a *smaller*
      FP4 win: cold gate_up streams **two** weight matrices per output, so it's even more
      weight-BW-dominated than down, and halving the weight bytes helps more (1.44–1.65× vs down's
      1.26–1.54×). FP8 keeps higher raw TB/s (5.7→7.5 vs FP4 4.4→6.6, convert overhead) but moves
      2× the bytes. Real **E=256 FP4-only** confirmed fault-free (28.5/45.9/81.6/150.4 µs, cos 1.0,
      matches E=128 → ratios carry); FP8 E=256 now runs (K3 Tier-2 landed, B5), same as down.
- [x] Scale layout: MXFP4 uses **Block2D<1,32> e8m0** (applied in-convert per lane chunk);
      the WIP's existing exact-f32 fold is kept for FP8 PerTensor/PerToken/Block2D. **Done** —
      both FP4 builders decode E8M0 via `e8m0_byte_to_f32` (bit-exact vs `fp4_utils`).
- [x] Correctness vs torch (MXFP4 dequant via `aiter.utility.fp4_utils`); perf A/B FP4-vs-FP8
      down at B∈{1,2,4,8}. **Done** — op_tests (`test_{down,gate_up}_fp4`, cos ≥ 0.99) + the
      cold-HBM A/B tables above confirm the ticket's **1.26–1.54× down** (and 1.44–1.65× gate_up)
      at B≥2, ~neutral at B=1, `cos = 1.0`.
- **Where:** new `build_down_reduce_fp4_module` / `build_gate_up_fp4_module` (or a `w_dtype`
  switch inside the existing builders); entry-point `w_dtype` arg.

### Phase C — BF16 weights + gfx942 fallback (G3, G4)  [x]
- [x] `w_dtype="bf16"` path (BF16×BF16 dot2) as a scaffold + non-fp8 correctness oracle
      **(2026-08-11).** Added dedicated `build_gate_up_bf16_module` / `build_down_reduce_bf16_module`
      (mirror the FP8 builders; a bf16 weight dword *is* a dot2 operand, so **no scaled convert
      and no weight scale** — down folds only `router_wt`, gate_up keeps the silu-GLU epilogue).
      Wired `flydsl_warp_decode_gate_up_bf16` / `flydsl_warp_decode_down_reduce_bf16` (5-ptr
      kernels, no scale tensor) + `lru_cache` getters. Op_test adds `BF16_GATE_UP_CASES` /
      `BF16_DOWN_CASES` over small + real **E=256** shapes; all pass at **cos ≈ 1.0** (only bf16
      dot2 rounding, no quant error) — **32/32 suite**. This is the unquantized oracle that
      isolates dequant bugs from reduce/routing bugs, and the scaffold for the gfx942 scalar path.
- [x] **BF16-oracle cross-check (2026-08-11).** `test_{down,gate_up}_fp4_matches_bf16_oracle`
      run the FP4 kernel and the BF16 kernel on the **same logical weights** (FP4's e8m0 scales
      are powers of two + LUT values exact ⇒ the fp32 dequant is bf16-exact, so `w_deq.to(bf16)`
      is precisely what the FP4 convert produces). Both stages at small + real **E=256** agree at
      **`fp4~bf16` cos = 1.000000** (only f32 dot2 reassociation could differ). This closes the
      "isolate the FP4 convert/scale from the reduce" gap noted under the Phase-B correctness item
      — a future FP4 convert regression now fails against the oracle, not just the fp32 matmul ref.
      **36/36 suite.**
- [x] `use_dot2=False` scalar-f32 path (bf16→f32 widen + FMA) for gfx942 portability;
      auto-select by arch (`get_gfx() == "gfx950"` ⇒ dot2, else scalar), mirroring the
      reference's `_is_gfx950`. Landed on the **BF16 path** (the natural scaffold — bf16
      weights need no scaled convert, so `v_dot2_f32_bf16` is the only gfx950-specific op).
      New helpers `dot2_f32_bf16_scalar` (arch-agnostic unpack+FMA) + `dot2_or_scalar`
      (dispatch) in `kernels/warp_decode_moe.py`; `use_dot2` threaded through both bf16
      builders, cache getters, and the `flydsl_warp_decode_{gate_up,down_reduce}_bf16`
      entry points (`use_dot2=None` auto-selects; explicit override forces a path).
      **FP8/FP4 gfx942 deferred**: those also need arch-specific *converts*
      (`cvt_scalef32_pk_bf16_*` are gfx950 too), not just a dot2 swap — a separate item.
- [x] Extend the op_test arch guard to exercise the scalar path where available.
      `test_{gate_up,down_reduce}_bf16_scalar_fallback` force `use_dot2=False` on gfx950
      and assert the fallback both (a) matches the fp32 torch ref and (b) matches the dot2
      path to near bit-exactness. **All 6 forced-scalar cases: `cos_vs_ref = cos_vs_dot2 =
      1.000000`** — the gfx942 scalar math is validated on gfx950 without MI300 hardware.
      (Representative gfx942 *perf* still needs real MI300; this covers correctness only.)

### Phase D — Occupancy levers: split-K + LDS (G5, G6)  [x]  ← both closed as no-wins
> **Split-K (G5) is tracked in its own sub-plan:**
> [`SILOTIGER-667-plan-Split-K.md`](./SILOTIGER-667-plan-Split-K.md) (steps + status).
- [~] **Split-K** (`k_batch`) — **evaluated & closed as a no-win (2026-08-11); not shipped.**
      Implemented + CU-count-gated on down (split INTER) with the FP32 atomic-add-into-zeroed-buffer
      epilogue (BF16 direct-store kept as the default non-split path), but the Step 3 A/B showed **no
      decode win** (kernel-only ≤1.03×, k8 regresses; ~2× end-to-end via the v1 zeros+finalize) because
      B=1 warp-decode is **launch/memory-latency-floor bound**, not occupancy-bound. Gated off by
      default; live impl **archived on `samaario/warp-decode-moe-splitk-attic`**, not in main. gate_up
      2-phase split-K + zero-init folding **not pursued** (same physics). See
      [`SILOTIGER-667-plan-Split-K.md`](./SILOTIGER-667-plan-Split-K.md).
- [x] **LDS `n_waves` cooperative staging — closed as a predicted no-win by feasibility
      analysis (2026-08-19); not built.** LDS staging is an *occupancy* lever, so per the §9
      guidance it was gated on first isolating the small-grid case. **Measured the smallest-grid
      production shape (Qwen3Next-TP1 H2048/I512/TOPK10) at low B, warm** (gfx950, CU=256, device
      timing) to test the occupancy-headroom premise:

      | B | gate_up grid (waves/CU) | gate_up µs | down grid (waves/CU) | down µs |
      |---|-------------------------|-----------|----------------------|---------|
      | 1 | 5120 (**20×**) | 5.03 | 1024 (**4×**) | 4.13 |
      | 2 | 10240 (40×) | 6.60 | 2048 (8×) | 5.31 |
      | 4 | 20480 (80×) | 9.41 | 4096 (16×) | 7.50 |
      | 8 | 40960 (160×) | 14.50 | 8192 (32×) | 12.19 |

      **Both stages are launch/memory-latency-floor bound, not occupancy-starved** — the exact
      physics that closed split-K (G5). Even at B=1 the grid already gives 20 waves/CU (gate_up) /
      4 waves/CU (down), and time is dominated by a ~4–5 µs fixed floor (µs/B falls 5.03→1.54 and
      4.13→1.16 as B grows 1→32, i.e. work grows 32× but time only ~9×). LDS cooperative caching
      targets redundant *activation* loads, but (a) it can't lower that latency floor at the decode
      B's that matter, and (b) the **weights dominate the cold BW** (the cold A/B win came entirely
      from FP4 halving *weight* bytes; the activation chunk is tiny by comparison). So there is no
      occupancy headroom or activation-traffic upside for LDS to capture. Closed by analysis rather
      than building the kernel (avoids repeating the split-K effort for another negative). Repro:
      `/tmp/g6_feasibility.py`.
- [x] Benchmark on small-grid Qwen (INTER=512/256/128, B=1) — done (table above). INTER=256/128
      are sub-`pick_kvector` for `down` (needs INTER%512) so aren't runnable there; INTER=512 (the
      real Qwen shape) is the binding small-grid case and it's already latency-floor bound.

### Phase E — Perf scheduling: ILP dot2 + prefetch (G7, G8)  [x]
- [x] **G7 selectable s_nop-free dot2 landed on FP8 (2026-08-19).** Added `dot2_acc` to
      `build_gate_up_fp8_module` / `build_down_reduce_fp8_module` (threaded through the cache
      getters + `flydsl_warp_decode_{gate_up,down_reduce}` entry points): `dot2_acc>1` routes the
      per-lane dots through the shared `drain_or_chain` → `dot2_f32_bf16_drain` (N independent f32
      accumulators + one drain add, s_nop only on each accumulator's final write); `dot2_acc=1`
      keeps the serialized `s_nop 2` baseline. Wired in all four scale branches (pertensor/pertoken
      span the whole K-range; block2d drains per K-block). **Correctness:** new
      `test_{gate_up,down_reduce}_fp8_ilp_dot2` force `dot2_acc=4` vs the `dot2_acc=1` baseline —
      **all 10 cases cos_vs_ref = cos_vs_base = 1.000000** (24/24 FP8 suite green). **A/B** (gfx950,
      warm/compute-bound, device timing, 200 iters, DeepSeek H7168/I2048 E8 TOPK2, ratio = acc1/acc4;
      >1 ⇒ ILP faster):

      | B | gate_up (acc1/acc4) | down (acc1/acc4) |
      |---|---------------------|------------------|
      | 1 | ~1.00 (neutral)     | ~1.01 (neutral)  |
      | 2 | **0.94 (ILP −6%)**  | 1.01–1.04 (≈neutral) |
      | 4 | **0.95 (ILP −5%)**  | **1.07 (ILP +7%)** |
      | 8 | 0.99 (≈neutral)     | ~1.00 (neutral)  |

      **Verdict — defaults stay `dot2_acc=1`.** gate_up is a **consistent ~5–6% regression** at
      B=2/4 (mirrors the FP4 gate_up G7 negative: the two independent gate/up dot2 streams already
      cover the accumulator-RAW hazard, and the 4-accumulator drain's extra VGPR pressure bites once
      the grid is occupancy-bound). down showed a ~7% win **only** at B=4 in this warm/compute-bound
      A/B, neutral at B=1/8.

      **Cold-HBM cross-check (the decision-grade regime) — ILP is a no-win (2026-08-19).** Re-ran
      via the of-record cold harness (`_gen_{down,gate_up}_fp8_pool` full E=128 pool ≫ MALL + router
      rotation, `Block2D<128,128>`, device timing) so launches stream fresh weights from HBM:

      | B | down (acc1/acc4) | gate_up (acc1/acc4) |
      |---|------------------|---------------------|
      | 1 | **0.92 (ILP −8%)** | 0.97 (ILP −3%) |
      | 2 | **0.94 (ILP −6%)** | 0.98 (ILP −2%) |
      | 4 | 0.99 (neutral)   | 0.99 (≈neutral) |
      | 8 | 1.00 (neutral)   | 0.98 (ILP −2%) |

      The warm B=4 down win **does not survive**: cold decode is **memory-latency-floor bound** (same
      physics that closed split-K G5), so the s_nop removal can't help and the drain's extra VGPR
      pressure slightly *hurts* at low B (down −6 to −8%). All cos = 1.0000.

      **Verdict — defaults stay `dot2_acc=1` and the knob ships OFF.** G7-on-FP8 is a decisive
      no-win in the production cold-streaming regime (warm compute-bound gains are an artifact that
      doesn't carry). Knob retained for A/B/experimentation only; not recommended for any shipped
      config. (Not ported to the FP8-act builder — same rationale.)
- [x] **Software-pipelined weight prefetch** — evaluated on FP4 `down` (2026-08-11).
      Implemented as a `prefetch` build flag on `build_down_reduce_fp4_module` (wired through
      `flydsl_warp_decode_down_reduce_fp4(..., prefetch=)`): hoists *every* activation/weight/
      E8M0-scale load for the expert up front (all outstanding before any convert), rather than
      the per-iter load→convert→append interleave. Both variants **cos 1.000** (26/26 op_tests
      pass). Cold-HBM E=256 A/B (`prefetch/base` µs ratio, 3 trials):

      | B | prefetch/base | verdict |
      |---|---------------|---------|
      | 1 | **0.944–0.955** | **~5% faster** (latency-bound MLP decode) |
      | 2 | 0.979 | ~2% faster |
      | 4 | 1.02 | ~neutral/slightly slower |
      | 8 | 0.99–1.02 | neutral (within noise) |

      So prefetch is a **clear ~5% win at B=1** and neutral above — matching the reference's
      "slower for FP8" note (holding all loads live inflates VGPR pressure, which bites once the
      grid is occupancy-bound at higher B). Because the flag is baked at **build** time while B is
      a runtime grid dim (one cached kernel can't switch per-B), it's kept **default off** and
      exposed as a tuning knob; **recommend `prefetch=True` for B≤2 decode**. Not yet ported to
      the FP8 or gate_up builders (gate_up is already occupancy-bound, cf. G7).

### Phase F — Validation + integration (G9, G10)  [x]
- [x] **CK-Tile side-by-side — DONE (2026-08-18; tracked in the companion
      [`SILOTIGER-667-plan-bench.md`](./SILOTIGER-667-plan-bench.md)).** `tickets/667/harness/compare.py`
      generates both sides and joins on the full `(H,I,E,TOPK,B)` key; the of-record artifact
      `tickets/667/g9_compare.{md,csv}` (repeats=3, reproduce via `run_g9_compare.sh`) covers all
      15 cells with **0 n/a** — every FlyDSL row has a CK peer across DeepSeek-V3 / MiniMax /
      Qwen3Next-TP1 at B∈{1,2,4,8}, including the FP8-act peer (B4) and DeepSeek FP8 E=256 (B5
      Tier-2). D7 additionally validates CK outputs vs torch (FP8/BF16/FP4, non-uniform Block2D
      scales, all cos=1.0). This closes the original plan's last open Phase-4 item.
- [x] **Register** the warp-decode entry points in `aiter/ops/flydsl/__init__.py` (behind
      `is_flydsl_available()`, added to `__all__`) — 2026-08-11. All **six** public entry points
      exported: `flydsl_warp_decode_{gate_up,down_reduce}` (FP8) + `_fp4` (MXFP4, ticket #1) +
      `_bf16` (unquantized oracle). Verified import + `__all__` membership in `flydsl_venv`.
- [x] **Widened perf sweep (2026-08-11):** extended the cold-HBM A/B to **3 real-E decode models**
      (DeepSeek-V3 E256, MiniMax E256, Qwen3Next-TP1 E512) × **B∈{1,2,4,8,32}**, both stages, FP4+FP8
      (FP8 at DeepSeek E256 now runs via the K3 Tier-2 i64 base, B5 2026-08-17; the old 2³¹
      auto-skip is gone). Driven by `_COLD_MODELS`/`_COLD_BATCHES` →
      `COLD_{DOWN,GATE_UP}_SHAPES`; rows now carry HIDDEN/INTER. All **fp4_cos=1.0**. Headline (device
      timing): gate_up FP4 beats FP8 **1.3–1.8×** (MiniMax) / **1.0–1.2×** (Qwen); down FP4 wins at
      B≥2, ~neutral at B=1 for the smaller shapes. These same shapes were then fed to the **CK
      side-by-side (G9)**, which is now complete (of-record `g9_compare` artifact, 0 n/a).

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
| DeepSeek-V3 | 7168 | 2048 | 8 | **256** | ✅ (kv16) | ✅ **A** real-E correctness (both stages); ✅ **F** cold perf B∈{1,2,4,8,32} (FP4; FP8 n/a at E256 — 2³¹, K3 Tier-2) |
| MiniMax | 3072 | 1536 | 8 | **256** | ✅ (kv16/kv8) | ✅ **F** cold perf B∈{1,2,4,8,32} FP4+FP8 (cos 1.0 gate = correctness) |
| Qwen3Next TP1 | 2048 | 512 | 10 | **512** | ✅ (kv8) | ✅ **A** real-E correctness (both stages); ✅ **F** cold perf B∈{1,2,4,8,32} FP4+FP8 |
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
| **Batch B** | 1, 2, 4, 8, 32 | ✅ 1,2,4,8,32 (**F**, cold real-E FP4/FP8 sweep, both stages) |
| **Scale layout** | pertensor, pertoken, block2d | ✅ all three (both stages) |
| **Weight dtype** | FP8, MXFP4, BF16 | ✅ FP8 · ✅ MXFP4 **B** · ✅ BF16 **C** (oracle) |
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
- [resolved, 2026-08-11] FP4 gate_up **accuracy** (ticket gates FP4 gate_up on accuracy;
  MXFP4 mantissa is tiny). **Discharged:** Phase C's `test_gate_up_fp4_matches_bf16_oracle`
  runs the FP4 and BF16 kernels on the same logical weights and gives **cos = 1.000000**, and
  D7 independently cross-checks FP4 outputs vs torch (non-uniform Block2D scales). The win is
  validated, not merely claimed.
- [resolved for split-K] Split-K / LDS were expected to pay only in the **occupancy-bound**
  regime (small-grid Qwen, low B). **Split-K (G5) measured → no-win even there**: B=1 warp-decode
  is launch/memory-latency-floor bound, so partitioning the contraction can't cut the ~4.4µs floor
  (closed, gated-off). **LDS (G6) — now also closed (2026-08-19)**: the small-grid Qwen isolation
  (Phase D) confirmed it shares split-K's fate — Qwen3Next B=1 is already 4–20 waves/CU and latency-
  floor bound (~4–5 µs fixed), so LDS cooperative staging has no occupancy headroom to capture.
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
- _G5 split-K closed as a no-win (2026-08-11)_ — implemented + CU-gated down split-K, but the
  Step 3 A/B (kernel-only + end-to-end) showed no decode win (≤1.03×, k8 regresses; ~2× end-to-end
  via v1 zeros+finalize) because B=1 warp-decode is launch/memory-latency-floor bound. Gated off by
  default; live impl archived on `samaario/warp-decode-moe-splitk-attic`. See the Split-K sub-plan.
- _G10 done — public API registered (2026-08-11)_ — exported all six warp-decode entry points
  (FP8 + MXFP4 + BF16, both stages) from `aiter/ops/flydsl/__init__.py` behind `is_flydsl_available()`
  and added to `__all__`; verified import + membership in `flydsl_venv`. **Remaining Phase F: G9 CK
  side-by-side + widen the perf sweep (§8.2).**
- _Widened perf sweep done (2026-08-11)_ — cold-HBM FP4-vs-FP8 A/B now covers DeepSeek-V3/MiniMax/
  Qwen3Next-TP1 × B∈{1,2,4,8,32}, both stages, via `_COLD_MODELS`/`_COLD_BATCHES`; FP8 legs auto-skip
  at DeepSeek E256 (2³¹). All fp4_cos=1.0; gate_up FP4 wins 1.3–1.8× (MiniMax). §8.2 model rows +
  Batch axis flipped ✅. **Only G9 (CK side-by-side) remains in Phase F.**
- _Doc hygiene: reconciled with post-08-11 work (2026-08-19)_ — this convergence doc had drifted
  from the companion [`-plan-bench.md`](./SILOTIGER-667-plan-bench.md). Flipped stale headers to
  match reality: **Phase A `[~]→[x]`** (both sub-items done — real-E regression cases + K3-scale
  B5 Tier-2), **Phase B `[ ]→[x]`** (FP4 down+gate_up landed, tested, cold-HBM win confirmed; closed
  its two lingering `[ ]` bullets — Block2D<1,32> e8m0 scale layout + torch-correctness/perf-A/B —
  and the `[~]` down-FP4 bullet), and **Phase F `[ ]→[x]`**. Folded in the two items this doc
  predates: **B4** (FP8-activation gate_up — `build_gate_up_fp8_act_module` +
  `flydsl_warp_decode_gate_up_fp8act`, the CK `gate_fp8_d2` peer, DONE 2026-08-18) and **D7** (CK
  outputs numerically cross-checked vs torch across FP8/BF16/FP4 with non-uniform Block2D scales,
  DONE 2026-08-18) — both landed via the G9 harness. **G9 (CK side-by-side) marked done**: of-record
  `tickets/667/g9_compare.{md,csv}` covers all 15 cells with 0 n/a (reproduce via
  `run_g9_compare.sh`). Also flipped the §9 **FP4 gate_up accuracy** risk `[open]→[resolved]`
  (`test_gate_up_fp4_matches_bf16_oracle` cos=1.0 + D7). Net: Phases A/B/C/F now `[x]`; the only
  in-scope open work left is Phase D LDS (G6) + Phase E selectable FP8 ILP (G7 on FP8).
- _G7 selectable ILP dot2 landed on FP8 + A/B'd (2026-08-19)_ — added the `dot2_acc` knob to the
  FP8 `gate_up`/`down` builders (shared `drain_or_chain` helper → `dot2_f32_bf16_drain`), threaded
  through the getters + entry points, across all four scale branches. Correctness: 10 forced
  `dot2_acc=4`-vs-baseline op_tests, all cos=1.0 (24/24 FP8 suite). A/B (gfx950, warm, DeepSeek
  H7168/I2048): **warm/compute-bound** — gate_up ILP is a ~5–6% regression at B=2/4, down ILP +7%
  only at B=4. **Cold-HBM cross-check (E=128 pool ≫ MALL, the production decode regime)** — the warm
  down win **evaporates**: cold is memory-latency-floor bound (same physics as split-K G5), so ILP is
  neutral-to-negative everywhere (down −6 to −8% at B=1/2, gate_up −2 to −3%). All cos=1.0. **Verdict:
  decisive no-win; defaults stay `dot2_acc=1`, knob ships OFF** (retained for A/B only). Closes Phase E
  (G8 prefetch already done). **Remaining in-scope: Phase D LDS (G6) — flagged in §9 as a likely B=1
  no-win; isolate the small-grid case first.**
- _G6 LDS closed as a predicted no-win by feasibility analysis (2026-08-19)_ — before building,
  isolated the smallest-grid production shape (Qwen3Next-TP1 H2048/I512/TOPK10) per the §9 guidance.
  At B=1 the grid already gives 20 waves/CU (gate_up) / 4 waves/CU (down) on CU=256, and both stages
  are launch/memory-latency-floor bound (~4–5 µs fixed floor; 32× work → ~9× time), not occupancy-
  starved — the same physics that closed split-K (G5). LDS cooperative *activation* staging can't
  lower that floor and the weights dominate cold BW, so there's no upside to capture. Closed by
  analysis (not built). Repro `/tmp/g6_feasibility.py`. **Phase D now `[x]` (both occupancy levers
  evaluated & closed as no-wins). Net: Phases A–F all `[x]` — the SILOTIGER-667 convergence is
  functionally complete** (ticket-critical FP4 #1 win, real-E, gfx942 fallback, and G9 CK validation
  all shipped; the three perf levers G5/G6/G7 were each measured and closed as decode no-wins, with
  G8 prefetch retained as a B≤2 knob).
