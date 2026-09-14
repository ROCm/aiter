# SILOTIGER-667 — Preshuffled gate/up performance

Close the G9 hole on warp-decode **preshuffled gate/up**. The 16×4 kpack-native
map has already landed (`_build_gate_up_*_preshuffled_native`); this is an
*occupancy and inner-loop schedule* track, not a new shuffle and not MFMA.
Math stays packed `v_dot2_f32_bf16`. Down’s native map is out of scope here
(see `SILOTIGER-667-plan-native-down.md`).

**In scope:** `aiter/ops/flydsl/kernels/warp_decode_moe.py` native gate/up
builders (`fp8`, `fp8_act`, `fp4`, `bf16`) and their dispatch in
`aiter/ops/flydsl/warp_decode_moe.py`. Tests and G9/667 benches as a gate.

**Out of scope (this track):** `fx.gemm` / MFMA; LDS tiling of B; making
shuffled B look k-contiguous; FlyDSL surface modernization
(`SILOTIGER-667-plan-modernize.md`); native-down 32-row / `k0` split-K
(already WontFix on down). Do not pull k-contiguous gather gate/up into this
plan except as a correctness oracle.

Work the subtasks **in order**. Later items assume earlier ones have landed.
Each subtask is surgical: verify correctness (and a G9 Qwen B=1 gate/up
spot-check when the hot loop or grid changed) before starting the next.

## Why this track

GPU 6 of-record-clock G9 (preshuffled, 1000 iters, 2026-09-14): gate/up FP8 is
the remaining CK gap. Down on the same path beats or matches CK.

| shape | B | op | dtype | FlyDSL/CK | fly %peak |
|---|---|---|---|---|---|
| deepseek-v3 | 1 | down | fp8 | 0.82 | 42 |
| deepseek-v3 | 1 | gate_up | fp8 | **1.78** | 35 |
| qwen3next | 1 | down | fp8 | 1.15 | 10 |
| qwen3next | 1 | gate_up | fp8 | **2.52–2.72** | 12 |

CK gate/up at DeepSeek B=1 is only ~10% slower than CK down despite two B
streams. FlyDSL gate/up is ~2.4× its own down. Bytes do not explain the hole.

Native gate/up grid is `B × TOPK × INTER/16` waves (16 output rows / wave),
not gather’s `B × TOPK × INTER`:

| shape | B=1 gate/up waves | waves/CU (256) | down waves/CU |
|---|---|---|---|
| deepseek-v3 | 1024 | **4** | 14 |
| qwen3next | 320 | **1.25** | 5 |

G7’s “`dot2_acc>1` is ~4% slower; B=1 is occupancy-bound” A/B was on the
**gather** grid. On this map, Qwen B=1 gate/up is occupancy-starved. Split-K
over `k0` was WontFix for **down** because down already fills CUs; gate/up
is the opposite.

The hot loop **now** interleaves gate/up `v_dot2` (subtask 1). `x` loads still
ignore `nlane`: all 16 nlanes issue the same activation dwordx4s every `k0`
(subtask 3).

## Progress

Track here as work lands (leave unchecked until that item is done). Per-subtask
gates: op_test with `FLYDSL_RUNTIME_ENABLE_CACHE=0` on GPU 6; plus a G9/667
preshuffled gate/up spot-check when the hot loop, wait/reduce, or grid changed.

- [x] 1. Interleave gate/up `v_dot2` (drop half the `s_nop`s)
- [x] 2. Re-A/B G7 `dot2_acc` on the **native** 16×4 grid
- [ ] 3. Dedup `x` loads within `nlane` (broadcast per `klane`)
- [ ] 4. Software-pipeline `k0` (prefetch next kpack)
- [ ] 5. Split-K over `k0` for small-`INTER` occupancy (Qwen B=1 first)

## Locked decisions

These locks apply to **this track** (subtasks 1–5).

### Test environment

Run all tests/benches in **`flydsl_venv`** **GPU 6**
(`HIP_VISIBLE_DEVICES=6`).

Record loaded SCLK in any G9/667 artifact (harness `ClockSampler`). Do not
claim a perf win from a run whose loaded SCLK median is below ~2 GHz; idle
94 MHz samples are not a gate. Prefer `tickets/667/harness/run_g9_compare.sh
--gpu 6`. If `timing=device` (rocprof) faults on a cell, `cuda_event` is an
allowed A/B clock — say so in the artifact, and use the **same** timer on
both sides of the A/B.

- **Keep the 16×4 map.** Native gate/up stays `(nlane, klane)` on
  `w_gate`/`w_up` `[E, INTER, HIDDEN]`. Do not invent gather `kVector` as a
  substitute, and do not add 32-row / two `n0` for decode B=1 (that halves an
  already thin grid and would make Qwen worse).
- **Activation stays k-contiguous.** Only B is kpack-native. `x` is still
  packed dwords; subtask 3 may broadcast those loads, not change layout.
- **Keep ISA-level helpers.** `v_dot2_f32_bf16` inline asm, `s_nop 2` / G7
  drain, `cvt_scalef32_pk_bf16_{fp8,fp4}`, and E8M0 `shl 23` stay; there is no
  `fx.gemm` atom for `v_dot2`.
- **Surgical numerics.** Cos ≥ 0.999 vs existing
  `op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py` cases (preshuffled and
  k-contiguous). Do not change scale folding or silu fusion.
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale HSACO
  cannot mask a bad rewrite.
- **Gate.**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=6 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py -v
  ```
  After any change that touches the gate/up hot loop, wait/reduce, or grid,
  spot-check G9 **gate_up** (FP8 and MXFP4) with
  `tickets/667/harness/run_g9_compare.sh --gpu 6 --iters 100 --flydsl-weight-layout preshuffled -- --shapes qwen3next,deepseek-v3 --batches 1,2`.
  Canary: Qwen B=1 gate_up FP8. Claiming a CK win needs 1000 iters on GPU 6
  with loaded SCLK in the of-record band.
- **No B=1/B=2 regression.** A change ships only if that G9 gate_up
  spot-check does not slow the previous of-record on **both** `--batches 1,2`
  (same timer, loaded SCLK in band, both shapes). A B=1 win that regresses
  B=2, or the reverse, is a no-ship unless that dtype/path is WontFix'd with
  the numbers. Qwen B=1 is the occupancy canary; it does not replace the B=2
  check. Spread of a few percent on an **unchanged** path is noise, not a
  veto.

## Subtasks

### 1. Interleave gate/up `v_dot2`

**Done (2026-09-14).** Native `k0` uses `drain_or_chain_gate_up`: default
`dot2_acc=1` emits `g0, u0, g1, u1, …`. Gate dots `serialize=False`; **every
up** dot `serialize=True` (half the nops). `dot2_acc>1` still does per-stream
G7. Down stays on `drain_or_chain`. Gather gate/up is still sequential.

Native gate/up `k0` is `range(num_kpack)` (`scf.for`), not
`range_constexpr` (full unroll). Fully unrolled DeepSeek `k0=112` plus
`has_side_effects=False` on `v_dot2` miscompiled (cos ~0.973). Forcing
`has_side_effects=True` was correct but slow (~107 µs DeepSeek B=1 FP8).
Looping `k0` + interleave + nop-on-every-up + `has_side_effects=False` is
the combo that is both correct and faster.

Do **not** nop only the last up: `g_i` then `u_i` then `g_{i+1}` reuses the
gate acc after one instruction (DeepSeek cold FP8 cos ~0.973). BF16 must
build pair lists at Python compile time and call the helper — `serialize
and last` inside `range_constexpr` became `scf.if` and cos ~0.68.

- [x] Interleaved gate/up dots in the native `k0` loop (FP8, FP8-act, FP4, BF16).
- [x] ISA (native FP8 preshuffled, `h1024`): unrolled k0 was `s_nop 2` **256→128**,
      `v_dot2` 256, `buffer_load_dwordx4` 64. Looped k0 body is **8 nops / 16
      dots / 4 dwordx4** per `k0` (same ratio; loads not scalarized).
- [x] op_test cache-off GPU 6: **100 passed**.
- [x] G9 spot-check GPU 6, 100 iters, 3 repeats, `timing=device`, loaded SCLK
      median **2380 MHz** (`/tmp/g9_gu_i1_ck.md`). Gate_up vs the 1000-iter
      living table above (same shapes; 100 vs 1000 iters):

| shape | B | dtype | act | flydsl_us | ratio(f/c) | vs living 1000-iter |
|---|---|---|---|---|---|---|
| qwen3next | 1 | fp4 | bf16 | 16.21 | 2.17 | ~flat (occupancy 1.25) |
| qwen3next | 1 | fp8 | bf16 | 19.99 | 2.57 | slight (was ~21 µs / 2.73) |
| qwen3next | 1 | fp8 | fp8 | 21.27 | 2.52 | flat vs ~2.53 |
| deepseek-v3 | 1 | fp4 | bf16 | 48.59 | **0.975** | was ~62 µs / 1.26 |
| deepseek-v3 | 1 | fp8 | bf16 | 74.59 | **1.52** | was ~83 µs / 1.78 |
| deepseek-v3 | 1 | fp8 | fp8 | 76.35 | **1.57** | was ~85 µs / 1.78 |
| deepseek-v3 | 2 | fp8 | bf16 | 90.72 | 1.03 | was ~1.11 |

Qwen B=1 is still occupancy-starved (subtask 5). DeepSeek B=1 FP8 closed
part of the hole; FP4 now matches CK on this 100-iter canary. Claiming a
CK win vs the of-record table still needs 1000 iters.

### 2. Re-A/B G7 `dot2_acc` on the native 16×4 grid

**Done (2026-09-14).** Native-grid sweep on GPU 6, `timing=device`, 100 iters,
20 warmup, 3 repeats, cache-off, preshuffled. Cos 1.0 on every cell. Spread
≤2.9% (most ≤1%). `vs` is relative to that shape/dtype’s **interleave +
acc=1**.

| shape | dtype | interleave | acc | us | spr% | vs i1/acc1 |
|---|---|---|---|---|---|---|
| qwen3next | fp8 | True | 1 | 20.48 | 0.7 | 1.000 |
| qwen3next | fp8 | True | 2 | 20.68 | 0.3 | 1.010 |
| qwen3next | fp8 | True | 4 | 20.86 | 0.7 | 1.019 |
| qwen3next | fp8 | True | 8 | 21.28 | 0.4 | 1.039 |
| qwen3next | fp8 | False | 1 | 22.00 | 0.3 | 1.074 |
| qwen3next | fp8 | False | 2 | 20.82 | 0.2 | 1.017 |
| qwen3next | fp8 | False | 4 | 21.21 | 0.3 | 1.036 |
| qwen3next | fp8 | False | 8 | 23.12 | 2.2 | 1.129 |
| qwen3next | fp4 | True | 1 | 16.55 | 1.0 | 1.000 |
| qwen3next | fp4 | True | 2 | 14.65 | 0.3 | **0.885** |
| qwen3next | fp4 | True | 4 | 14.90 | 0.2 | 0.900 |
| qwen3next | fp4 | True | 8 | 15.57 | 0.6 | 0.941 |
| qwen3next | fp4 | False | 1 | 18.17 | 0.6 | 1.098 |
| qwen3next | fp4 | False | 2 | 14.76 | 0.8 | 0.892 |
| qwen3next | fp4 | False | 4 | 15.31 | 0.7 | 0.925 |
| qwen3next | fp4 | False | 8 | 17.58 | 0.4 | 1.062 |
| deepseek-v3 | fp8 | True | 1 | 76.19 | 2.9 | 1.000 |
| deepseek-v3 | fp8 | True | 2 | 75.76 | 0.2 | 0.994 |
| deepseek-v3 | fp8 | True | 4 | 73.74 | 0.3 | 0.968 |
| deepseek-v3 | fp8 | True | 8 | 77.83 | 0.3 | 1.021 |
| deepseek-v3 | fp8 | False | 1 | 77.47 | 0.5 | 1.017 |
| deepseek-v3 | fp8 | False | 2 | 76.28 | 0.3 | 1.001 |
| deepseek-v3 | fp8 | False | 4 | 77.19 | 0.6 | 1.013 |
| deepseek-v3 | fp8 | False | 8 | 78.80 | 0.3 | 1.034 |
| deepseek-v3 | fp4 | True | 1 | 48.90 | 0.8 | 1.000 |
| deepseek-v3 | fp4 | True | 2 | 45.73 | 1.0 | **0.935** |
| deepseek-v3 | fp4 | True | 4 | 45.90 | 0.8 | 0.939 |
| deepseek-v3 | fp4 | True | 8 | 48.34 | 1.1 | 0.989 |
| deepseek-v3 | fp4 | False | 1 | 53.07 | 0.9 | 1.085 |
| deepseek-v3 | fp4 | False | 2 | 46.37 | 1.4 | 0.948 |
| deepseek-v3 | fp4 | False | 4 | 46.88 | 0.7 | 0.959 |
| deepseek-v3 | fp4 | False | 8 | 65.35 | 0.7 | 1.336 |

**Keep `dot2_acc=1`.** Qwen B=1 gate_up FP8 (the change gate) is best at
interleave+acc=1; acc>1 is 1–4% slower. DeepSeek B=1 FP8 acc=4 is ~3%
faster but that cell’s acc=1 spread was 2.9%, and changing the default
would move Qwen the wrong way. Interleave already covers the RAW; extra
G7 accs add VGPR without filling Qwen’s 1.25 waves/CU.

Native **FP4 acc=2** is a real leftover: Qwen **−11.5%**, DeepSeek **−6.5%**,
both outside spread. Do not fold that into the shared default (it would
also hit k-contiguous gather). A later dtype-split (`fp4=2`, `fp8=1`) is
allowed; this subtask does not.

`interleave_gate_up` stays a compile-time knob (default True). `dot2_acc>1`
with interleave now pairs G7 accs across gate/up instead of two sequential
drains.

- [x] Native-grid A/B table (Qwen/DeepSeek B=1, FP8+FP4) recorded here.
- [x] Default: **keep 1** (Qwen B=1 FP8 does not improve).
- [x] op_test cache-off GPU 6: **100 passed** (`dot2_acc=1` default).

### 3. Dedup `x` loads within `nlane`

`x_word0 = (token_b * hidden + k_base) // 2` depends on `klane`, not
`nlane`. All 16 nlanes issue the same 32 B activation load every `k0`. B
kpacks are unique per `(klane, nlane)`; A is not.

Load `x` once per `klane` and broadcast (`readlane` / DPP). Do not move `x`
into LDS unless broadcast VGPR pressure forces it (that is a last resort,
not the first design).

- [ ] One A transaction per `klane` per `k0`, 16 nlanes share it.
- [ ] ISA check: activation `buffer_load_dwordx4` count per `k0` drops ~16×
      (or coalesces to the 4 unique `klane` addresses); B kpack loads stay
      16 B / lane.
- [ ] **Done when:** op_test + G9 canary; Qwen B=1 gate_up FP8 `%peak` or `us`
      improves or is explained if it does not (VGPR/occupancy trade).

### 4. Software-pipeline `k0`

Each `k0` is currently load `x`+`wg`+`wu`, convert, dots, optional block2d
scale, then the next `k0`. Overlap next-kpack VMEM with current convert/dot
(`s_waitcnt` so the ALU is not waiting on the same iter’s loads).

Prefetch depth 1 is enough to start. Do not add `SharedAllocator` B tiles.

- [ ] Double-buffered kpack/`x` loads in the native `k0` loop (all four
      gate/up dtypes), or a documented reason the compiler already overlaps
      them (ISA waitcnt vs load/dot).
- [ ] **Done when:** ISA shows loads of `k0+1` in flight during `k0` dots, or
      WontFix with the waitcnt dump; G9 canary does not regress.

### 5. Split-K over `k0` for small-`INTER` occupancy

Native gate/up at Qwen B=1 is **1.25 waves/CU**. Extra `k0` shards multiply
the grid and shorten the per-wave INTER-side K loop — the opposite of native
down, where the grid already filled CUs.

Policy (mirrors down’s CU-aware split, but for **gate/up native**): only
when `base_grid * k_batch <= CuCount` and `num_kpack % k_batch == 0`.
DeepSeek B=1 (4 waves/CU) may take a small `k_batch` if VGPR allows; Qwen
B=1 is the must-win. Epilogue: `klane==0` `atomic_add_f32` into a zeroed
workspace, then one silu×up store (or fused last-shard silu if that stays
bit-identical enough for cos ≥ 0.999 — prefer a clear two-step if fusion
fights the atomic).

Do **not** turn this on for large-B DeepSeek if `%peak` is already high
(B=32 gate_up FP8 ~74%): extra atomics would cost bandwidth.

- [ ] Native gate/up `k_batch` (or equivalent `k0` shard) + dispatch.
- [ ] Auto policy vs CU count; Qwen B=1 uses `k_batch>1`; saturated grids
      stay `k_batch=1`.
- [ ] **Done when:** op_test covers split and non-split; G9 Qwen B=1 gate_up
      FP8 vs CK improves vs the baseline in this file; DeepSeek B=32 does
      not regress outside noise.

## Non-goals (do not pull into this plan)

- 32-row / two `n0` on decode B=1 (occupancy).
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=6`.
- MFMA / `fx.gemm`, or describing packed FP8/FP4 as unpacked element layouts.
- Replacing native gate/up with gather, or unifying k-contiguous and native
  builders into one kernel body.
- Native-down schedule work (prefetch / G7 defaults stay on the down plan).
- Mass-comment cleanup as its own commit unless a subtask’s diff is
  unreadable without it.
