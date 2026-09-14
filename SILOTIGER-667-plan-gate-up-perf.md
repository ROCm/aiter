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

The hot loop **now** interleaves gate/up `v_dot2` (subtask 1). Native `x` loads
already use 4 unique `klane` addresses per wave; explicit nlane-0 broadcast
was tried and reverted (subtask 3).

## Progress

Track here as work lands (leave unchecked until that item is done). Per-subtask
gates: op_test with `FLYDSL_RUNTIME_ENABLE_CACHE=0` on GPU 6; plus a G9/667
preshuffled gate/up spot-check when the hot loop, wait/reduce, or grid changed.

- [x] 1. Interleave gate/up `v_dot2` (drop half the `s_nop`s)
- [x] 2. Re-A/B G7 `dot2_acc` on the **native** 16×4 grid
- [x] 3. Dedup `x` loads within `nlane` — **WontFix** (broadcast regresses)
- [x] 4. Software-pipeline `k0` (prefetch next kpack; FP4 WontFix)
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
- **Activation stays k-contiguous.** Only B is kpack-native. `x` is packed
  dwords; subtask 3 showed nlane-0 + broadcast does not beat coalesced
  same-address loads.
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

**WontFix (2026-09-14).** Wave ISA already has **2** activation
`buffer_load_dwordx4` + **2** B kpack `dwordx4` per `k0` (`h1024` loop body).
That is one SIMT instruction for all 64 lanes, not 16 separate A loads. The
four `klane` groups already use four unique `x` addresses; hardware coalesces
the 16 nlanes of a klane onto that address.

Explicit “nlane 0 loads, others broadcast” **compiles and is correct** (ternary
`scf.if` + zeros on non-lead; do not put `if nlane==0` in a Python helper —
that is not AST-rewritten). It does not win:

| variant | Qwen B=1 fp8 | Qwen B=1 fp4 | DS B=1 fp8 | DS B=1 fp4 |
|---|---|---|---|---|
| subtask 1 G9 (all-lane `x`) | 19.99 | 16.21 | 74.59 | 48.59 |
| nlane0 + xor-add tree (G9 `/tmp/g9_gu_xdedup_ck.md`, SCLK med 2380) | 22.19 | 23.36 | 84.14 | **84.17** |
| nlane0 + `ds_bpermute` (local 100-iter device) | 21.82 | 17.72 | 77.77 | **63.27** |

Xor-add: `v_add` 2→34 in the `k0` body. `ds_bpermute` is 8 (FP8) / 16 (FP4)
permutes per `k0` and still loses, worst on DeepSeek FP4 (~+30%). Qwen B=1
FP8 `us`/`%peak` do not improve (occupancy 1.25). Keep all-lane coalesced
`x` loads. Do not put `x` in LDS.

- [x] ISA: not a 16× drop in wave `dwordx4` count (already 2 A + 2 B); A
      addresses already unique per klane; B kpack loads stay.
- [x] Explained: broadcast ALU/LDS-permute cost > coalesced duplicate-address
      A traffic. Kernel left on the subtask 1 load path.
- [x] op_test: 100 passed on the xor-add path (reverted). Default kernel
      unchanged vs subtask 2.

### 4. Software-pipeline `k0`

**Done (2026-09-14), with FP4 WontFix.** Native FP8, FP8-act, and BF16 use a
peeled depth-1 pipeline: a prologue issues `k0=0`; the `num_kpack-1` main loop
carries current `x`/gate/up dwords, issues `k0+1`, then consumes `k0`; the
epilogue consumes the last kpack. `final_state = init_state` keeps
`num_kpack==1` legal. No speculative OOB load and no LDS.

FP8 ISA (`h1024`) now issues four next-kpack `buffer_load_dwordx4`s before
current-kpack converts/dots, with staged `vmcnt(5/4/...)` waits. VGPRs rise
**32→48**, without spilling. BF16 likewise has next-kpack dwordx4 loads in
the loop during current dots (48 VGPRs, no spill). The compiler owns exact
`s_waitcnt` placement; source does not inject raw waits.

FP4 is deliberately left unpipelined. Carrying all 16 `x` dwords plus 8 B
dwords improved Qwen B=1 locally (16.2→14.3 us) but regressed DeepSeek B=1
**48.6→56.3 us**. Carrying only B and loading current `x` in the consume
step was worse (Qwen 18.6 us, DeepSeek 65.8 us). The FP4 register/loop-carry
cost is larger than its latency benefit. G9 below therefore has the original
FP4 path and confirms it stays flat.

GPU 6 G9 (`/tmp/g9_gu_prefetch_ck.md`), 100 iters, 3 repeats,
`timing=device`, loaded SCLK median **2380 MHz**; comparison is against
subtask 1 `/tmp/g9_gu_i1_ck.csv`:

| shape | B | dtype | act | before us | prefetch us | delta |
|---|---|---|---|---|---|---|
| qwen3next | 1 | fp4 | bf16 | 16.21 | 16.11 | -0.6% (unchanged path) |
| qwen3next | 1 | fp8 | bf16 | 19.99 | **17.58** | **-12.0%** |
| qwen3next | 1 | fp8 | fp8 | 21.27 | **18.31** | **-13.9%** (5.3% spread) |
| qwen3next | 2 | fp8 | bf16 | 21.07 | **18.77** | **-10.9%** |
| qwen3next | 2 | fp8 | fp8 | 22.16 | **19.81** | **-10.6%** |
| deepseek-v3 | 1 | fp4 | bf16 | 48.59 | 48.53 | -0.1% (unchanged path) |
| deepseek-v3 | 1 | fp8 | bf16 | 74.59 | **70.14** | **-6.0%** (5.3% spread) |
| deepseek-v3 | 1 | fp8 | fp8 | 76.35 | **72.05** | **-5.6%** |
| deepseek-v3 | 2 | fp8 | bf16 | 90.72 | **86.41** | **-4.7%** |
| deepseek-v3 | 2 | fp8 | fp8 | 93.29 | **90.29** | **-3.2%** (5.6% spread) |

- [x] Depth-1 `x` + gate/up B prefetch: native FP8, FP8-act, BF16.
- [x] FP4 WontFix documented after full and B-only prefetch regressions.
- [x] ISA shows next loads in flight during current compute; no spills.
- [x] Cache-off op_test GPU 6: **100 passed**; G9 canary does not regress.

### 5. Split-K over `k0` for small-`INTER` occupancy

Native gate/up at Qwen B=1 is **1.25 waves/CU**. Extra `k0` shards multiply
the grid and shorten the per-wave INTER-side K loop — the opposite of native
down, where the grid already filled CUs.

Do **not** copy down’s `base_grid * k_batch <= CuCount` gate. That rule
never enables Qwen B=1 (`320 > 256` already at `k_batch=1`), which is why
native-down split-K is WontFix and would make this subtask a no-op.

**Auto policy (occupancy, not CU-fill):** `waves = B × TOPK × (INTER/16)`,
`occ = waves / CuCount`. Default `k_batch=1`. Else pick the largest
`k ∈ {4, 2}` such that all of:

- `occ < 2` (under-occupied; Qwen B=1 is 1.25, Qwen B=2 is 2.5, DeepSeek
  B=1 is 4)
- `num_kpack % k == 0`
- `occ * k <= 5` (do not drive past ~5 waves/CU)

That turns on **only Qwen B=1** in the of-record G9 matrix (`k=2` → 2.5
waves/CU, `k=4` → 5). Qwen B=2 and all DeepSeek batches stay `k_batch=1`
unless a dedicated A/B beats of-record **both** B=1 and B=2 (locked
decision). After-prefetch VGPRs are 48; extra waves may not resident —
that is an A/B, not a reason to skip the Qwen B=1 must-win. Depth-1
prefetch still has enough `k0` at `k=2` (16) or `k=4` (8) on Qwen FP8
(`num_kpack=32`).

Qwen B=1 is the must-win. Epilogue: `klane==0` `atomic_add_f32` into a
zeroed workspace, then one silu×up store (or fused last-shard silu if that
stays bit-identical enough for cos ≥ 0.999 — prefer a clear two-step if
fusion fights the atomic).

Do **not** turn this on for large-B DeepSeek if `%peak` is already high
(B=32 gate_up FP8 ~74%): extra atomics would cost bandwidth. The occupancy
gate above already keeps those grids at `k_batch=1`.

- [ ] Native gate/up `k_batch` (or equivalent `k0` shard) + dispatch.
- [ ] Auto occupancy policy as above; Qwen B=1 uses `k_batch>1`; Qwen B=2
      and DeepSeek stay `k_batch=1` unless an A/B wins both batches.
- [ ] **Done when:** op_test covers split and non-split; G9 Qwen B=1 gate_up
      FP8 vs CK improves vs the baseline in this file; G9 gate_up B=1 and
      B=2 do not regress vs of-record (lock).

## Non-goals (do not pull into this plan)

- 32-row / two `n0` on decode B=1 (occupancy).
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=6`.
- MFMA / `fx.gemm`, or describing packed FP8/FP4 as unpacked element layouts.
- Replacing native gate/up with gather, or unifying k-contiguous and native
  builders into one kernel body.
- Native-down schedule work (prefetch / G7 defaults stay on the down plan).
- Mass-comment cleanup as its own commit unless a subtask’s diff is
  unreadable without it.
