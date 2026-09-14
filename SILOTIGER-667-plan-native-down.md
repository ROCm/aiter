# SILOTIGER-667 — Native preshuffled down

Replace gather-on-shuffled-B in warp-decode **down** with the same N-major
kpack-native map gate/up already uses: 16 hidden rows per wave (`nlane`), 4
klanes on INTER, 16B pack load, xor-16/32 reduce, `klane==0` stores. Math stays
packed `v_dot2_f32_bf16`. This is a *kernel-map* track, not a layout-algebra
rewrite and not k-contiguous `kh_per_warp=2`.

**In scope:** `aiter/ops/flydsl/kernels/warp_decode_moe.py`,
`aiter/ops/flydsl/warp_decode_moe.py`. Tests and G9/667 benches as a gate.

**Out of scope (this track):** making shuffled B look k-contiguous for
`load_i32_words(word0, n_wwords)`; `make_layout_tv` / `partition_S` on the
nested kpack memref; `fx.gemm` / MFMA; LDS/`SharedAllocator`; FlyDSL surface
modernization (see `SILOTIGER-667-plan-modernize.md`). Nested `(N, K_dword)`
addressing for the existing gather/native-gate_up paths has landed and is not
a subtask here.

Work the subtasks **in order**. Later items assume earlier ones have landed.
Each subtask is surgical: verify correctness (and a G9 B=1,2 down spot-check
when the hot loop changed) before starting the next.

## Progress

Track here as work lands (leave unchecked until that item is done). Per-subtask
gates: op_test with `FLYDSL_RUNTIME_ENABLE_CACHE=0` on GPU 1; plus a G9/667
spot-check when the hot loop or wait/reduce path changed.

- [x] 1. FP8 native down, `k_batch=1`, 16-row TOPK-parallel waves — landed as
      the preshuffled FP8 default; Qwen B=1 vs gather accepted
- [x] 2. FP4 native down — TOPK-parallel 16×4, same grid/epilogue as FP8
- [x] 3. BF16 native down — TOPK-parallel 16×4 oracle; G9 has no BF16 down cell
- [x] 4. Optional second N tile (32 rows) — **WontFix** (cuts waves; 16 N is not occupancy-bound)
- [x] 5. Split-K over `k0` — **WontFix** (native grid already fills CUs; extra `k0` shards would not help B=1,2)

**Landed:** TOPK-parallel 16×4 (`grid = B*TOPK*(HIDDEN/16)`, one expert per
wave, `atomic_add_f32` into a zeroed FP32 `y`) is the preshuffled FP8, FP4,
and BF16 down default on legal tiles. Serial-TOPK is abandoned. No INTER/grid
cutoff. Qwen B=1 vs gather is accepted for FP8; FP4 Qwen B=1 **beats** gather.
Subtask 4 (32-row / two `n0`) is **WontFix**. Subtask 5 (extra `k0`
split-K) is **WontFix**.

## Locked decisions

These locks apply to **this track** (subtasks 1–5).

- **Test environment:** run all tests/benches in **`flydsl_venv`** **GPU 1**
  (`HIP_VISIBLE_DEVICES=1`).
- **Keep the 16×4 map.** Native down is gate/up’s `(nlane, klane)` contract on
  `w_down[E, HIDDEN, INTER]`, not k-contiguous “`kh` rows × 64-way K”. Do not
  invent `kh_per_warp=2` on the gather grid as a substitute for this map.
- **Activation stays k-contiguous.** Only B is kpack-native. `inter` loads
  remain consecutive dword/`buffer_ops` (or the current packed path).
- **Keep ISA-level helpers.** `v_dot2_f32_bf16` inline asm, `s_nop 2` / G7
  drain, `cvt_scalef32_pk_bf16_{fp8,fp4}`, and E8M0 `shl 23` stay.
- **Gather remains the fallback** when HIDDEN % 16 or INTER is not a native
  kpack tile (FP8: INTER % 64; FP4: INTER % 128; BF16: INTER % 32), matching
  gate/up. Legal-tile preshuffled FP8/FP4/BF16 down does **not** keep a gather
  cutoff for small INTER or Qwen B=1.
- **Do not fold native down into the k-contiguous builder body.** Early-return
  a dedicated `_build_down_*_preshuffled_native` (same pattern as gate/up).
- **Compile cache.** After kernel-source edits, run with
  `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or clear `~/.flydsl/cache`) so a stale HSACO
  cannot mask a bad rewrite.
- **Gate.**
  ```bash
  source /path/to/flydsl_venv/bin/activate
  HIP_VISIBLE_DEVICES=1 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
    python3 -m pytest op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py -v -k preshuffled
  ```
  After subtask 1 (and after any later change that touches the down hot loop),
  spot-check G9 B=1,2 **down** with
  `tickets/667/harness/run_g9_compare.sh --gpu 1 --iters 100 --flydsl-weight-layout preshuffled`.
  Cos ≥ 0.999 vs k-contiguous down on the existing op_test cases.

## Subtasks

### 1. FP8 native down, `k_batch=1`, 16-row waves

The G9 hole: preshuffled down FP8 is ~2× CK at B=1,2 because it still gathers.
Mirror `_build_gate_up_fp8_preshuffled_native` onto down:

- Grid `B * TOPK * (HIDDEN/16)` waves of 64 (TOPK in the grid, like gate/up).
  `nlane = lane % 16`, `klane = lane // 16`. One expert per wave.
- Load this klane’s INTER slice of `inter` (k-contiguous), `_native_kpack_words`
  on `w_down`, `v_dot2`, fold `router_wt * ds`, `_reduce_klane4_f32`,
  `klane==0` `atomic_add_f32` into a caller-zeroed FP32 `y` (host finalizes bf16).
- No `kh_per_warp` (16 N already shares one INTER chunk). No `k_batch`.
- Scales: pertensor/pertoken first; add block2d if G9 FP8 down needs
  `block2d(128,128)` to pair CK.
- Early-return from `build_down_reduce_fp8_module` when `preshuffled` and the
  native tile constraints hold; else keep gather.

- [x] `_build_down_fp8_preshuffled_native` (or equivalent) + dispatch.
- [x] `test_preshuffled_fp8_down_matches_k_contiguous` (and combined) still
      cos ≥ 0.999.
- [x] G9 B=1,2 down fp8 vs CK: FlyDSL should move off the gather ~20% peak
      plateau; record the new ratios in this section when the run lands.
- [x] **Done when:** preshuffled FP8 down uses 16×4 pack loads, not
      `_kpack_load_i32_words` gather; op_test + G9 B=1,2 down fp8 spot-check
      done, and the PoC is fast enough to justify landing.

**Serial-TOPK PoC (abandoned):** `grid = B*(HIDDEN/16)` was correct (cos 1.0)
but geomean **2.56× slower** than gather at B=1,2 (3–19% peak). Occupancy, not
the 16×4 pack load, was the failure.

**Landed map:** `grid = B*TOPK*(HIDDEN/16)`, atomic FP32 epilogue. Native is
the default whenever `preshuffled and k_batch==1 and INTER%64==0 and
HIDDEN%16==0`. Qwen B=1 remaining slower than gather is accepted (no cutoff).

**G9 (GPU 1, 100 iterations, 3 repeats, clocks ~94 MHz, cosine 1.0).**
B=1,2 vs prior gather / serial native (`/tmp/g9_native_down_topk_ck.*`):

| Shape | B | gather µs | serial µs | native µs | n/gather | n/CK | %peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 1 | 65.19 | 137.37 | 36.85 | 0.57 | 0.87 | 39.8 |
| DeepSeek-V3 | 2 | 127.57 | 154.67 | 58.73 | 0.46 | 0.94 | 50.0 |
| MiniMax | 1 | 23.57 | 96.90 | 23.56 | 1.00 | 0.97 | 20.0 |
| MiniMax | 2 | 44.15 | 104.50 | 27.08 | 0.61 | 0.87 | 34.9 |
| Qwen3-Next | 1 | 10.47 | 41.66 | 15.35 | 1.47 | 1.39 | 8.5 |
| Qwen3-Next | 2 | 14.78 | 41.69 | 15.87 | 1.07 | 1.06 | 16.5 |

B=1,2 geomean vs gather **0.79**; vs serial **0.31**; vs CK **~1.00**.

Full preshuffled sweep in `tickets/667/g9_compare_ck.{md,csv}` (same kernel;
B=1,2 cells match the table within a few percent, Qwen B=1 spread 0.5% so
the gather loss is real). Down FP8 vs CK: DeepSeek **0.84–0.93** (B=32
**0.81**, 72% peak); MiniMax **0.79–0.99**; Qwen **1.41 / 1.07** at B=1,2
then **0.74–0.95** from B=4. All-B geomean vs gather **0.53**, vs CK
**0.89** (13/15 cells beat both).

### 2. FP4 native down

Copy the FP8 native skeleton. `packed_k = INTER/2`, e8m0 applied in-convert
(`scale_bk` multiple of 32, divides INTER). Same 16×4 map and TOPK-parallel
grid.

- [x] `_build_down_fp4_preshuffled_native` + dispatch from
      `build_down_reduce_fp4_module`.
- [x] `test_preshuffled_fp4_matches_k_contiguous` down/combined still pass.
- [x] **Done when:** preshuffled FP4 down is kpack-native; gather only on
      illegal tiles.

**G9 B=1,2 down FP4** (GPU 1, 100 iters, 3 repeats, clocks ~94 MHz, cos 1.0;
`/tmp/g9_native_down_fp4_ck.{md,csv}`). Gather µs from the prior preshuffled
sweep. Native is the default on INTER % 128 and HIDDEN % 16.

| Shape | B | gather µs | native µs | n/gather | n/CK | %peak |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 1 | 38.61 | 27.70 | 0.72 | 0.98 | 28.2 |
| DeepSeek-V3 | 2 | 72.16 | 46.62 | 0.65 | 1.29 | 33.5 |
| MiniMax | 1 | 20.22 | 17.73 | 0.88 | 0.88 | 14.1 |
| MiniMax | 2 | 25.26 | 20.38 | 0.81 | 0.89 | 24.6 |
| Qwen3-Next | 1 | 10.44 | 9.39 | 0.90 | 0.88 | 7.4 |
| Qwen3-Next | 2 | 12.35 | 10.09 | 0.82 | 0.92 | 13.8 |

Geomean vs gather **0.79** (all six win, including Qwen B=1); vs CK **0.96**.
DeepSeek B=2 is the only CK loss (1.29×). Land as the FP4 default; no cutoff.

### 3. BF16 native down

Unquantized oracle: 16B kpack is 8 bf16 along K (`INTER % 32`), no weight
scale, only `router_wt`. Same TOPK-parallel 16×4 map as FP8/FP4. `use_dot2`
stays wired (`dot2_or_scalar`) so the gfx942 scalar fallback still works.

- [x] `_build_down_bf16_preshuffled_native` + dispatch from
      `build_down_reduce_bf16_module`.
- [x] BF16 preshuffled vs k-contiguous down still matches.
- [x] **Done when:** BF16 preshuffled down is kpack-native.

G9 has no BF16-down vs CK cell (`FLYDSL_CELLS` is FP8/FP4 only). Gate is
op_test: 35 preshuffled passed, including `test_preshuffled_bf16_matches_k_contiguous`
and `test_down_reduce_bf16` / combined. Native when INTER % 32 and HIDDEN % 16;
gather only on illegal tiles.

### 4. Optional second N tile (32 rows)

Only if 16 N is occupancy-bound on G9 down after subtask 1 (B=1,2 still
latency- or occupancy-limited, not bandwidth). This is **two `n0` tiles per
wave** (or equivalent), not k-contiguous `kh_per_warp=2`. Skip entirely if
subtask 1 already saturates the interesting decode batches.

**No-go.** 32-row is two `n0` per wave, so `grid = B*TOPK*(HIDDEN/32)` — half
the waves of 16×4. That is the wrong direction wherever B=1,2 is still
occupancy- or latency-limited (Qwen), and it is unnecessary where 16 N already
streams.

Of-record G9 (`tickets/667/g9_compare_ck.md`, GPU 1, native 16×4 default):

| Shape | B | down FP8 %peak | down FP4 %peak |
|---|---:|---:|---:|
| DeepSeek-V3 | 1 | 64.8 | 34.1 |
| DeepSeek-V3 | 32 | 74.5 | 52.3 |
| MiniMax | 1 | 32.2 | 25.8 |
| MiniMax | 32 | 67.5 | 54.8 |
| Qwen3-Next | 1 | 20.6 | 13.5 |
| Qwen3-Next | 32 | 63.9 | 54.7 |

FP8 DeepSeek B=32 is bandwidth-saturated (~75% peak). FP4 B=32 tops out ~52–55%
peak because packed B is half the bytes plus convert, not because 16 N starves
the CU; cutting waves would not add bytes in flight at B=32 and would worsen
Qwen B=1. G9 has no BF16-down cell; the same 16×4 grid applies.

- [x] Measure G9 B=1,2 (and B=32) down after subtasks 1–3; **no-go**.
- [x] **WontFix:** no 32-row kernel, no extra dispatch gate.

**Done when:** explicit WontFix in Progress (this section).

### 5. Split-K last

Native already splits INTER 4 ways via `klane`. Extra `k_batch` over `k0`
tiles only if INTER is large enough that more waves help. Do not port
k-contiguous split-K onto the gather path as a stand-in.

**No-go.** `klane` already partitions INTER 4 ways inside the wave. Extra
`k_batch` over `k0` would multiply the grid (`B*TOPK*(HIDDEN/16)*k_batch`) and
add another `atomic_add_f32` into the same FP32 `y` (on top of TOPK). G5
auto-split-K only fires when `base_grid * k <= CuCount` (256 on this MI355X)
and returns 1 for already-full grids. Native TOPK-parallel 16×4 already
exceeds that:

| Shape | B | native down waves | waves/CU | FP8 `k0` (`INTER/64`) |
|---|---:|---:|---:|---:|
| DeepSeek-V3 | 1 | 3584 | 14 | 32 |
| DeepSeek-V3 | 32 | 114688 | 448 | 32 |
| MiniMax | 1 | 1536 | 6 | 24 |
| Qwen3-Next | 1 | 1280 | 5 | 8 |

Auto `split_k` would be **1** on every G9 native grid. DeepSeek B=1 down FP8
is already ~65% peak (B=32 ~75%). Qwen B=1 is the occupancy-looking cell
(~21% peak) because INTER is **short** (`k0=8`), not because the grid is
smaller than the CU count; splitting those 8 tiles makes each wave shorter
and adds atomics. That is the opposite of “INTER large enough that more
waves help.” Default stays `k_batch=1` / native; `split_k>1` remains the
illegal-tile gather fallback, not a native `k0` shard.

- [x] Decide from G9 / DeepSeek INTER after 1–4; **no-go**.
- [x] **WontFix:** no native `k0` split-K kernel.

**Done when:** explicit WontFix in Progress (this section).

## Non-goals (do not pull into this plan)

- Unifying k-contiguous and native down into one kernel body via a view.
- `make_layout_tv` on the nested kpack memref (profile mismatch; addressing
  already uses nested `fx.slice`).
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=1`.
- Changing k-contiguous down defaults (`kh_per_warp`, `dot2_acc`, split-K
  policy) except as they dispatch into native when `preshuffled`.
- Gate/up native work (already landed).
