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

- [ ] 1. FP8 native down, `k_batch=1`, 16-row waves — PoC is correct but a
      performance no-go; do not land as-is
- [ ] 2. FP4 native down
- [ ] 3. BF16 native down
- [ ] 4. Optional second N tile (32 rows) if 16 N is occupancy-bound
- [ ] 5. Split-K last, only if extra waves over `k0` help

**PoC stop:** subtasks 2–5 are paused. The first implementation reduced the
grid by 8× versus k-contiguous H2 and serialized all TOPK experts in each wave.
At B=1,2 that lost the parallelism needed to hide the long weight stream:
geomean latency was 2.56× the existing gather path. Before porting FP4/BF16,
revise the map to expose more waves (the leading candidate is splitting TOPK
across waves with an accumulation epilogue), then repeat subtask 1.

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
  gate/up.
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

- Grid `B * (HIDDEN/16)` waves of 64. `nlane = lane % 16`, `klane = lane // 16`.
- Per expert in TOPK: load this klane’s INTER slice of `inter` (k-contiguous),
  `_native_kpack_words` on `w_down`, `v_dot2`, fold `router_wt * ds` into the
  per-lane partial, `_reduce_klane4_f32`, `klane==0` writes `y[b, n0*16+nlane]`.
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
- [ ] **Done when:** preshuffled FP8 down uses 16×4 pack loads, not
      `_kpack_load_i32_words` gather; op_test + G9 B=1,2 down fp8 spot-check
      done, and the PoC is fast enough to justify landing.

**PoC result (GPU 1, 100 iterations, 3 repeats, B=1,2):**

| Shape | B | gather µs | native µs | native/gather | native/CK |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V3 | 1 | 65.19 | 137.37 | 2.11 | 3.26 |
| DeepSeek-V3 | 2 | 127.57 | 154.67 | 1.21 | 2.48 |
| MiniMax | 1 | 23.57 | 96.90 | 4.11 | 4.00 |
| MiniMax | 2 | 44.15 | 104.50 | 2.37 | 3.35 |
| Qwen3-Next | 1 | 10.47 | 41.66 | 3.98 | 3.78 |
| Qwen3-Next | 2 | 14.78 | 41.69 | 2.82 | 2.80 |

All six cells had cosine 1.0 and low FlyDSL spread (≤0.6%), so this is not
measurement noise. Native reached only 3–19% of peak. The 16-row wave performs
the right coalesced kpack loads, but there are only `B*HIDDEN/16` waves and each
loops over all TOPK experts. This validates the map's numerics, not its
performance. Do not port this form to FP4/BF16.

### 2. FP4 native down

Copy the FP8 native skeleton. `packed_k = INTER/2`, e8m0 applied in-convert
(`scale_bk` multiple of 32, divides INTER). Same 16×4 map.

- [ ] `_build_down_fp4_preshuffled_native` + dispatch from
      `build_down_reduce_fp4_module`.
- [ ] `test_preshuffled_fp4_matches_k_contiguous` down/combined still pass.
- [ ] **Done when:** preshuffled FP4 down is kpack-native; gather only on
      illegal tiles.

### 3. BF16 native down

Unquantized oracle: 16B kpack is 8 bf16 along K (`INTER % 32`), no weight
scale, only `router_wt`. Thinner than 1–2; still a separate builder so the
fp8/fp4 hot paths stay readable.

- [ ] `_build_down_bf16_preshuffled_native` + dispatch from
      `build_down_reduce_bf16_module`.
- [ ] BF16 preshuffled vs k-contiguous down still matches.
- [ ] **Done when:** BF16 preshuffled down is kpack-native.

### 4. Optional second N tile (32 rows)

Only if 16 N is occupancy-bound on G9 down after subtask 1 (B=1,2 still
latency- or occupancy-limited, not bandwidth). This is **two `n0` tiles per
wave** (or equivalent), not k-contiguous `kh_per_warp=2`. Skip entirely if
subtask 1 already saturates the interesting decode batches.

- [ ] Measure G9 B=1,2 down after subtask 1; decide go/no-go here.
- [ ] If go: 32-row variant, cosine vs 16-row and vs k-contiguous.
- [ ] **Done when:** either explicitly WontFix in Progress, or 32-row path
      gated and G9 B=1,2 down re-checked.

### 5. Split-K last

Native already splits INTER 4 ways via `klane`. Extra `k_batch` over `k0`
tiles only if INTER is large enough that more waves help. Do not port
k-contiguous split-K onto the gather path as a stand-in.

- [ ] Decide from G9 / DeepSeek INTER after 1–3 (and 4 if it shipped).
- [ ] If go: split `k0` across waves, `atomic_add_f32` epilogue like
      k-contiguous `k_batch>1`.
- [ ] **Done when:** either explicitly WontFix, or split-K native down matches
      non-split cosine and does not regress B=1,2.

## Non-goals (do not pull into this plan)

- Unifying k-contiguous and native down into one kernel body via a view.
- `make_layout_tv` on the nested kpack memref (profile mismatch; addressing
  already uses nested `fx.slice`).
- Changing GPU from the locked `HIP_VISIBLE_DEVICES=1`.
- Changing k-contiguous down defaults (`kh_per_warp`, `dot2_acc`, split-K
  policy) except as they dispatch into native when `preshuffled`.
- Gate/up native work (already landed).
