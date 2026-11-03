# ROUND-FP8PERF — recover fp8 speed without WMMA hazards; fp8 decode-group default w64

> **ROUND-FP8PERF2 update (`5e01dd8c`):** the GPU A/B chose new defaults, draft fp8
> **`qpre`** and decode-group fp8 **`base`** (w64). See "GPU decision 2026-11-03" below.

Base: ROUND-W64FIX2 `5fb8740e` (kernel sha `a3e2ae3f…`). Branch `qsa-index-scorer/opt-w64fix`.
Final kernel sha `e407fa15dbb35fb650d2da87a213e8a5c6d1f061da9ce0d335dbeb85e01d20c4`.
CPU/no-device evidence only. **No GPU run in this round.**

## Problem (root GPU, a08-1 GPU3, `5fb8740e`)

The hazard-free intrinsic pk8 converter regressed fp8:

| shape | asm | intrin |
|---|---|---|
| draft fp8 c32768 | 11.6 | 12.7–13.0 µs |
| draft fp8 c65536 | 19.3 | 20.6 µs |

Decode fp8 with COLS=64 (intrin) was the best decode build: c65536 26.6 µs, c32768 14.7 µs.

## Analysis (ISA, draft fp8 tdm w32, i32)

| build | instr | VGPR | SGPR | v_nop | s_delay_alu | Q-load waits | hazards |
|---|---|---|---|---|---|---|---|
| asm (`_cvtasm`) | 798 | 106 | 80 | 1 | 51 | `s_wait_loadcnt` 0x8/0x6/0x4/0x2 | 15 (latent) |
| intrin `5fb8740e` | 811 | 91 | 66 | 13 | 58 | 4× `s_wait_loadcnt 0x0` | 0 |

The v_nop padding is not the main cost: only +12 v_nop in total, 4 + 1 + 4 + 2 in the hot
block. The real change is the Q-fragment schedule.

* The asm build issues all 8 Q `buffer_load_b128` together, right after
  `s_wait_tensorcnt 0`.
* With the compiler-visible converter, the machine scheduler lowered VGPR pressure
  (106 → 91). It **sank each Q pair next to its WMMA pair** and reused `v[20:27]`.
* Result: **4 serial global-memory round trips per wave** (`buffer_load` → `s_wait_loadcnt 0x0`
  → WMMA), all of them after the TDM wait.
* Draft is latency-bound: one row per wave, about 2 waves per SIMD. So the extra round trips
  plausibly explain +1.3 µs.

The 13 v_nops come from the same register squeeze. The register allocator reuses
just-read WMMA A/B registers (for example Q `v[20:27]` as the next B), and LLVM must pad the
B1 WAR window (kernel-wiki
[`mi400-wmma-hazards`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-wmma-hazards.md)
§1.2 B1: 4 V_NOP or *independent* VALU).

Per kernel-wiki
[`mi400-valu-coexec`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-valu-coexec.md)
§2.2/§3, WMMA (XDL) co-executes with core/side-MACC VALU. Hardware does not interlock
those pairs, so the right filler is the *independent* conversion VALU of the next tile, not
v_nop.

## What changed (kernel)

### Draft fp8 schedule: `QSA_DRAFT_FP8_SCHED` (import time; intrinsic converter only)

| value | description |
|---|---|
| `kpipe` (`2e20f23b` default; opt-in `_skpipe` since FP8PERF2) | Two changes:<br>1. **Q under TDM:** Q fragments are issued between the TDM gathers and `s_wait_tensorcnt 0`, the decode-group order. The wait is a side-effecting fence, so the scheduler cannot sink the loads past it, and the Q latency hides under the gather latency.<br>2. **Software pipeline:** `sched_barrier(0)`-fenced regions `[cvt tile 0] [WMMA tile 0 ‖ cvt tile 1] … [WMMA tile n-1]`. Tile t+1's conversion temps are distinct live ranges from tile t's A/B/D, so no padding is needed, and the conversion co-executes with the WMMA chain. |
| **`qpre`** (default since FP8PERF2) | Change 1 only; LLVM interleaves freely. VGPR 69, 16 v_nop. |
| `kall` | Change 1, then every K fragment of the slab is converted before a `sched_barrier`, then one 8-WMMA burst (two independent accumulator chains). Fewest instructions (796) and 2 v_nop, but VGPR 172 and no conversion/WMMA co-execution. |
| `base` | `5fb8740e` order. The instruction stream is identical to `5fb8740e` apart from the name. |

* Non-default values tag the kernel name `_s<variant>`. The default name is unchanged.
* bf16 always uses `base`, so it is byte-identical.
* With `QSA_FP8_CVT_ASM=1` the knob is ignored (`base`), so the asm opt-out stays
  byte-identical to `5fb8740e`'s.
* **Q pre-issue effect on raw draft:** `raw` (opt-in, or used for unaligned caches) has no TDM
  wait, so `qpre` is identical to `base` there.

### Decode-group fp8 schedule: `QSA_DECG_FP8_SCHED`

| value | description |
|---|---|
| `kpipe` (`2e20f23b` default; opt-in `_skpipe` since FP8PERF2) | The same pipeline for `_pass_values`. Q was already issued before the tensor wait. |
| **`base`** (default since FP8PERF2) | `5fb8740e` structure. Stream-identical to `5fb8740e` for w32 and for w64. |

Non-default values tag the name `_sbase`.

### Per-dtype decode-group COLS

* `QSA_DECODE_GROUP_COLS` now defaults to **`auto`**: bf16 w32, **fp8 w64**.
* `32` or `64` still force both dtypes.
* With `QSA_FP8_CVT_ASM=1`, fp8 is still w32.
* The default fp8 decode kernel is now `…_fp8_decg_…_w64_flydsl_tdm`. That name is identical
  to `5fb8740e`'s COLS=64 kernel, but the ISA differs (kpipe). The fp8-r16 16×4 shapes route
  to it too.

### Correctness semantics

Unchanged in every variant:

* the same converter;
* the same Q/K fragment per (tile, step);
* the same per-tile accumulation order s = 0..3;
* the same epilogue and masks.

Only the instruction schedule changes. The new test class `FP8Sched` executes the actual
nested sources (`_tile_k`, `_tile_mma`, `_values`; decg `_pass_values`) with symbolic fx for
all variants and for 2 and 4 tiles. It checks that:

* the outputs (full dataflow expressions) are identical to `base`;
* the multiset of conversion and WMMA events is identical;
* the region order matches the intended schedule (cvt|, cvt mma|, …, mma|).

## ISA evidence (`fp8perf_isa_evidence.json`, `FP8PERF_ISA_OK`)

* **Setup:** 142 real-wrapper compiles; a07-3 container
  `rocm/fw-bringup-mlperf:gfx1250-atom-20261015` (`sha256:d1434db7…`); `--network none`;
  no devices.
* **(a) Identity:** 38/38 non-draft/decg-fp8 kernels are byte-identical to `5fb8740e`. That
  covers:
  * every bf16 kernel, including bf16 COLS=32/64 decode;
  * fp8 r32 tdm cvtfast and raw;
  * fp8 scalar;
  * legacy env.
* **(c) Hazards:** **0 violations in all 72 kernels compiled from the final source**
  (default, legacy and every variant), with zero tolerance for B1 and A1.
* **(e) asm opt-out:** 8/8 kernels are byte-identical to `5fb8740e`'s `QSA_FP8_CVT_ASM=1`
  builds, and the schedule knobs are ignored under asm.

In the table below, `nd` means instructions excluding v_nop and s_delay_alu. All kernels are
i32; qp64 kernels have the same deltas.

| kernel | instr | nd | VGPR | SGPR | v_nop | s_delay_alu | hazards |
|---|---|---|---|---|---|---|---|
| draft fp8 tdm w32, asm (`_cvtasm`) | 798 | — | 106 | 80 | 1 | 51 | 15 latent |
| draft fp8 tdm w32, `5fb8740e` | 811 | 740 | 91 | 66 | 13 | 58 | 0 |
| **draft fp8 tdm w32, default kpipe** | **820** | 748 | **115** | 80 | **1** | 71 | **0** |
| draft `_sqpre` | 833 | 743 | 69 | 80 | 16 | 74 | 0 |
| draft `_skall` | 796 | 740 | 172 | 80 | 2 | 54 | 0 |
| draft fp8 raw w64, `5fb8740e` | 944 | 895 | 148 | 66 | 13 | 36 | 0 |
| draft fp8 raw w64, kpipe | 969 | 899 | 156 | 66 | 6 | 64 | 0 |
| draft fp8 tdm w64 (COLS=64), base | 1474 | 1333 | 163 | 98 | 19 | 122 | 0 |
| draft fp8 tdm w64 (COLS=64), kpipe | 1441 | 1329 | 154 | 104 | 4 | 108 | 0 |
| decg fp8 w32, `5fb8740e` default | 3170 | 2873 | 115 | 96 | 45 | 252 | 0 |
| decg fp8 w32, kpipe (COLS=32) | 3160 | 2884 | 124 | 96 | 17 | 259 | 0 |
| decg fp8 w64, `5fb8740e` COLS=64 (GPU 26.6/14.7) | 5774 | 5347 | 229 | 105 | 31 | 396 | 0 |
| **decg fp8 w64, default kpipe** | 5908 | 5370 | **145** | 105 | **15** | 523 | **0** |

Notes on the table:

* In draft kpipe the 8 Q loads are issued before `s_wait_tensorcnt 0x0`, with a cascading
  `s_wait_loadcnt 0x7/0x5/0x3/0x1`.
* The tile-0 WMMAs are followed by tile-1 conversion VALUs (co-execution), and the 4 tile-1
  WMMAs come last.
* The kpipe instruction growth is almost entirely `s_delay_alu`, which are dependency hints.

## CPU gates (`container_fp8perf_cpu.sh`; same container, no devices)

| mode | env | result |
|---|---|---|
| cpu-default | (none) | **19/19 exit 0** |
| cpu-legacy | `QSA_DECODE_GROUP=0 QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0` | **19/19 exit 0** |

Changes to `test_decg_cpu.py` (28 → 33 tests):

* per-dtype COLS defaults: `test_default_on_all` and the new `test_per_dtype_default_cols`;
* `test_knobs`: fp8 → 64;
* the `sched_barrier` allowance in `test_builder_signature_and_seam`;
* the new class `FP8Sched`, with 4 tests:
  * the knobs, their defaults, invalid values and the asm override;
  * the name tags, and that bf16 is pinned to `base`;
  * the Q-before-wait order in `_wait_and_q`;
  * the executed dataflow equivalence of all draft and decg schedules.

## GPU plan (root, a08-1, gpu-lock)

**Use a fresh `FLYDSL_RUNTIME_CACHE_DIR` per tag.** The default names of draft fp8
(`…_fp8_draft_…_w32_flydsl_tdm`) and decg fp8 w64 are unchanged, but their ISA changed.

1. **`fp8perf-default`** (no env):
   * `run_gpu.sh correctness`, expect 22/22, then the pad 13 shapes.
   * Perf: draft fp8 c32768/c65536, against ≤ 12.2 / 21.3 µs and asm 11.6 / 19.3 µs.
   * Perf: decode fp8 c32768/c65536, against `5fb8740e` COLS=64 14.7 / 26.6 µs.
   * Perf: decode bf16 sanity (unchanged, 15.4 / 27.0 µs).
2. **Draft A/B** (perf on draft fp8 c32768/c65536, plus one correctness run per env):
   * `QSA_DRAFT_FP8_SCHED=qpre`
   * `QSA_DRAFT_FP8_SCHED=kall`
   * `QSA_DRAFT_FP8_SCHED=base` (equals `5fb8740e`)
3. **Decode A/B:**
   * `QSA_DECG_FP8_SCHED=base`: this is exactly `5fb8740e` COLS=64 (26.6 / 14.7). If it beats
     kpipe, flip `_DECG_FP8_SCHED_DEFAULT`.
   * `QSA_DECODE_GROUP_COLS=32`: fp8 w32 kpipe.
4. **Pick the defaults:** make whichever schedule wins the default by editing
   `_DRAFT_FP8_SCHED_DEFAULT` / `_DECG_FP8_SCHED_DEFAULT`. That is a one-line change, and the
   tags follow automatically.

## GPU decision 2026-11-03 (ROUND-FP8PERF2)

Root GPU A/B on `2e20f23b` (a08-1 GPU3, perftest graph replay, µs). Correctness: 22/22 for
default, `qpre` and `kall`; pad 13 PASS.

| env | draft fp8 c65536 | draft fp8 c32768 | decode fp8 c65536 | decode fp8 c32768 |
|---|---|---|---|---|
| default (draft kpipe, decg kpipe w64) | 20.6 | 11.9 | 26.8 | 16.0 |
| `QSA_DRAFT_FP8_SCHED=qpre` | **18.4** | **11.6** | 26.1 | 16.0 |
| `QSA_DRAFT_FP8_SCHED=kall` | 25.0 | 14.2 | 26.5 | 15.9 |
| `QSA_DECG_FP8_SCHED=base` (w64) | 20.6 | 12.0 | 27.0 | **14.7** |
| `QSA_DECODE_GROUP_COLS=32` | 21.0 | 12.0 | 28.1 | 15.3 |

Decision (commit `5e01dd8c`):

* `_DRAFT_FP8_SCHED_DEFAULT = "qpre"`: it is faster than kpipe at both sizes, and it beats the
  asm build (19.3 / 11.6 µs).
* `_DECG_FP8_SCHED_DEFAULT = "base"`: the fp8 decode stays w64. It is 1.3 µs faster at c32768
  and 0.2 µs slower at c65536.
* `kpipe` and `kall` stay as opt-ins. Draft `base` and decode-group `kpipe` are opt-ins too.
* Tags mark non-default choices only. Default names carry no `_s` tag. The opt-ins are named
  `_skpipe`, `_skall` and `_sbase` for draft, and `_skpipe` for decode-group.
* The kernel comment edits are line-count neutral, so `.loc` lines and bf16 bytes are unchanged.

The tables in the sections above describe the `2e20f23b` defaults (kpipe). Read "default
kpipe" there as the `_skpipe` opt-in now.

CPU/ISA evidence, from `host_fp8perf_cpu.sh STAGE {cpu-default|cpu-legacy|isa2}` with
`STAGE=/data/blyu/qsa/work/fp8perf2-cpu-trace-20261103` (`final` = `5e01dd8c`, `base` =
`2e20f23b`), is recorded in `fp8perf2_isa_evidence.json`. `isa2` gates the following
(`summarize_fp8perf2_isa_cpu.py`, `FP8PERF2_ISA_OK`):

* Every bf16, legacy, COLS and non-draft/decg kernel is byte-identical to `2e20f23b`.
* The default draft fp8 ISA equals `2e20f23b` `QSA_DRAFT_FP8_SCHED=qpre`, and the default decg
  fp8 ISA equals `2e20f23b` `QSA_DECG_FP8_SCHED=base`. Equality is full ISA text after
  substituting the kernel name; the old names carried the `_sqpre` / `_sbase` tag.
* The `_skpipe` opt-ins equal the `2e20f23b` defaults, modulo name.
* The kall/base opt-ins and the asm opt-out are byte-identical.
* There are 0 WMMA hazards.

### Results

* **Kernel sha:** `bf49226206b999f948e8b1eb0605935157b3d291b0a0b243581897d8ca661b93`. The base
  `2e20f23b` sha is `e407fa15…`.
* **Container:** `rocm/fw-bringup-mlperf:gfx1250-atom-20261015` (`sha256:d1434db7…`) on a07-3,
  with `--network none` and no devices.

| gate | result |
|---|---|
| cpu-default | **19/19 exit 0** (`test_decg_cpu.py` 33/33) |
| cpu-legacy | **19/19 exit 0** |
| isa2 | **`FP8PERF2_ISA_OK`**, 164 compiles |

The `isa2` result breaks down as follows:

* 38/38 byte-identical. This covers bf16, legacy, COLS, fp8 r32/scalar, and `fp8-r32-tdm`.
* 14/14 new defaults equal the `2e20f23b` `_sqpre` / `_sbase` ISA. The cases are draft tdm/raw/w64
  and decg w64 (decode, r16 tdm/raw) plus COLS=32 w32, each ×{i32, qp64}.
* 18/18 opt-ins pass:
  * `_skpipe` equals the `2e20f23b` default;
  * `_skall` and `_sbase` are byte-identical.
* 8/8 asm opt-outs are identical.
* There are **0 WMMA hazards in all 70 final-source kernels**.

Default kernels (i32):

| kernel | instr | VGPR | v_nop |
|---|---|---|---|
| draft fp8 tdm w32 (= old `_sqpre`) | 833 | 69 | 16 |
| draft fp8 raw w64 | 944 | 148 | 13 |
| decg fp8 w64 (= old `_sbase` = `5fb8740e` COLS=64) | 5774 | 229 | 31 |

## Reproduce

Run `host_fp8perf_cpu.sh STAGE {cpu-default|cpu-legacy|isa}` with
`STAGE=/data/blyu/qsa/work/fp8perf-cpu-trace-20261103`. It needs:

* `final/source`: this commit;
* `base/source`: `5fb8740e`.

`container_fp8perf_quick_cpu.sh` is a quick ISA/hazard loop over arbitrary
`phase|label|dtype|shape|env` lines.
