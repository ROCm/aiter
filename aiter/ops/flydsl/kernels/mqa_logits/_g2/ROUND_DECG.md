# ROUND-DECG — opt-in decode row-group kernel (`QSA_DECODE_GROUP`)

Base: champion `c45b24e0` (kernel sha `944a42b8…`). Branch `qsa-index-scorer/opt-decg`.
CPU/no-device evidence only. **No GPU timings exist for this round.** Default behaviour is
unchanged: with the knob unset, all 18 specializations × {int32, pos64} compile to ISA that is
byte-identical to the base (36/36).

## Knobs (read once at import)

| env | values | effect |
|---|---|---|
| `QSA_DECODE_GROUP` | `0` / `1` = `bf16` / `fp8` / `all` (**default since ROUND-DECG2**; was `0`) | route the enabled dtype(s) to `_build_decode_group_kernel` when rows > requests, rows % 4 == 0, the champion would pick r16 (rows < 32·requests), raw access, the dtype's WMMA path, and a 128 B-aligned cache base |
| `QSA_DECODE_GROUP_COLS` | `32` (default) / `64` | columns per wave slab (one WG = 4 rows × 4 slabs); since ROUND-DECG2 `64` applies to bf16 only — fp8 always w32 |

## Why

Decode bf16 (64 rows / 16 requests) runs `kernel_tdm_r16_nobar`. Each wave owns 4 rows × 32
columns and runs 4 request-dedup passes, but for decode 3 of those passes are empty. Each empty
pass still does all of the following:

* re-loads the 4 rows' t2r/pos/ctx through `s_buffer_load` (`_r16_stage_active`);
* waits on TENSORcnt and DScnt;
* evaluates the VGPR pass plan.

The nobar r16 int32 ISA has 3382 instructions.

## Design (`_build_decode_group_kernel`, kernel `…_{bf16,fp8}_decg_D128_H4_P4_w{32,64}_flydsl_tdm[_qp64]`)

* **Geometry.** One WG covers 4 consecutive rows (`grid.y = ceil(rows/4)`) × 4 waves, and each
  wave owns a private `wave_cols` slab (`grid.x = ceil(columns/(4·wave_cols))`). For c32768 the
  grid is (256, 16) with w32 or (128, 16) with w64.
* **Metadata.** All metadata is SGPR: `s_buffer_load` reads t2r, ctx and pos for each of the 4
  rows, and positions go through `_pos_loaders(pos64)`. The ISA has 0 `v_readfirstlane`.
  `same` = the 4 clamped rows have one request id. This is a uniform SCC test.
* **Single pass when `same` (decode):**
  * `vmax = max visible`. If `col0 ≥ vmax`, the wave stores −inf only.
  * Otherwise the wave issues `wave_cols/8` TDM 8-row gathers with SGPR descriptors into its
    private padded tile, loads the Q fragments while the gathers are in flight, waits with
    `s_wait_tensorcnt 0`, then runs the WMMA and stores all 4 rows.
  * There is no WG barrier: TENSORcnt is per wave and decrements on tensor-done of that wave's
    op, so this wave's own wait is enough before it reads its own tile. Kernel-wiki
    [`mi400-tdm-sync`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-tdm-sync.md)
    lines 50-59 cover this, including "never conflate XACK with completed LDS visibility".
* **Mixed groups (any `token_to_request`).** Passes p = 1..3 are enabled. Pass p owns row p
  alone, uses request `req_p` and `vmax = visible[p]`. Pass 0 owns only row 0. After each live
  pass, `s_wait_dscnt 0` is the WAR guard before the next pass's gather reuses the tile. This is
  the nobar r16 rule; see kernel-wiki
  [`mi400-memory-ordering-barriers`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-memory-ordering-barriers.md).
  Every row is stored exactly once in both regimes, including partial last groups (`row_ok`
  masks).
* **Per-slot math is the champion r16 tile:**
  * A of lane (n_in, kgrp) is `Q[t0 + n_in//4, head n_in%4]`.
  * C elements 0..3 / 4..7 are the heads of rows 2·kgrp / 2·kgrp+1.
  * bf16 post-processing is `(r0+r1)+(r2+r3)`; fp8 is `relu(a_h·scale)` summed left to right.
    Both are then multiplied by `1/divisor`.
  * Masks are `page_ok & c < visible[row]`.
  * Stores go to the r16 clamped in-plane offsets (`_store_offsets` is textually the r16
    helper).
  * fp8 K uses the draft kernel's GPU-verified exact pk8 converter, which gives the same bf16
    values as the champion converter.
  * So the outputs should be bitwise equal to the default kernels. GPU `torch.equal` will
    confirm this.
* **LDS** = 4 · wave_cols · pitch: bf16 34,816 B (w32) / 69,632 B (w64); fp8 18,432 / 36,864 B.
  See kernel-wiki
  [`mi400-occupancy-resources`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-occupancy-resources.md)
  for the LDS and VGPR residency limits.

## Evidence (a07-3 container `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, `--network none`, no devices)

* **CPU gates.** `host_decg_cpu.sh STAGE cpu-default`, kernel sha `70e6bed1…`. All 17 existing
  gates plus the new `decg` gate exit 0; see `decg_cpu_evidence.json`. One explicit gate edit:
  `test_draft_cpu.test_module_wide_position_seam_census` now counts 11 QP descriptors instead of
  10. The new one is the decg kernel's, and it is also sized with `qp_bytes`.
* **`test_decg_cpu.py` (19 tests):**
  * routing and knobs: default off; bf16-only for `1`; fp8 and all; draft, prefill, rows%4 and
    alignment exclusions;
  * the wrapper AST: a flag-gated block after the draft block and before the original branches;
  * the real wrapper with `_run` captured: default → nobar; opt-in → decg with `pos64`, w32 and
    grid (cols/128, rows/4);
  * source gates: seam, SGPR-only metadata, uniform TDM descriptors, tensor_wait → values →
    dscnt order, no barrier;
  * champion-identical helpers;
  * the pass plan for decode, mixed and partial groups;
  * exactly-once stores for w32/w64 over rows {4, 8, 10, 64} × 8 column tails × {decode, mixed,
    random t2r};
  * lane-level emulation of the A/B/C layouts against the contract formula (rtol 1e-12) **and**
    bitwise against an r16-pass emulation. The emulation covers bf16 and fp8, w32 and w64,
    decode, mixed (incl. invalid requests), and partial groups.
* **ISA** (`host_decg_cpu.sh STAGE isa`, 80 real-wrapper compiles, `decg_isa_evidence.json`):
  * (a) Default env == base `c45b24e0` for 18 specs × {i32, qp64}. That includes the 64×16
    decode routes (bf16 → `_tdm_nobar`, fp8 → r16 raw c64). The raw sha, normalized sha, name and
    resources all match.
  * (b) The decg kernels:

    | kernel | instr i32 → qp64 | VGPR | SGPR | LDS | scratch | s_barrier | tensor_load | wmma | v_readfirstlane |
    |---|---|---|---|---|---|---|---|---|---|
    | bf16 w32 | 1981 → 2018 | 97 | 92 | 34,816 | 0 | 0 | 16 | 32 | 0 |
    | bf16 w64 | 3481 → 3519 | 193 | 105 | 69,632 | 0 | 0 | 32 | 64 | 0 |
    | fp8 w32 | 3126 → 3159 | 97 | 96 | 18,432 | 0 | 0 | 16 | 32 | 0 |
    | fp8 w64 | 5720 → 5752 | 192 | 105 | 36,864 | 0 | 0 | 32 | 64 | 0 |

    The static counts include the 4 unrolled mixed-group passes. The decode path executes only
    pass 0: 4 (w32) or 8 (w64) gathers and 8 or 16 WMMA. The nobar r16 bf16 kernel is 3382 int32
    instructions, 97 VGPR and 105 SGPR. w64 needs about 193 VGPRs, which lowers occupancy, so w32
    is the default.

## GPU plan (a08-1 GPU3, gpu-lock, run_perftest graph replay, golden int64 positions)

1. **Correctness.** Run all 22 shapes with `QSA_DECODE_GROUP=all` and with
   `QSA_DECODE_GROUP=all QSA_DECODE_GROUP_COLS=64`. Compare against the default output with
   `torch.equal` on logits and visible_groups (bitwise is expected), then against golden. Add one
   synthetic mixed-request decode case: rows 64, t2r shifted by one row, plus some −1 entries.
2. **Perf A/B.** Compare default against `QSA_DECODE_GROUP=1` (w32) and
   `QSA_DECODE_GROUP=1 QSA_DECODE_GROUP_COLS=64` on decode bf16 c32768 and c65536. Then do the
   same with `QSA_DECODE_GROUP=fp8` on decode fp8 c32768 and c65536.
3. If w32 lands ≤ 15.5 µs on bf16 c32768 and no shape regresses, flip the default for bf16 to
   `1`. Do the same for fp8 if it wins.

# ROUND-DECG2 — default on (`QSA_DECODE_GROUP=all`), fp8 never w64

GPU input (a08-1 GPU3, root-measured, 2026-11-03):

* `QSA_DECODE_GROUP=all` (w32) is 22/22 correct. Decode is 0.64–0.74× ATOM:
  * bf16 c32768: 16.6 → 15.3 µs; c65536: 27.8 → 27.1 µs.
  * fp8 c32768: 16.7 → 15.0 µs; c65536: 29.4 → 26.5 µs.
* With `QSA_DECODE_GROUP_COLS=64`, **fp8 is wrong**. Failing shapes: I5_min_decode_fp8, I6_2req_fp8,
  tail_columns_1000_fp8, edge_bad_page_fp8, I2_e2e_decode_fp8. bf16 w64 is correct but slower.

## Changes (kernel sha `30c6213d…`)

* **Default flipped.** `QSA_DECODE_GROUP` now defaults to `all`. `0` opts out to the exact
  c45b24e0 routing; the ISA proves this, see below. `QSA_DECODE_GROUP_COLS` still defaults to
  `32`.
* **fp8 w64: GPU-incorrect (root run 2026-11-03), disabled.**
  * `decode_group_route` returns `_DECODE_GROUP_FP8_COLS = 32` for fp8 whatever
    `QSA_DECODE_GROUP_COLS` is.
  * `_build_decode_group_kernel(is_fp8=True, wave_cols=64)` raises `ValueError`.
  * fp8 keeps the GPU-correct w32 kernel and is never disabled. bf16 still honours COLS=64.
* **Behaviour note.** The default decode-group route also takes r16-regime shapes that run with
  `QSA_BF16_K_LOAD=raw` / `QSA_FP8_K_LOAD=raw`. Those K_LOAD knobs apply only to the WMMA
  kernels, so use `QSA_DECODE_GROUP=0` to reach them. This was already true of the opt-in in
  bbe25c33.
* **Docstrings and comments.** The Round DECG block, `decode_group_route`, the builder and the
  wrapper comment all say "default 'all'".

## CPU gate edits (each stated)

* **`test_decg_cpu.py`:**
  * `test_default_off` became `test_default_on_all`, plus a new `test_opt_out_zero` for `0`.
  * New `test_fp8_never_w64` covers route and builder guard for COLS=64 under default, `all`
    and `fp8`.
  * In `test_knobs`, all+COLS=64 fp8 now gives 32.
  * The wrapper launch test now expects default → decg, `0` → nobar, `1` → decg.
  * 21 tests in total.
* **`test_launch_cpu.py`:**
  * `_build_decode_group_kernel` is recorded alongside the WMMA builders.
  * The expected builder is decg when `QSA_DECODE_GROUP` is not `0` **and**
    `decode_group_route(...)` accepts the fixture. The route needs a 128 B cache base, and CPU
    allocations are only 64 B-aligned.
  * A decg launch shifts metadata indices by +1 because of its KS slot.
  * All launch-argument identity checks now also cover the decg launch.
* **`test_pad_bucket_cpu.py`:**
  * `model_run` accepts the `{bf16,fp8}_decg` argument layout (with KS).
  * `_build_decode_group_kernel` is stubbed like the WMMA builders. A cold real build reads
    `FLYDSL_*` env and would break the getenv census.
  * The idx and arg offsets are now decg-aware.
  * The exact-vs-pad kernel check compares `(dtype_tag, geometry)`.
  * The native-bucket census pre-warms the per-device cached dummy KS plane `K._dummy_scale`
    (a one-time `aten::zeros`).
* **`test_pad_bucket_trace_cpu.py`:** pins `QSA_DECODE_GROUP=0`, the same way it already pins
  `QSA_DRAFT_PATH=0`. The trace executor instruments only the WMMA bodies. The decg stores and
  layouts are covered by `test_decg_cpu` (exactly-once plan plus lane emulation) and by the GPU
  22/22.
* **`test_nosync_i32_cpu.py`:**
  * `model_run` accepts the decg layout.
  * A decg builder stub records `pos64`.
  * Metadata indices are decg-aware.
  * The census pre-warms `K._dummy_scale`.
* `test_bf16_routing_cpu.py` and the other 12 gates are unchanged: they pin their own env or
  test helpers that do not depend on the decg default. All of them are green in both envs.
* `container_decg_cpu.sh` adds a `cpu-off` mode, which exports `QSA_DECODE_GROUP=0`, and a new
  DECG2 `isa` mode (150 compiles). `host_decg_cpu.sh` mounts `/prev` = bbe25c33.
  `summarize_decg2_isa_cpu.py` is new.

## Evidence (a07-3 container, `--network none`, no devices; `decg2_cpu_evidence.json`, `decg2_isa_evidence.json`)

* **CPU.** All 18 gates exit 0 in the default env and with `QSA_DECODE_GROUP=0` (`cpu-default` /
  `cpu-off`). Staging is `/data/blyu/qsa/work/decg2-cpu-trace-20261103`.
* **ISA** (150 real-wrapper compiles, `ALL_DECG2_ISA_OK`):
  * `QSA_DECODE_GROUP=0` matches base c45b24e0 for 18 specs × {i32, qp64}: raw sha,
    normalized sha, name and resources.
  * The default env matches bbe25c33 with `QSA_DECODE_GROUP=all` for all 36. The decode-routed
    specs (bf16/fp8 r16 tdm/raw 16×4, decode 64×16) are the DECG w32 kernels:
    * bf16 i32 `9730c7a2…` / qp64 `3bfb460b…`;
    * fp8 i32 `40651506…` / qp64 `612b8e5c…`.
    These are the same shas as the bbe25c33 opt-in round. The other 10 specs × 2 also equal
    base.
  * COLS=64: bf16 is the w64 kernel, identical to bbe25c33 w64. fp8 is the **w32** kernel,
    identical to bbe25c33 fp8 w32.

## fp8 w64 bug — notes only (no fix shipped, no GPU evidence)

The coarse ISA ordering of the bbe25c33 fp8 w64 kernel looks the same as w32 and bf16 w64:

* all 8 `tensor_load_to_lds` and their `s_wait_tensorcnt` come before the first `ds_load`, so
  no LDS read is hoisted above the TENSORcnt wait;
* there is no `s_barrier` and no scratch.

What is specific to fp8 w64:

* 18 outstanding `ds_load`s, then a `s_wait_dscnt` cascade before the first WMMA;
* 192 VGPRs with the pk8 converter inlined, for 4 N-tiles.

Candidates to check on GPU:

* a wrong `s_wait_dscnt`/`s_wait_loadcnt` immediate in that cascade, or a `v_perm` lane-select
  error under register pressure;
* the scale `_ld_f32` (per tile, VMEM) being consumed before its `s_wait_loadcnt`;
* a TDM fp8 gather of more than 4 groups per wave hitting a descriptor field limit. The LDS
  byte offset is ≤ 35,712 B, well inside the bf16 w64 range, which is correct.

Suggested bisect:

* fp8 w64 with `n_tiles` forced to 2, i.e. 32 columns computed from a 64-column tile;
* fp8 w64 with the gather loop limited to 4 groups.