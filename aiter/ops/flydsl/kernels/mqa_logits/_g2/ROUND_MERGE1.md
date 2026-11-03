# ROUND-MERGE1: NATIVE64 + DRAFT merged, DRAFT/NOBAR on by default

Base: NATIVE64 `60b9aade` (on champion `8a8d368a`). Merged branch `qsa-index-scorer/opt-draft`
(`f6a54cc9`). Branch `qsa-index-scorer/opt-merge1`. CPU/no-device evidence only. GPU has not
been run on this merge (it has been run on each input separately).

## Merge

* There was one textual conflict: the bf16 WMMA builder's kernel-name suffix. Resolution:
  `..._tdm` + `_nobar` (if `r16_nobar`) + `qp_tag`. The int32 names are unchanged and the pos64
  name is `..._tdm_nobar_qp64`.
* **Draft builder through the NATIVE64 seam.** `_build_draft_kernel(..., pos64=False)` now does
  three things:
  * it takes its loaders from `_, ld_uniform_pos, qp_bytes, qp_tag = _pos_loaders(pos64)`;
  * it builds the QP descriptor as `_plane(QP, _plane_bytes(rows, qp_bytes))`;
  * it reads the SGPR position with `pos = ld_uniform_pos(qp_rsrc, safe_row)`. The kernel name
    gets `+ qp_tag`.

  The wrapper calls `_build_draft_kernel(is_fp8, float(divisor), draft[0], draft[1], **qp_kw)`.
  int32 callers pass no keyword, so they hit the same lru_cache entry. int64 positions take the
  `_qp64` draft kernel and the zero-copy `[2*rows]` int32 view, with no clamp launch.
* **Merge bug fixed: nobar QP extent.** `kernel_tdm_r16_nobar` came from the DRAFT round and
  predates the seam. Its body still built the QP descriptor with `rows * 4`. For pos64 that
  extent would have cut off the 8-byte position reads of the upper half of the rows (the loads
  return 0). It now uses `qp_bytes`, as the barriered `kernel_tdm_r16` does. The loads already
  went through the shared `_rows_meta_uniform` / `_r16_stage_active` seam helpers. A new gate
  (`test_draft_cpu.DraftPos64.test_module_wide_position_seam_census`) pins that all 10 QP
  descriptors use `qp_bytes` and that no raw `* 4` position load remains.

## Defaults (import-time, as before)

| env | old | new | opt-out |
|---|---|---|---|
| `QSA_DRAFT_PATH` | `0` | **`1`** (`QSA_DRAFT_K_LOAD=tdm`, `QSA_DRAFT_COLS=auto` → 32 for TDM; misaligned cache bases fall back to raw w64) | `0` → champion r16 routing |
| `QSA_R16_NOBAR` | `0` | **`1`** (bf16 r16 TDM decode kernel without its 2 WG barriers) | `0` → barriered `kernel_tdm_r16` |

* The draft routing is unchanged: only `rows <= requests`, raw access, and the dtype's WMMA path.
* NOBAR still only replaces the r16 TDM non-persistent kernel.
* The CPU AST gates that exec `_build_wmma_kernel` in reduced namespaces with no
  `TDM_R16_NOBAR` binding still see the barriered kernel (`globals().get(..., False)`).

## CPU gate changes (explicit)

* `test_draft_cpu.py`:
  * `test_default_off` became `test_default_on`: the default routes `(tdm, 32)` for
    rows <= requests and `None` for 64/16.
  * New `test_opt_out`: `QSA_DRAFT_PATH=0` routes `None` everywhere.
  * `NoBarrier.test_default_off_and_knob` became `test_default_on_and_knob`: default True,
    `"0"` gives False.
  * `test_selection_and_name` also pins `_nobar` before `qp_tag`.
  * New class `DraftPos64`, which covers four things:
    * the seam signature and source of the draft builder;
    * the int32 binding equals the DRAFT-round IR call;
    * the wrapper passes `**qp_kw`;
    * the module-wide QP census.
* `test_bf16_routing_cpu.py`:
  * `assert_selection` expects `kernel_tdm_r16_nobar` and the `_tdm_nobar` name when
    `TDM_R16_NOBAR` is set and the kernel is TDM r16.
  * New `test_r16_nobar_default_on_and_opt_out`: default True, `"0"` gives the champion kernel,
    and there is no nobar for raw r16 or r32.
* `test_pad_bucket_cpu.py`:
  * `model_run` accepts the draft argument layout (`bf16_draft` / `fp8_draft`).
  * The exact path asserts that the draft kernel is routed iff `K._DRAFT_PATH and rows <= reqs`.
  * The geometry equality is only checked for non-draft exact routes.
* `test_pad_bucket_trace_cpu.py` pins `QSA_DRAFT_PATH=0` at import, as it already pins
  `QSA_BF16_K_LOAD=raw`. Its trace executor only instruments WMMA bodies. The draft kernel is
  covered by the lane-level emulation in `test_draft_cpu.py` and by GPU correctness.
* No source-hash pins changed. `test_persist_cpu` pinned hashes are unaffected.

## Evidence (a07-3 container, `--network none`, no devices; kernel sha `944a42b8…`)

* **CPU gates** (17): contract, address, q-lds, trace16, trace32, k-tdm, routing, fp8-k-tdm,
  bm64, persist, fp8-cvt, native64, draft, pad-bucket, pad-bucket-trace, nosync-i32 and launch.
  All of them exit 0 in two environments:
  * the default env;
  * `QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0`.

  See `merge1_cpu_evidence.json`.
* **ISA** (`host_merge1_cpu.sh STAGE isa`, 87 real-wrapper compiles, `merge1_isa_evidence.json`):
  * (a) Opt-out == NATIVE64 `60b9aade`. The ISA sha, kernel name and resources are identical for
    all 12 specializations × {int32, pos64}.
  * (a2) The default env == NATIVE64 for 11 of the 12 specializations (both metas). bf16 r16 TDM
    becomes `_nobar` (and `_nobar_qp64`).
  * (b) The default int32 kernels are byte-identical to the DRAFT-branch opt-in kernels
    (`QSA_DRAFT_PATH=1 QSA_R16_NOBAR=1`). This holds for draft bf16/fp8 TDM w32, draft bf16/fp8
    raw w64, and nobar r16.
  * (c) pos64 deltas: VGPR, SGPR, LDS and scratch are unchanged everywhere.

    | kernel | instr i32 → qp64 | main change |
    |---|---|---|
    | draft bf16 TDM w32 | 514 → 523 (+9) | `s_buffer_load_b32` → `_b64` + SALU saturate |
    | draft fp8 TDM w32 | 798 → 808 (+10) | same |
    | draft bf16 raw w64 | 426 → 436 (+10) | same |
    | draft fp8 raw w64 | 927 → 937 (+10) | same |
    | nobar bf16 r16 TDM | 3382 → 3476 (+94) | same pattern as the barriered r16 TDM qp64 (+94): 8× `s_buffer_load_b64`, +20 `v_readlane_b32` (SGPR 105) |

## GPU plan (a08-1 GPU3, gpu-lock, run_perftest graph replay)

1. Correctness 22/22 in the default env with golden int64 positions. This now exercises
   `draft_*_qp64` and `..._nobar_qp64`, which no GPU has run yet. The nobar extent fix is the
   main reason to run it.
2. Correctness 22/22 with `QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0`. Expect the NATIVE64 numbers.
3. Perf for draft bf16/fp8 c32768/c65536 and decode bf16 c32768/c65536, comparing the default
   against NATIVE64 and DRAFT. Expected: draft gains stack with the −2.5…−6 µs clamp removal.
