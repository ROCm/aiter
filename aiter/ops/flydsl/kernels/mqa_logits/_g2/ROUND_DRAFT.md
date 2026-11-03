# Round DRAFT — one-row-per-wave draft kernel + barrier-free r16 TDM (opt-in)

Base champion `8a8d368a`. CPU/no-device evidence only. **No GPU timings exist for this round.**
Default behaviour is unchanged: the default-env ISA of all 7 champion kernels is byte-identical
to the base ISA.

> **ROUND-MERGE1 update:** after GPU verification (a08-1 GPU3, 22/22 correct) the defaults
> are now `QSA_DRAFT_PATH=1` (`QSA_DRAFT_K_LOAD=tdm`, `QSA_DRAFT_COLS=auto` = 32 for TDM) and
> `QSA_R16_NOBAR=1`. `QSA_DRAFT_PATH=0` / `QSA_R16_NOBAR=0` opt out to the champion kernels
> (ISA-identical to NATIVE64). Both kernels gained `_qp64` (int64 query_positions) variants.
> See `ROUND_MERGE1.md`. The table below records the round as originally landed (opt-in).

## Knobs (read once at import, all default off at the time of this round)

| env | values | effect |
|---|---|---|
| `QSA_DRAFT_PATH` | `0` / `1` | `1`: when `rows <= requests` (draft), use the dtype's WMMA path with raw access: route bf16 and fp8 to `_build_draft_kernel` |
| `QSA_DRAFT_K_LOAD` | `tdm` (default) / `raw` | K via wave-private TDM tile, or per-lane buffer loads (TDM falls back to raw if the cache base is not 128 B aligned) |
| `QSA_DRAFT_COLS` | `auto` / `32` / `64` | columns per wave (auto: 32 for TDM, 64 for raw); one WG = 1 row x 4 waves |
| `QSA_R16_NOBAR` | `0` / `1` | `1`: bf16 r16 TDM kernel (decode/draft) without its 2 WG barriers per pass (`kernel_tdm_r16_nobar`, name suffix `_nobar`) |

## Why

* Draft rows each have their own request, so each r16 wave (4 rows) runs 4 serial dedup passes,
  and each pass stores 1 of its 4 WMMA rows.
  * bf16 r16 TDM c32: every pass runs a gather, `s_wait_tensorcnt`, a WG barrier, WMMA, and
    another WG barrier. That adds up to 16 TDM gathers and 16 barriers per WG.
  * fp8 r16 uses raw loads and the 4-byte converter: 148 VGPRs, 5533 static VALU, 1024
    `v_cvt_f32_f16`.
* At c32768 the grids are tiny:
  * fp8 r16 c64: grid (512,1) = 2048 waves, which is 0.5 waves per SIMD on 1024 SIMDs.
  * bf16 r16 TDM c32: grid (1024,1) = 4096 waves.
* The draft kernel gives one wave one row and a 32/64-column slab:
  * The request, visible count and page table are all SGPR loads.
  * A slab with `col0 >= vis` writes -inf only.
  * TDM gathers go into a wave-private tile, so `s_wait_tensorcnt 0` is enough and no barrier is
    needed (kernel-wiki `mi400-tdm-sync`: data is in LDS after tensor-done, and
    `S_WAIT_TENSORCNT` waits for it).
  * fp8 uses the GPU-verified `v_cvt_scale_pk8_bf16_fp8` exact converter.
  * Grid at c32768: (256,16) = 16384 waves.
* Exactness: each WMMA slot sees the same Q head, K row and k-steps as the champion r16 tile.
  Post-processing follows the champion order per dtype:
  * bf16: `(r0+r1)+(r2+r3)`;
  * fp8: `relu(a*scale)` summed left-to-right;
  * then `*1/divisor`.
* Masks: the -inf masks are the same as the champion's. Lane `l` stores column `col0+32j+l`
  exactly once.

## Evidence

* `test_draft_cpu.py` (14 tests) covers:
  * routing and knobs;
  * the AST placement of the wrapper block;
  * exactly-once store coverage, including tails;
  * a lane-level emulation of the WMMA A/C layouts against the contract formula, with invalid
    requests and pages, tails and visibility;
  * `kernel_tdm_r16_nobar` == `kernel_tdm_r16` except for the 2 barrier hooks;
  * wave-private tile offsets.
* `compile_draft_cpu.py` + `host_compile_draft_cpu.sh` run in the no-device container
  (`--network none`, no `/dev/kfd`, FlyDSL 0.3.3.dev886). They compile and dump the ISA for all
  8 draft variants and the nobar kernel, then run every existing CPU gate.

## GPU plan (a08-1 GPU3, run_perftest graph replay, 4 rotated copies)

1. Correctness: for each shape, compare the opt-in output with the default output using
   `torch.equal` on logits and visible_groups. Bitwise equality is expected. Then compare both
   against golden.
2. Perf A/B per shape:
   * draft fp8/bf16, 16/16, c32768 and c65536:
     * `QSA_DRAFT_PATH=1` (TDM w32);
     * `QSA_DRAFT_PATH=1 QSA_DRAFT_K_LOAD=raw` (w64);
     * `... QSA_DRAFT_COLS=64` (TDM w64);
     * `... K_LOAD=raw QSA_DRAFT_COLS=32`.
   * decode bf16 64/16 c32768 and c65536: `QSA_R16_NOBAR=1`. Also check the bf16 draft shapes
     when `QSA_DRAFT_PATH=0`.
