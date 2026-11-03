# ROUND-W64FIX2 — compiler-visible pk8 converter by default, fp8 decode-group w64 back

Base: champion `a78c285c` (kernel sha `30c6213d…`). Predecessor: ROUND-W64FIX `eed2a4ed`.
Branch `qsa-index-scorer/opt-w64fix`. Final kernel sha `a3e2ae3f2561604b5acb70341d02a3ed97edede5bdb2bf100fe45acf5425994f`.
CPU/no-device evidence only. **No GPU run in this round.**

## Why

The root GPU run (a08-1 GPU3, run_gpu correctness, 22 cases) confirmed ROUND-W64FIX
hypothesis 1:

* `asis` fails 5 fp8 decode-group cases;
* `nop4` passes 22/22;
* `intrin` passes 22/22;
* `tdmwait` fails 6 cases.

Root cause: the WMMA software hazard (kernel-wiki `mi400-wmma-hazards` §1.2 B1). The
inline-asm VALU instructions of the pk8 converter are not padded by LLVM, so they can land
inside the 4-VALU window after a WMMA. The `intrin` build of fp8 w64 performed the same as
the default w32 build: decode fp8 c65536 26.8 vs 26.6 µs, c32768 14.8 vs 15.1 µs.

## What changed (kernel)

### Converter

* `_e4m3x8_to_bf16x4_fast` is now a dispatcher.
  * By default it calls `_e4m3x8_to_bf16x4_intrin`. The intrinsic converter uses no inline
    asm: `llvm.amdgcn.cvt.scale.pk8.bf16.fp8` with scale 0x7f and sel 0, `llvm.amdgcn.perm`,
    `llvm.umin.v2i16`, and and/or. It is the converter that passed 22/22 on GPU.
  * With `QSA_FP8_CVT_ASM=1`, read at import, it calls `_e4m3x8_to_bf16x4_asm`, the
    unchanged former inline-asm body.
* Every fp8 path that used the asm converter now goes through the dispatcher, because they
  all call `_e4m3x8_to_bf16x4_fast`. Static count: 4 call sites.
  * r32 TDM `_pair_frag_tdm_fast` (`…_r32_c64_flydsl_tdm_cvtfast`);
  * draft `_kfrag_fp8_raw` and `_kfrag_fp8_tdm`;
  * decode-group `_kfrag_fp8` (w32 and w64).
* No other code in the kernel file uses inline asm. Gate `test_wmma_hazard_scan_cpu.py`
  checks this.

### `QSA_FP8_CVT_ASM` opt-out

* The opt-out is kept for A/B and debugging only.
* Kernel names gain `_cvtasm`, so runtime caches cannot mix the two converters:
  `…_cvtfast_cvtasm[_qp64]`, `…_draft_…[_qp64]_cvtasm` and `…_decg_…[_qp64]_cvtasm`.
* Its instruction stream equals the champion's apart from the kernel name (ISA gate (e)).

### fp8 decode-group w64 re-enabled

* `_DECODE_GROUP_FP8_COLS` is removed. `decode_group_route` returns `QSA_DECODE_GROUP_COLS`
  for both dtypes. The default COLS stays **32**, so the default route does not change.
* With `QSA_FP8_CVT_ASM=1`, fp8 is routed to w32 and the builder rejects fp8 w64. That build
  is known to be GPU-wrong.

### Bisect knob removed

* `QSA_DECG_FP8_W64_DEBUG` (asis/nop4/intrin/tdmwait), the `_dbg*` names,
  `_e4m3x8_to_bf16x4_nop4` and the `tdmwait` hook are all removed, because the bisect
  answered its question.
* `container_w64fix_cpu.sh` and `summarize_w64fix_isa_cpu.py` are kept as the historical
  ROUND-W64FIX record. They need the predecessor source `eed2a4ed`.

## ISA evidence (`w64fix2_isa_evidence.json`, `W64FIX2_ISA_OK`)

* **Setup:** 104 real-wrapper compiles, a07-3 container `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`,
  `--network none`, no devices.
* **Identity:** 34/34 kernels are byte-identical to `a78c285c`: raw sha, normalized sha,
  name and resources. These are every kernel that does not use pk8:
  * all bf16 kernels: 18-spec bf16 rows, bf16 COLS=64, legacy-env bf16;
  * fp8 r32 raw and fp8 scalar;
  * the legacy-env (`QSA_DECODE_GROUP=0 QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0`) fp8 r16 decode
    and draft kernels.
* **pk8 kernels:** names, LDS, WMMA count, pk8 count and `tensor_load_to_lds` count are
  unchanged. Scratch is 0 and there are 0 inline-asm blocks (the champion had 208–832).

  | kernel (i32; qp64 has the same deltas) | instr | VGPR | SGPR | v_nop | hazards |
  |---|---|---|---|---|---|
  | fp8 decg w32 (`decode-fp8-64x16`; also `fp8-r16-{tdm,raw}` 16×4) | 3126→3170 (+44) | 97→115 | 96→96 | 46→45 | 28→**0** |
  | fp8 r32 tdm cvtfast (prefill) | 18551→18535 (−16) | 168→168 | 83→83 | 97→95 | 0→0 |
  | draft fp8 tdm w32 (default draft) | 798→811 (+13) | 106→91 | 80→66 | 1→13 | 15→**0** |
  | draft fp8 raw w64 (opt-in `QSA_DRAFT_K_LOAD=raw`) | 927→944 (+17) | 105→148 | 66→66 | 9→13 | 20→**0** |
  | **fp8 decg w64** (`QSA_DECODE_GROUP_COLS=64`) | 5774 (w32: 3170) | 229 | — | 31 | **0** |

  The fp8 decg w64 numbers match the GPU-correct ROUND-W64FIX `intrin` build (5774 / 229).
* **Hazard scan** (`wmma_hazard_scan_cpu.py`, B1 + A1, zero tolerance): **0 violations in
  all 48 scanned kernels**:
  * 18 specs × {i32, qp64} in the default env;
  * 4 legacy-env specs × 2;
  * COLS=64 bf16/fp8 × 2.
* **`QSA_FP8_CVT_ASM=1`:** the instruction stream equals the champion for all 8 pk8
  compiles. Its hazard counts are the latent ones (15/20/28; r32 0). COLS=64 fp8 then builds
  `…_w32_…_cvtasm`.

## CPU gates (`container_w64fix2_cpu.sh`; same container, no devices)

| mode | env | result |
|---|---|---|
| cpu-default | (none) | 19/19 exit 0 |
| cpu-legacy | `QSA_DECODE_GROUP=0 QSA_DRAFT_PATH=0 QSA_R16_NOBAR=0` | 19/19 exit 0 |

The 19 gates are the 18 DECG2 gates plus the new `wmma-hazard` gate,
`test_wmma_hazard_scan_cpu.py`. That gate checks the scanner's B1 WAR/WAW/RAW, the 4-VALU
window, that SALU does not count as padding, ASM tagging, A1, the branch cut and the
`--fail-on-any` exit code, all on synthetic ISA. It also checks that the only inline-asm user
in the kernel source is `_e4m3x8_to_bf16x4_asm`. The zero-tolerance scan of the real ISA is
part of the `isa` mode (gate (c) of `summarize_w64fix2_isa_cpu.py`).

Test edits:

* **`test_decg_cpu.py`** (28 tests):
  * fp8 routes to w64 with COLS=64, and to w32 with `QSA_FP8_CVT_ASM=1`; the builder guard;
    the removed identifiers are gone.
  * New class `FP8CvtIntrinsic`, which covers:
    * the knob, including invalid values;
    * that all 4 call sites go through the dispatcher and the dispatcher alone calls asm and
      intrin;
    * the `_cvtasm` name tags;
    * that the actual asm source is bitwise exact and the actual intrin source is bitwise
      exact, with 1 pk8, 4 umin, 4 perm and no asm (2,048 sweep words + 3,000 random words);
    * that the dispatcher selects intrin by default.
* **`test_fp8_cvt_cpu.py`** (13 tests): the mocked LLVM also models the intrinsics and the
  bitcasts. The actual-source tests now run on **both** converters:
  * all 256 codes × 8 byte positions × both ambient FTZ modes;
  * all NaN payloads;
  * the full TDM consumer `_pair_frag_tdm_fast` over all rows, chunks and lanes, against the
    independent TDM model.

## GPU plan (root, a08-1, gpu-lock)

Use a fresh `FLYDSL_RUNTIME_CACHE_DIR` per tag. The default fp8 kernel **names are
unchanged** while their ISA changed, so a stale cache would silently reuse the old asm build.

1. **`w64fix2-default`** (no env): `run_gpu.sh correctness`, expect 22/22.
   * This is the first GPU run of the intrinsic converter in fp8 decg w32, draft fp8 tdm and
     r32 tdm cvtfast.
   * Then run perf for decode fp8 c32768/c65536, draft fp8 c65536 and prefill fp8 (r32
     cvtfast bucket) against the champion numbers (decg fp8 w32 26.6 / 15.1 µs; draft fp8
     19.4 µs).
   * Watch fp8 decg w32: VGPR 97→115, +44 instr.
2. **`w64fix2-cols64`** (`QSA_DECODE_GROUP_COLS=64`): correctness 22/22. The kernel is the
   same as the GPU-verified `intrin` build, renamed (no `_dbgintrin`).
   * Perf on decode fp8 and bf16 c32768/c65536.
   * Make w64 the default only if it wins on both dtypes.
3. Optional A/B: `QSA_FP8_CVT_ASM=1`, correctness plus the same perf. It is identical to the
   champion's instruction stream.

## Reproduce

Run `host_w64fix2_cpu.sh STAGE {cpu-default|cpu-legacy|isa}` with
`STAGE=/data/blyu/qsa/work/w64fix2-cpu-trace-20261103`. It needs `final/source` (this
commit) and `base/source` (`a78c285c`).
