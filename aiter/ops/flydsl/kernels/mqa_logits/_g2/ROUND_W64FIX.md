# ROUND-W64FIX — fp8 w64 decode-group bug: root cause + GPU bisect builds

Base: champion `a78c285c` (kernel sha `30c6213d…`). Branch `qsa-index-scorer/opt-w64fix`.
Kernel sha `47366a19…`. CPU/no-device evidence only — **no GPU run in this round**.
Default behaviour is unchanged: fp8 w64 stays disabled, and the default-env ISA of all
18 specs × {i32, qp64} is byte-identical to `a78c285c` (36/36).

## Root cause (ranked hypotheses)

1. **WMMA data hazards that inline asm hides from LLVM.** This is the main hypothesis, with
   strong static evidence.
   * Every instruction of the exact pk8 converter `_e4m3x8_to_bf16x4_fast` is an inline asm
     statement: `v_cvt_scale_pk8_bf16_fp8`, `v_pk_min_u16`, `v_perm_b32` and `v_and_or_b32`.
     LLVM's hazard recognizer only pads hazards for instructions it knows. An `INLINEASM`
     is not a VALU to it, so it never pads them.
   * gfx1250 dense bf16 WMMA has software-enforced hazards, listed in kernel-wiki
     [`mi400-wmma-hazards`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-wmma-hazards.md)
     §1.2 row B1 (SPG §4.6.12.1, image-verified, "correctness failure, not merely a
     slowdown"). A VALU must be 4 V_NOPs or independent VALUs after the WMMA when it:
     * has a vdst that overlaps the WMMA's Matrix A/B (WAR), or
     * has a vdst that overlaps Matrix D (WAW), or
     * reads Matrix D (RAW).
   * `_g2/wmma_hazard_scan_cpu.py` scans the real no-device ISA for these cases:

     | kernel | WMMA | violations | of which compiler-generated | GPU status |
     |---|---|---|---|---|
     | bf16 w32 / w64 decg | 32 / 64 | 0 / 0 | 0 | correct |
     | fp8 r32 tdm cvtfast | 160 | 0 | 0 | correct |
     | fp8 w32 decg | 32 | 28 | 0 (all asm) | correct |
     | fp8 draft tdm w32 / raw w64 | 8 / 16 | 15 / 20 | 0 | correct |
     | **fp8 w64 decg** | 64 | **31** | 0 (all asm) | **wrong** |

   * Every violation is an asm instruction, and every kernel without the asm converter is
     clean. That agrees with the claim that LLVM pads everything it can see.
   * **What separates w64 from the correct kernels.** Look at the decode pass 0 (ISA lines
     1371–2393; the failing GPU cases are all single-pass decode groups). At 192 VGPRs, the
     register allocator reuses Q-fragment registers (Matrix **A**, dead after the last tile)
     and middle B registers at distance 0:
     * `v_perm_b32 v14` overwrites **A[4]** at +0 VALU (WMMA L2344);
     * `v_pk_min_u16 v26` overwrites A[0] at +0, and `v_perm v27` overwrites A[1] at +2
       (L2326);
     * `v_and_or_b32 v64` overwrites **B[2]** at +0 (L1471);
     * `v_cvt_scale_pk8 v[50:53]` overwrites B[0:3] at +2 (L1583).

     The fp8 w32 kernel, which is GPU-correct, never overwrites A, and its only non-pk8
     write at +0 is to B[0]. A plausible reading: operand VGPRs are read progressively, so
     B[0] at +0 survives but A[4] and B[2] do not. This is an inference, not a measurement.
     In the mixed passes (p ≥ 1) the w64 ISA also has asm WAW on **D**: the pk8 reuses dead
     accumulator halves at +0/+2 (L3945, L4302…). That affects mixed groups.
   * Conclusion: the w32 and draft kernels pass by register-allocation luck. Their latent
     violations are the same class of bug.
2. **TDM gather count / TENSORcnt / descriptor fields** (8 gathers per wave, 144 B pitch,
   9,216 B per-wave tile). Low likelihood:
   * bf16 w64 issues the same 8 gathers per wave with larger LDS offsets (up to 67,456 B, vs
     35,712 B for fp8) and is correct;
   * all fp8 offsets are multiples of 128 B (9,216 = 72·128, 1,152 = 9·128);
   * the ISA keeps all 8 `tensor_load_to_lds` and the `s_wait_tensorcnt 0` ahead of the first
     `ds_load`. Per kernel-wiki
     [`mi400-tdm-sync`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-tdm-sync.md),
     TENSORcnt is per wave, so that wait is sufficient.
   * The `s_wait_tensorcnt 0xa` between issues is a compiler-inserted no-op cap.
   * Variant `tdmwait` tests this hypothesis.
3. **ds_load offsets, DScnt cascade, scale loads, store offsets.** Low likelihood:
   * the largest `ds_load` immediate is 9,200 B, far below the 16-bit field limit;
   * the `s_wait_dscnt` cascade 0x11…0x0 matches the 18 outstanding loads (kernel-wiki
     [`mi400-dependency-counters`](C:/Users/blyu/Documents/Workspace/MI_Kernel_Agent/skills/kernel-wiki/wiki/mi400/mi400-dependency-counters.md));
   * the scale `buffer_load` is followed by `s_wait_loadcnt` before use;
   * the store offsets come from the same helper as bf16 w64, which is correct;
   * the CPU lane emulation of fp8 w64 is exact for decode, mixed and partial groups.

## Variants (`QSA_DECG_FP8_W64_DEBUG`, read at import; needs `QSA_DECODE_GROUP_COLS=64`)

| value | change | kernel suffix | ISA (i32): instr / VGPR / hazards |
|---|---|---|---|
| `0` (default) | fp8 w64 disabled: route → w32, builder raises | — | ISA identical to base |
| `asis` | original bbe25c33 w64 body (control: expected to be GPU-wrong) | `_dbgasis` | 5720 / 192 / **31** |
| `nop4` | each converter asm statement is preceded by 4 `v_nop` | `_dbgnop4` | 12376 / 192 / **0** |
| `intrin` | **best fix**: the converter is built from compiler-visible intrinsics (`llvm.amdgcn.cvt.scale.pk8.bf16.fp8` with scale 0x7f and sel 0, `llvm.amdgcn.perm`, `llvm.umin.v2i16`, and/or), so LLVM pads the hazards itself (31 compiler `v_nop`) | `_dbgintrin` | 5774 / 229 / **0** |
| `tdmwait` | `asis` plus `tensor_wait(0)` after each gather | `_dbgtdmwait` | 5679 / 193 / 33 |

* No variant uses scratch.
* The LDS footprint is 36,864 B in every variant.
* Every variant has 64 WMMA and 32 `tensor_load_to_lds`.

## GPU plan (root, a08-1, gpu-lock; one session; correctness only first)

Run `run_gpu.sh correctness <tag>` once per environment. Each run executes all 22 cases;
the cases that matter are the 5 fp8 decode-group cases (I5_min_decode_fp8, I6_2req_fp8,
tail_columns_1000_fp8, edge_bad_page_fp8, I2_e2e_decode_fp8). Use a fresh
`FLYDSL_RUNTIME_CACHE_DIR` per tag. The kernel names differ per variant, but a fresh
directory avoids stale-cache doubt.

| tag | env |
|---|---|
| w64fix-default | (none) — expect 22/22, identical to the champion |
| w64fix-asis | `QSA_DECODE_GROUP_COLS=64 QSA_DECG_FP8_W64_DEBUG=asis` — expect the same 5 fp8 failures (reproduction) |
| w64fix-nop4 | `QSA_DECODE_GROUP_COLS=64 QSA_DECG_FP8_W64_DEBUG=nop4` |
| w64fix-intrin | `QSA_DECODE_GROUP_COLS=64 QSA_DECG_FP8_W64_DEBUG=intrin` |
| w64fix-tdmwait | `QSA_DECODE_GROUP_COLS=64 QSA_DECG_FP8_W64_DEBUG=tdmwait` |

How to read the results:

* nop4 and intrin pass, while asis and tdmwait fail → the cause is the WMMA hazard
  (hypothesis 1).
* tdmwait passes → TDM is implicated (hypothesis 2).
* All of them fail → the cause is something else (hypothesis 3). Next bisect: a separate
  scale path, or n_tiles=2.

If intrin is 22/22, measure it on decode fp8 c32768 and c65536 against the default w32. Then
consider porting the intrinsic converter to the w32, draft and r32 fp8 paths. They have the
same latent asm hazards, and fp8 w32 already has 28.

## Evidence (a07-3 container `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, `--network none`, no devices)

* Staging: `/data/blyu/qsa/work/w64fix-cpu-trace-20261103`, run with
  `host_w64fix_cpu.sh STAGE {cpu-default,isa}`.
* **CPU** (`cpu-default`): all 18 gates exit 0.
* **ISA** (`isa`, 84 real-wrapper compiles, `W64FIX_ISA_OK`, `w64fix_isa_evidence.json`):
  * Default env matches base `a78c285c` for 36/36: raw sha, normalized sha, name and
    resources.
  * `QSA_DECODE_GROUP_COLS=64` without the knob still builds the fp8 **w32** kernel, with
    the same sha as the default.
  * Variant table as above; nop4 and intrin have 0 hazard violations.
* **`test_decg_cpu.py`** (27 tests, previously 21):
  * Knob tests: default/`0` still route fp8 to w32; each variant routes fp8 to 64 only when
    COLS=64; an invalid value raises.
  * Source hooks.
  * New lane emulation: fp8 w64 mixed and partial groups.
  * **Executed actual source** of both new converters with mocked LLVM: bitwise equal to
    `reference_pk8` over all 256 codes × 8 byte positions plus 3,000 random words. nop4 is
    checked to issue 13 asm statements, each with 4 `v_nop`. intrin is checked to use no
    inline asm and exactly 1 pk8, 4 umin and 4 perm.
* **Gate edit:** `test_champion_identical_helpers` normalizes the builder-local `cvt8` name.
  It is `_e4m3x8_to_bf16x4_fast` unless a debug variant is built.
