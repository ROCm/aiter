# Native MXMoE GEMM2 BK128 Support

## Status

Approved design for adding explicit `BK=128` support to the native FlyDSL
MXMoE GEMM2 path.

## Context

The native MXMoE GEMM2 currently accepts `BN` and `BK` in its Python wrapper,
but the production path still behaves as a fixed `BN=256, BK=256` kernel:

- the kernel name tokenizer reads `<BM>x<BN>x<BK>`, but the native GEMM2 parser
  drops `BN` and `BK`;
- `fused_moe` therefore reaches the wrapper through its default `BN=BK=256`;
- native AOT jobs and generated GPU symbols do not distinguish `BK`;
- the tuner emits only `x256x256` names;
- the device kernel assumes two K128 MFMA halves, a 128-byte FP4 LDS row, a
  2048-byte B-weight tile step, and one K256 scale word per GEMM K tile.

Consequently, an actual `inter_dim=384` cannot use the native GEMM2. GEMM1 can
produce 384 values because its tiled output is `2 * inter_dim = 768`, but GEMM2
uses `inter_dim` directly as its contraction K.

The v2 GEMM2 already contains validated BK128 geometry and scale-word handling.
This design reuses those algorithms as evidence while keeping the native
kernel, layouts, epilogues, and tuning path independent.

## Goals

1. Support native GEMM2 with `BN=256` and `BK in {128, 256}`.
2. Run actual, unpadded K dimensions that are multiples of 128, including
   `inter_dim=384`.
3. Preserve every currently supported native BM/NT/epilogue combination.
4. Propagate the explicitly encoded BK through runtime, AOT, cache keys,
   generated symbols, and tuner candidates.
5. Preserve explicit dispatch: the configured kernel name determines BK.
6. Keep BK256 computation and performance unchanged apart from its newly
   explicit generated-symbol suffix.

## Non-goals

- Supporting `BN=128` or generic BN.
- Changing GEMM1.
- Changing the tuned CSV schema or shape key.
- Automatically changing BK at runtime.
- Adding BK128 padding or `inter_real` behavior to the acceptance contract.
- Refactoring native and v2 GEMM2 into a shared implementation.
- Retuning or adding model config rows as part of the kernel capability change.

## Public and Internal Contracts

### Kernel names

Native names remain:

```text
flydsl_mxmoe_g2_a4w4_<BM>x256x<BK>[_atomic][_nt][_f4out][_cshuffle]
```

`BK` is exact:

- `...x256x256` always selects BK256;
- `...x256x128` always selects BK128;
- no fallback or heuristic may reinterpret the name.

The legal native tile set is `BN=256, BK in {128, 256}`.

### Shape lookup

`inter_dim` continues to come from the actual `w1` and `w2` tensor shapes.
Configuration lookup and tuner shapes do not inspect `w2.inter_real`.

Examples:

- actual K384 uses an `inter_dim=384` row and a BK128 name;
- actual K512 may be tuned with both BK128 and BK256 candidates.

### Generated GPU symbols

Both BK variants receive an explicit suffix:

```text
..._bk128
..._bk256
```

This intentionally invalidates old BK256 binary-cache entries once. It avoids
symbol collisions when one process or AOT module compiles the same shape with
both BK values. CSV kernel names are unaffected.

## Architecture and Parameter Flow

The complete native path becomes:

```text
CSV kernelName2
  -> native kernel-name parser (BM, BN, BK, flags)
  -> fused_moe stage2 metadata and wrapper
  -> flydsl_mxfp4_gemm2 runtime wrapper and launcher cache
  -> compile_gemm2_a4w4_port specialization
```

The native parser returns `BM`, `BN`, and `BK`. The unified stage2 parser
returns the same tile fields for native and v2 names. `fused_moe` parses once,
then forwards `BN` and `BK` through both native GEMM2 call sites: normal output
and MXFP4 output.

The runtime wrapper validates the exact native contract before compiling:

- `BN == 256`;
- `BK in {128, 256}`;
- `D_INTER % BK == 0`;
- `D_HIDDEN % BN == 0`;
- `(BM, use_nt, epilogue)` belongs to the existing supported set.

The launcher cache already includes `BN` and `BK`; that structure remains.

### Affected components

- `aiter/ops/flydsl/mxfp4_kname.py`
  - retain native GEMM2 BN/BK in parsed metadata.
- `aiter/fused_moe.py`
  - propagate parsed BN/BK through the native stage2 wrappers and both output
    paths.
- `aiter/ops/flydsl/mxfp4_gemm2_kernels.py`
  - enforce the public tile contract and preserve BN/BK in the compile cache.
- `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py`
  - provide the correct ceil-divided K256 B-scale chunk stride.
- `aiter/ops/flydsl/kernels/mxfp4_gemm2.py`
  - implement BK-derived A/B geometry, scale selection, MFMA halves, pipeline
    indexing, and generated-symbol suffixes.
- `aiter/aot/flydsl/mxfp4_moe.py`
  - carry BN/BK in native jobs, keys, and compile calls.
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`
  - generate and benchmark legal native BK128/BK256 candidates.
- `op_tests/flydsl_tests/`
  - add parser, propagation, AOT, tuner, cache, and GPU correctness coverage.

## Device-Kernel Design

### Compile-time geometry

The device kernel derives:

```text
kHalves            = BK // 128
KH_TILE            = BK // 2              # packed FP4 bytes per A row
kTiles             = D_INTER // BK
tilesPerScaleChunk = 256 // BK
kScaleSubBlocks    = max(1, BM // 32)
```

For BK128, `kHalves=1` and `KH_TILE=64`. For BK256, `kHalves=2` and
`KH_TILE=128`. The scaled MFMA remains `16x16x128`; only the number of MFMA
halves per GEMM K tile changes.

### A global-to-LDS copy

Replace the fixed 128-byte-row geometry with compile-time-derived geometry:

```text
lanesPerRow = KH_TILE // 16
rowsPerCall = 64 // lanesPerRow
nLoadWaves = min(4, BM // rowsPerCall)
rowsPerWave = BM // nLoadWaves
loadGroups = rowsPerWave // rowsPerCall
```

This gives:

- BK128: four lanes per 64-byte row and 16 rows per wave call;
- BK256: the current eight lanes per 128-byte row and eight rows per wave call.

The LDS swizzle receives `KH_TILE` explicitly. This selects the existing
64-byte swizzle for BK128 and the current 128-byte swizzle for BK256. A-copy
load groups are independent of the 32-row scale/MFMA groups.

### A LDS-to-register and B weight loads

A and B fragments use `kHalves` instead of a fixed length of two.

- Each A half still reads 64 bytes per row.
- Each B half still represents a 16-column by K128 packed-FP4 fragment and
  consumes 1024 bytes across the wave.
- The B tile step becomes `kHalves * 1024` bytes instead of a fixed 2048 bytes.
- BK128 issues one K128 MFMA per tile; BK256 issues two.

For a fixed total K, the total number of MFMAs is unchanged.

### E8M0 scale words

Scale storage remains K256-granular. It must not be mechanically retiled to
K128. BK128 tile pairs share one scale word:

```text
scaleChunk = kt // tilesPerScaleChunk
scaleShift = (kt % tilesPerScaleChunk) * 16
```

For BK128, even tiles use the low 16 bits and odd tiles shift the high 16 bits
down before using the existing low-half opsel values. For BK256, no shift is
performed and the two MFMA halves retain the existing low/high opsel mapping.

The B-scale K-chunk count changes from floor division to `ceil(K / 256)`.
Existing K values divisible by 256 are unaffected; K384 correctly receives two
scale chunks per N-pack and per expert.

### K pipeline

Keep the current two-stage prefetch and optional third rotating A slot.
Increasing the number of K tiles for BK128 does not change the pipeline model.

Every MFMA-cluster call receives the real `kt`. This is required for BK128
scale-word low/high selection and also removes the current long-loop dependence
on the helper's default `kt=0`.

Half validity is expressed as `kt * kHalves + half`. Existing BK256
`D_INTER_REAL` behavior remains intact, but BK128 padding is not an acceptance
requirement.

### Epilogues

The accumulator shape and the atomic, nonatomic, cshuffle, and MXFP4-output
epilogues do not depend on BK. Their algorithms and output layouts do not
change.

### BK256 stability

All generalized quantities are compile-time constants. BK256 should fold back
to the existing loads, addresses, MFMA sequence, and pipeline. Normalized
IR/ISA comparison ignores the intentional symbol-name suffix. If generalized
source changes the substantive BK256 code, sensitive load or MFMA blocks retain
a compile-time BK256 branch.

## AOT Design

Native stage2 AOT jobs store `BN` and `BK`; their job keys include both fields.
The compile call forwards both values.

AOT derives the stage1/stage2 intermediate compile shape from the BK encoded in
`kernelName2`:

- K384/BK128 compiles K384;
- K512/BK128 compiles K512;
- K512/BK256 compiles K512.

Stage1 and stage2 jobs use the same intermediate K. BK128 and BK256 jobs for
the same K512 shape remain distinct and generate the corresponding explicit
symbol suffixes.

## Tuner Design

The native GEMM2 kernel-name builder receives `BN` and `BK`; it no longer
hard-codes `x256x256`.

Candidate generation follows:

- if `K % 256 == 0`, benchmark BK128 and BK256 for every otherwise legal native
  candidate;
- if `K % 128 == 0` but `K % 256 != 0`, benchmark only BK128;
- otherwise, generate no native MXMoE GEMM2 candidate.

BK is encoded in `kernelName2`, not added as a CSV column. The tuner continues
to write only the lowest-`us` winner for each existing shape key.

## Error Handling

Invalid tile parameters fail before compilation. Diagnostic messages include
the kernel name, `D_INTER`, `BN`, and `BK` where available.

In particular:

- `x256x256` with K384 fails; it never silently selects BK128;
- BN values other than 256 fail;
- BK values other than 128 or 256 fail;
- unsupported BM/NT/epilogue combinations retain their current rejection.

## Verification

### Host-side tests

1. Parser:
   - native BK128 and BK256 names round-trip;
   - atomic, nonatomic, NT, f4out, and cshuffle flags remain intact;
   - old `x256x256` names keep their meaning.
2. Runtime propagation:
   - both native stage2 call sites forward parsed BN/BK;
   - no fallback rewrites an explicit BK.
3. AOT:
   - jobs, keys, compile arguments, and symbols distinguish BK128/BK256;
   - K384/BK128 compiles stage1 and stage2 at K384;
   - one K512 input can produce distinct BK128 and BK256 jobs.
4. Tuner:
   - K384 produces only BK128 candidates;
   - K512 produces BK128 and BK256 candidates;
   - K320 produces no native candidates.

### GPU correctness matrix

Cover all eleven existing native variants:

```text
atomic:    BM 16/32/64 x NT off/on
nonatomic: BM 128
f4out:     BM 128
cshuffle:  BM 32/64/128
```

Each variant runs:

```text
K384 / BK128
K512 / BK128
K512 / BK256
```

The 33 core combinations use a small `D_HIDDEN=256` shape with at least two
experts and `topk > 1`, plus token counts that exercise partial and multiple
BM blocks.

- Atomic output is compared with a Torch stage2 reference built from the exact
  quantized operands.
- Nonatomic and cshuffle outputs are reduced through their intended path before
  comparison.
- MXFP4 output validates values, scales, and dequantized output.
- Every output must be finite.
- Repository-standard tolerances are used: `atol=1.0`, `rtol=0.05`, and at
  most 5% mismatched elements.

Add one high-level FMoE smoke case with actual `inter_dim=384` and an explicit
native `x256x128` kernel name to verify the complete dispatch chain.

### Cache, AOT, and performance

- Compile K512/BK128 and K512/BK256 in one process and verify separate launcher
  keys and GPU symbols.
- Run representative correctness cases with cold and warm caches.
- Run representative cases from AOT artifacts.
- Compare old and new BK256 normalized IR/ISA while ignoring only the symbol
  suffix.
- Investigate any BK256 median benchmark regression above 3%; a single noisy
  measurement is not a hard CI failure.
- Record BK128 K384 and K512 performance without an absolute threshold. The
  tuner decides adoption by measured `us`.

## Risks and Mitigations

1. **64-byte LDS row overrun**
   - Mitigation: derive lanes per row and pass `KH_TILE` to the swizzle; cover
     every supported BM.
2. **Wrong BK128 scale half**
   - Mitigation: share K256 words by tile pair and shift odd tiles; test K384,
     which exercises low/high/next-low ordering.
3. **Wrong per-expert B-scale stride**
   - Mitigation: use `ceil(K/256)` and require at least two experts in GPU tests.
4. **Long K-loop uses the wrong tile index**
   - Mitigation: pass `kt` explicitly at every MFMA-cluster call and test K512
     BK128, which has four tiles.
5. **BK128/BK256 cache or symbol collision**
   - Mitigation: include BK in AOT/runtime keys and suffix every generated
     symbol.
6. **BK256 performance drift**
   - Mitigation: compile-time folding, normalized IR/ISA comparison, and
     targeted benchmarking.

## Rollout

Land capability, parser/dispatch/AOT/tuner propagation, and tests together.
Do not add or retune model CSV rows in the same change. After correctness and
BK256 regression verification, tune K384/K512 shapes separately; only measured
winners should enter model configs.
