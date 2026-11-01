# ROUND-FP8CVT — exact opt-in native eight-code BF16 conversion

Base: `1ed0581a523b615d2a6cd82e4daafe2488f087fb`. Branch: `qsa-index-scorer/opt-fp8cvt`.
No GPU execution, package modifications, push, or frozen-directory writes.

## Change

Set `QSA_FP8_CVT=fast` **before importing the kernel module**. Unset/default or
`QSA_FP8_CVT=current` retains champion code. Only full single-request fp8 r32
padded-TDM B fragments use the fast helper. Scalar, r16/decode, raw, copy-atom,
mixed/tail fallback, bf16, and persistent geometry are unchanged.

The source now converts two little-endian dwords (eight codes) directly to four
BF16 pairs with stock gfx1250 `v_cvt_scale_pk8_bf16_fp8`, E8M0 scale127 = 1.
This bypasses the legacy half-bitfield / half multiply256 / scalar f32 widen /
BF16 packing sequence. Original per-token fp32 scales are **not** folded into
this instruction; scale, ReLU and left-to-right head-sum arithmetic are unchanged.

Native E4M3FN decodes `0x7f/0xff` as NaN, but champion deliberately decodes them
as `+/-480`. Exact correction:

1. Clear input signs with `0x7f7f7f7f` before conversion.
2. Packed **unsigned** min with BF16(480), `0x43f043f0`.
3. Reattach original signs with byte-permutation and `v_and_or_b32`, including -0.

Every finite native magnitude is <=448 (BF16 `0x43e0`), while **all254 BF16 NaN
encodings, either sign**, are unsigned16 greater than BF16(480), `0x43f0`.
Therefore min preserves every finite magnitude and maps either endpoint to480,
independently of canonical NaN payload or sign.

`v_cvt_pk_f16_fp8` also assembles on gfx1250, but direct native BF16 is cheaper
because WMMA consumes BF16 and needs no subsequent half-to-f32 conversion or
packing. Stock dev886 also exposes `rocdl.cvt.scale.pk8.bf16.fp8`; inline assembly
is used to explicitly enforce the **mandatory early-clobber** destination.
All32 production pk8 destinations are disjoint from their two source registers.
The original B fragment reuse across the two M tiles was already present in
champion, so option(3) offered no further hoisting gain.

## Real ISA evidence — no-device stock container

Image: `rocm/fw-bringup-mlperf:gfx1250-atom-20261015`, image ID
`sha256:d1434db7030a8355a332b2a9237598fadb07a8bf7bb6dbbd081567a13d54686a`.
FlyDSL `0.3.3.dev886`; gfx1250; `COMPILE_ONLY=1`; container asserts neither
`/dev/kfd` nor `/dev/dri` exists, uses no device flags and network none.

Production scope = one full64-column/128-K chunk **per wave**, four N16 tiles
x four K32 steps, 8192 lane-code conversions. One B fragment is shared by both M tiles.

| Quantity | Champion/current | Fast |
|---|---:|---:|
| Packed half multiply256 instructions | 128 | 0 |
| Scalar half-to-f32 widen instructions | 256 | 0 |
| Native8-code FP8-to-BF16 instructions | 0 | 32 |
| Complete conversion scalar VALU operations, including packing (helper-derived) | 1216 | 480 |
| WMMA instructions | 32 | 32 |
| LDS b128 loads | 32 | 32 |
| VGPRs (`next_free_vgpr`) | 193 | 168 |
| SGPRs (`next_free_sgpr`) | 83 | 83 |
| LDS bytes | 9216 | 9216 |
| Scratch bytes | 0 | 0 |
| FP16/64 denormal mode | 3 (preserve) | 3 (preserve) |
| Waterfall/readfirstlane markers | 0 | 0 |

Per N16/K32 B fragment: 8 packed half multiplies +16 widens become2 native8
instructions. Per K32 step across64 columns: 32 packed half multiplies +64
widens become8 native8 instructions. Full-kernel static WMMA/LDS counts remain
160/32, including the unchanged raw mixed/tail fallback.

An independent no-device micro trace **executes the actual converter helpers**
and keeps all eight outputs live. Excluding memory/address/scalar setup/delays,
complete conversion+packing costs32 issued VALU instructions/8 codes before
(38 scalar operations, with VOPD counted as2), versus15 instructions/15
operations after. Fast decomposition: 2 input masks +1 native8 +4 packed min
+4 sign permutations +4 sign-and-or. This is53.125% fewer issued conversion
instructions and60.526% fewer conversion scalar operations — **not a measured
kernel speedup**. The native instruction itself has a4-cycle blocking class.

Dispatch names:

```text
qsa_paged_mqa_logits_page4_gfx1250_fp8_wmma_D128_H4_P4_r32_c64_flydsl_tdm
qsa_paged_mqa_logits_page4_gfx1250_fp8_wmma_D128_H4_P4_r32_c64_flydsl_tdm_cvtfast
```

ISA SHA256:

```text
current/champion: f68c8c7719649287116e8a15cb9146a2deb92e528a16a5648caaf536f182c667
fast:            7d54e9e766bb6dfe6a7450b57ac80a2cfb1b4f93d1f431994b0a13e5e93d01e6
```

Machine-readable counts are in [fp8cvt_isa_evidence.json](<aiter/ops/flydsl/kernels/mqa_logits/_g2/fp8cvt_isa_evidence.json>).
Compiler corroboration: [LLVM VOP3 TableGen](https://raw.githubusercontent.com/llvm/llvm-project/main/llvm/lib/Target/AMDGPU/VOP3Instructions.td)
places native pk8 under `ReadsModeReg=0`, `@earlyclobber $vdst`, wave32 and
`GFX1250BlockingCycles=4`. This is compiler/source-reported evidence, not GPU
execution evidence; stock assembler and final ISA establish availability here.

## CPU correctness and preservation

[Exhaustive tests](<aiter/ops/flydsl/kernels/mqa_logits/_g2/test_fp8_cvt_cpu.py>) cover:

- all256 codes in both half positions and all8 byte positions;
- preserve, input-FTZ and output-FTZ domains, kept explicitly distinct;
- all254 NaN encodings, both signs, with actual new source helper execution;
- 65,536 adjacent-code pairs and10,000 randomized full dword pairs;
- actual legacy helper execution, independently checked against integer and
  literal scalar decoders;
-512 actual padded-TDM consumer fragments (4 N tiles x4 K chunks x32 lanes),
  including invalid pages, exactly matching actual champion fragments and the
  independent global reference, with identical2x16-byte load addresses;
- required early-clobber constraint in the actual helper source.

**Denormal caveat:** production emits preserve mode3, where all256 old/new codes
are bit-identical. Deliberately forcing legacy FP16 input FTZ loses14 codes
(`01..07,81..87`) into signed zero; the direct BF16 path has no half intermediate
and does not reproduce that broken, non-production mode. Tests explicitly prove
this difference rather than falsely claim mode-independent legacy identity.
Native outputs are BF16 normals (minimum2^-9), so output BF16 FTZ loses no codes.
A hypothetical hardware **FP8 source FTZ** would lose the same14 codes and must
be excluded by the device code-sweep before any promotion.

All15 existing CPU/strict-ISA gates pass: contract, address, Q-LDS, trace16/32
(16 cases each), K-TDM (37 tests), routing (12), fp8 K-TDM (18), BM64 (12),
persist (18), exhaustive fp8 conversion (11), strict-TDM, routing-ISA, fp8-ISA
and persist-ISA. The new strict conversion-ISA gate also passes. Kernel source
SHA256 is `ab5fd6d273637d73c95109396507c2a2db6fe2a95b164150656b1fb10dbf1ac5`;
no-device compile and CPU-test source hashes match exactly. See
[fp8cvt_cpu_evidence.json](<aiter/ops/flydsl/kernels/mqa_logits/_g2/fp8cvt_cpu_evidence.json>).
46 switch-off ISA files across auto/off/raw/tdm/p2/p4/p8 are byte-identical to
fresh champion rebuilds. With fast enabled,12 scalar/bf16/r16/raw ISA files in
auto/raw configurations are also byte-identical. All masks, descriptor extents,
Q/K addresses, guards, wrapper API and score/store helpers are unchanged.

## GPU plan and risks

No speedup is claimed. For the903.8us fp8 bmge64 prefill case, a20% reduction is
only an unmeasured hypothesis (plausible range0..35%; regression remains possible).
Unchanged paths have expected0% effect because their actual ISA is identical.

1. Under the shared-host GPU lock, run the actual converter micro code-sweep:
   all256 codes, all byte/half positions, signed zeros and endpoints; compare
   raw BF16 words against the preserve-mode champion and the CPU proof.
2. Run full32-row single-request padded-TDM logits current/fast, randomized
   bf16 Q and fp32 token scales; require bitwise equality (finite values and
   infinities), with endpoint/subnormal-heavy K, invalid page entries, column
   tails and visibility tails. Run mixed/partial requests to exercise fallback.
3. Confirm the emitted dispatch name, preserve metadata, zero scratch and no
   waterfall; smoke-test scalar, r16, raw and bf16 regressions.
4. Locked unprofiled A/A then alternating A/B with warmups and held-out rerun
   of bmge64 fp8 prefill, plus representative smaller fp8 prefill cases.
   Report medians/dispersion and use903.8us only as the provided prior baseline.
5. If timing improves, collect ATT/VALU evidence separately; counts and VGPR
   reduction alone do not establish the bottleneck or a kernel speedup.

Risks: native hardware FP8 subnormal policy and instruction scheduling remain
unexecuted; opaque inline assembly may constrain scheduling; lower VGPR counts
need not change occupancy; LDS/memory/store costs remain unchanged. Remain opt-in
until all device bit-identity gates and timing gates pass.
