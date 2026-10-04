# GEMM1 Fused A-Preshuffle Quantization Optimization Log

## Scope

- Source worktree: `a4w4-refactor`
- Source branch: `hyg/moe_a4w4_pr_refactor`
- Starting commit: `c0d98474c1a2bc937b07930d35221d7bfc8dd713`
- Target shape: `E64/T1536/topk8/M7168/I2048`, A4W4, SiLU, no bias
- Goal: fuse the standalone GEMM1-to-GEMM2 quantization into GEMM1 while
  preserving the A-preshuffle payload and ScaleA layouts consumed by the
  baseline optimized GEMM2.

## Current frozen implementation

The fused implementation is frozen at the validated v21 code. Its production
kernel suffix no longer carries an experimental version number:

```text
apreqb16batchs16direct_scaletdm
```

`AITER_FLYDSL_GEMM1_FUSED_QUANT=1` selects this implementation and remains the
default. `AITER_FLYDSL_GEMM1_FUSED_QUANT=0` retains the original two-kernel
GEMM1 plus standalone quant pipeline. There is no runtime selector for any
other fused optimization version.

After the rename, random seed 0 and const0 again produced bitwise-identical
payload, ScaleA, GEMM2 valid rows, and MoE output. A three-round const0
performance regression run produced:

```text
fused GEMM1 samples: 94.570, 94.728, 92.666 us
fused GEMM1 median:  94.570 us
MoE samples:         228.04, 227.19, 221.71 us
MoE median:          227.19 us
```

The final source SHA256 is:

```text
964651c9d75565f8ea416f252714d0dacdfd2bdd06fab1962735bbc2dddbbf96
```

## New 20% optimization campaign baseline

The optimization campaign uses the frozen v21 implementation with production
kernel suffix `apreqb16batchs16direct_scaletdm`. The target remains exactly
`E64/T1536/topk8/M7168/I2048`, A4W4, SiLU, no bias. Kernel eligibility,
quantized output layout, arithmetic semantics, and accuracy requirements must
remain unchanged.

The baseline was measured on a07-3 after confirming that every GPU and KFD were
idle before and after the command:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

Results:

| Metric | Samples (us) | Median (us) |
|---|---|---:|
| fused GEMM1 | 96.858, 103.192, 93.843 | **96.858** |
| MoE e2e | 225.85, 238.55, 223.74 | **225.85** |

All three rounds reported `logits_diff=0`, `rel_l2=0`, matching GEMM1 and GEMM2
hashes, and matching MoE/reference output hashes. A 20% GEMM1 latency reduction
from this baseline requires a median no greater than **77.486 us**.

This document records only:

1. the first implementation that passes correctness;
2. later changes with a stable performance improvement;
3. the exact validation/performance commands and measured data.

Failed or unvalidated experiments are not promoted into the retained-version
sections. They may be summarized briefly when needed to explain a constraint.

## First Correct Version

### `apreqb32v1`

The first correctness-complete implementation uses the tuned A-preshuffle
GEMM1 schedule with `stage1_quant_out=1` and the kernel-name suffix
`_q1r6_apreqb32v1` for the target E64 shape.  It makes these changes:

- GEMM1 applies SiLU in f32 and rounds the result to BF16 at the same semantic
  boundary as the original two-kernel path.
- The BF16 values feed the native MXFP4 amax/scale and packed conversion.
- Each pair of `kgrp` lanes exchanges its four values with
  `shuffle_xor(16)`.  `kgrp0` writes one complete packed `b32` per WN.
- Four WNs fill the four contiguous dwords in a 16-byte MX32 payload segment.
- A three-dimensional TDM store writes
  `[M16 tile, local MX block, 256 bytes]` directly into the
  `shuffle_weight_f4` layout.
- Scale bytes are written directly into the matching
  `shuffle_scale_f4(..., wmma_rep=6)` layout.
- Fusion is enabled only when GEMM1's quant producer layout matches GEMM2's
  selected A-input layout.  Mixed row-major/A-preshuffle schedules retain the
  standalone conversion kernel.

The random diagnostic compares the two launches in canonical route order.
This is required because atomic route scatter may assign different grouped-row
numbers to the same route on consecutive launches.

Random seed 0 results:

```text
ScaleA byte mismatch: 0
FP4 semantic mismatch after canonicalizing +0/-0: 0
GEMM2 valid-route output: bitwise identical
MoE output: bitwise identical
logits_diff=3.3849e-06
rel_l2=2.6019e-03
```

The packed payload had 641,363 raw byte differences over 12,582,912 bytes.
Every difference was only the FP4 zero sign (`0x0` versus `0x8`); all
25,165,824 FP4 values matched after canonicalizing signed zero.

Const0 results:

```text
Payload byte mismatch: 0
ScaleA byte mismatch: 0
GEMM2 valid-route output: bitwise identical
MoE output: bitwise identical
logits_diff=0
rel_l2=0
```

The standalone
`moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2` kernel is absent
from the fused MoE profiler sequence.

Initial const0 performance:

| Round | Fused GEMM1 (us) | GEMM2 (us) | MoE e2e (us) | Correctness |
|---:|---:|---:|---:|---|
| 1 | 116.174 | 67.248 | 244.824 | bitwise equal |
| 2 | 115.462 | 69.772 | 244.878 | bitwise equal |
| 3 | 104.749 | 67.834 | 236.211 | bitwise equal |
| **Median** | **115.462** | **67.834** | **244.824** | **bitwise equal** |

The first correct version is slower than the reboot baseline.  Its median
GEMM1 time is 25.2% above the baseline GEMM1 + standalone quant median
(`115.462 us` versus `92.232 us`), and its MoE median is 14.3% higher
(`244.824 us` versus `214.226 us`).  The concurrent control kernel also moved:
GEMM2 rose from `59.925 us` to `67.834 us`, so an interleaved baseline rerun is
required before attributing the entire difference to the fused epilogue.

## Retained Performance Improvements

### `apreqb16v2`: local-kgrp packing with split `b16` LDS stores

The first correct version exchanged four values across `kgrp` and executed four
pk8 conversions per row/MX block.  `apreqb16v2` keeps each kgrp's 16 activated
values local, packs two WNs with one pk8 conversion, and splits the result into
two `b16` LDS stores.  This removes the peer shuffles and halves the number of
native FP4 conversion instructions.

Random seed 0 and const0 both produced exact payload bytes, exact ScaleA bytes,
bitwise-identical GEMM2 valid-route output, and bitwise-identical final MoE
output compared with the original standalone quant path.

Const0 performance:

| Round | Fused GEMM1 (us) | GEMM2 (us) | MoE e2e (us) | Correctness |
|---:|---:|---:|---:|---|
| 1 | 93.699 | 69.872 | 224.946 | bitwise equal |
| 2 | 90.672 | 67.339 | 217.632 | bitwise equal |
| 3 | 88.796 | 63.780 | 214.842 | bitwise equal |
| **Median** | **90.672** | **67.339** | **217.632** | **bitwise equal** |

Relative to `apreqb32v1`, the median fused GEMM1 time improved by 21.47% and
the MoE median improved by 11.11%.  Relative to the reboot baseline's combined
GEMM1 + standalone quant time, fused GEMM1 improved by 1.69% (`90.672 us`
versus `92.232 us`).  The absolute MoE comparison remained affected by clock
drift: the concurrent GEMM2 control was 12.37% slower than in the initial
baseline measurement.

### `apreqb16batchs16direct_htv10`: retained final candidate

The retained candidate builds on the exact `b16` payload path:

- batches all eight WNs for one output row before draining the SiLU chain;
- reuses the already rounded BF16 vector directly for native FP4 conversion;
- packs each wave's two adjacent ScaleA bytes into one aligned `b16` store;
- marks the payload TDM store `HT` (`cache_modifier=2`) so GEMM2's immediate
  read has higher cache-retention priority.

The E64 random diagnostic passed for seeds 0, 1, and 2.  E96/T16384/topk6/I3072
and E256/T16384/topk8/I2048 random diagnostics also passed.  In every retained
test:

```text
payload byte mismatch=0
ScaleA byte mismatch=0
GEMM2 valid-route output bitwise mismatch=0
MoE output bitwise mismatch=0
```

Representative final hashes:

| Shape/data | payload hash128 | ScaleA hash128 | GEMM2 valid rows hash128 | MoE output hash128 |
|---|---|---|---|---|
| E64/T1536, random seed 0 | `ffcd62935984e66214295ed60a21aa2c` | `382b8c35932655b4827edfe140d84051` | `9b6727d9cab4e669cc6bd80c1df18889` | `5069dae4a9dc8e4eeafe5773d2694405` |
| E64/T1536, random seed 1 | `d7ef7849376530fe6e95b4382d0e4699` | `382b8c35932655b4827edfe140d84051` | `67637e2c5f9eea9edd6a9363fd5c880c` | `179fdaa09570bbee9d7796ccf7332193` |
| E64/T1536, random seed 2 | `e05f7114e161c746f61d29621639b322` | `382b8c35932655b4827edfe140d84051` | `45c3f653aae4d122ccae203acceae1e4` | `593be7cc209c18b6619ce9b8929efc6b` |
| E64/T1536, const0 | `8261297ddb9250f0ff92b95920826d2f` | `2945ce41bfcad0530732759b254d0c68` | `e783482f0c3885ba82c76d991803d4ab` | `6bebf6409ef198fe1a0255681f4f784f` |
| E96/T16384, random seed 0 | `af32712bf622d839279af352c4f3b57b` | `40c3af8d76f4672d9c11e521a67991a0` | `81cb2e7e826b9321cec8f8d3234380a6` | `1556fc617347e2dabc9cff19dbfd822b` |
| E256/T16384, random seed 0 | `94833a5502f4618b892dfc8620ca5c96` | `95b3bad9d075338c3205e0e7137ab145` | `e468f3a7964060272af7861b5c0e152f` | `a62355d90642228b640e688fedd17538` |

The compiled E64 kernel uses 243,712 bytes of LDS, 64 SGPRs, 640 VGPRs, and has
no VGPR spills.  Compared with `apreqb16v2`, the generated ISA contains 113
fewer instructions and 22 fewer `s_wait_alu` instructions.

Three-round, 100-iteration interleaved A/B results:

| Case | GEMM1 samples (us) | quant samples (us) | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) | Correctness |
|---|---|---|---:|---:|---:|---|
| baseline | 80.347, 77.046, 75.907 | 11.447, 8.465, 8.622 | 85.511 | 56.655 | 196.544 | bitwise equal |
| fused HT | 88.726, 81.978, 82.104 | removed | 82.104 | 60.077 | 199.325 | bitwise equal |

The fused GEMM1 is 3.407 us, or 3.98%, faster than baseline GEMM1 plus the
standalone quant kernel.  The full MoE remains 2.781 us, or 1.42%, slower in
this run because GEMM2 is 3.422 us slower after consuming the fused producer's
output.  The remaining optimization target is therefore the producer-to-GEMM2
cache handoff rather than GEMM1 arithmetic itself.

A second independent 3-round run confirmed the same result:

| Case | GEMM1 samples (us) | quant samples (us) | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) | Correctness |
|---|---|---|---:|---:|---:|---|
| baseline | 77.628, 74.492, 74.806 | 13.469, 8.573, 8.612 | 83.418 | 56.137 | 194.462 | bitwise equal |
| fused HT | 92.427, 83.651, 83.135 | removed | 83.651 | 61.622 | 202.333 | bitwise equal |

Across both runs (six samples per case), the medians are:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline | 85.020 | 56.538 | 196.301 |
| fused HT | 83.393 | 60.240 | 201.772 |

The fused GEMM1 stage is 1.91% faster than baseline GEMM1 plus standalone
quant across the combined sample set.  The remaining 2.79% MoE regression is
downstream: GEMM2 is 3.702 us slower at the median even though its input bytes
and ScaleA bytes are exact.

The following alternatives were rejected:

- adding `s_wait_storecnt 0` after the output TDM (`waitv8`) increased fused
  GEMM1 median from about 82.1 us to 82.8 us and did not recover GEMM2 time;
- changing the payload store from `HT` to `WB` (`wbv11`) increased fused GEMM1
  median to 86.4 us and MoE median to 206.0 us;
- changing the payload store to `NT_HT` (`nthtv12`) produced a fused GEMM1
  median of 82.911 us but a MoE median of 201.149 us versus its 194.736 us
  baseline;
- adding the standalone producer's 32-byte per-MX-block LDS padding
  (`htpadv13`) produced a fused GEMM1 median of 84.319 us and a MoE median of
  203.963 us versus its 198.750 us baseline;
- using two output-TDM waves (`otdmw2v9`) did not improve the pipeline;
- setting the fused GEMM2 A-load temporal hint to `HT` increased the GEMM2
  median to 64.482 us and was removed;
- using device-scope `HT` for the payload store (`devhtv14`) regressed the
  fused GEMM1 median to 88.257 us and the MoE median to 209.912 us, versus
  84.888 us and 198.094 us for its interleaved baseline;
- reversing GEMM1's N-cluster order so low-K A2 blocks were written last
  (`htnrevv15`) did not improve the GEMM2 handoff: fused GEMM1 was 84.273 us,
  GEMM2 was 63.959 us, and MoE was 204.878 us, versus baseline medians of
  88.919 us for GEMM1 + quant, 57.277 us for GEMM2, and 200.670 us for MoE.

The cache-policy experiments follow the CDNA5 ISA cache-control definitions in
`mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`,
section 4.1.1 (pages 44-46): `HT=2`, `WB=3`, and `NT_HT=6`.

### Re-measured and rejected ScaleA TDM-store experiment

`scaletdmv20` staged the 768-byte ScaleA tile in the unused tail of the
existing output LDS arena and replaced the six per-wave `global_store_b16`
operations with one collective `tensor_store_from_lds`. The payload store and
its `HT` temporal hint were unchanged.

The generated E64 ISA confirmed that the intended instruction changes were
present:

| Static item | `htv10` | `scaletdmv20` |
|---|---:|---:|
| Instructions | 4,982 | 4,879 |
| `global_store_b16` | 6 | 0 |
| `s_wait_xcnt 0x0` | 6 | 0 |
| `s_wait_storecnt_dscnt 0x0` | 1 | 0 |
| `tensor_store_from_lds` | 1 | 2 |
| LDS bytes | 243,712 | 243,712 |
| VGPRs | 640 | 640 |
| SGPRs | 62 | 64 |

Random seeds 0, 1, and 2 and const0 all produced exact payload bytes, exact
ScaleA bytes, bitwise-identical GEMM2 valid rows, and bitwise-identical MoE
output.

The earlier interleaved-run numbers reported for v20 are superseded. The
authoritative remeasurement used the requested `run_moe_prefill_switch_ab.sh`
path with its default 20 profiler iterations. A standalone-quant baseline was
measured immediately before v20 and again immediately after it. Each command
was gated by a host GPU/KFD idle check, and the post-run check was also idle.

Commands:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=0 \
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048

ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| Case | GEMM1 samples (us) | GEMM1 median (us) | quant samples (us) | GEMM1 + quant median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|---:|---|
| baseline before | 78.784, 81.215, 79.290 | 79.290 | 13.200, 12.500, 13.900 | 93.190 | 58.817, 60.551, 60.449 | 60.449 | 212.21, 215.27, 214.32 | 214.32 | bitwise equal |
| fused `scaletdmv20` | 94.093, 98.333, 94.068 | 94.093 | removed | 94.093 | 72.400, 69.480, 69.528 | 69.528 | 226.79, 228.99, 226.98 | 226.98 | bitwise equal |
| baseline after | 79.862, 78.171, 79.758 | 79.758 | 11.800, 12.900, 14.300 | 91.662 | 61.071, 59.511, 60.043 | 60.043 | 211.49, 209.83, 220.64 | 211.49 | bitwise equal |

Against the baseline immediately before it, v20 regressed the producer by
0.903 us (0.97%), GEMM2 by 9.079 us (15.02%), and MoE by 12.660 us (5.91%).
Against the baseline immediately after it, v20 regressed the producer by
2.431 us (2.65%), GEMM2 by 9.485 us (15.80%), and MoE by 15.490 us (7.32%).
Removing the explicit vector-store waits did not improve the requested MoE
benchmark, so v20 was rejected and `htv10` was restored.

The tested v20 source SHA256 was:

```text
5ae9c1bd7c1c320f3c88d10b3418d70798b1d37177b0990c9dcb41b4df2b8564
```

### Rejected fused-output `TILES_PER_GROUP=8` experiment

This experiment changed only the fused GEMM1 launch swizzle from the tuned
16-M-tile group to an 8-M-tile group.  The intent was to finish all N slices
for a smaller M region closer together in time, making GEMM2's A payload more
recent in cache.  The BF16 and GEMM2 schedules retained their original
16-M-tile group.  Random validation remained byte-exact:

```text
payload byte mismatch=0
ScaleA byte mismatch=0
GEMM2 valid-route output bitwise mismatch=0
MoE output bitwise mismatch=0
```

The GPU/KFD idle check passed before and after this three-round, 100-iteration
interleaved A/B run:

| Case | GEMM1 samples (us) | quant samples (us) | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---|---|---:|---:|---:|
| baseline | 79.735, 75.138, 75.605 | 11.167, 8.590, 8.707 | 84.312 | 56.519 | 197.052 |
| fused TPG8 | 97.959, 84.132, 85.642 | removed | 85.642 | 61.044 | 205.031 |

The smaller group reduced B reuse and did not recover the GEMM2 handoff.  It
was rejected and the retained `apreqb16batchs16direct_htv10` source was
restored.

### Rejected packed-BF16 integer amax experiment

The `bf16umaxv17` experiment replaced each MX block's FP32 `abs`/max tree with
packed unsigned-16 max operations over the already rounded BF16 values.  It
then converted only the winning BF16 magnitude to FP32 for the unchanged E8M0
RoundUp operation.  This was byte-exact for the random diagnostic, including
payload, ScaleA, valid GEMM2 rows, and final MoE output.

The three-round, 100-iteration result was:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline control | 84.578 | 56.940 | 196.172 |
| fused `bf16umaxv17` | 82.918 | 62.006 | 201.679 |

An immediately following v10 control run measured:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline control | 84.768 | 56.703 | 196.375 |
| fused `apreqb16batchs16direct_htv10` | 82.829 | 59.301 | 199.623 |

The integer reduction did not reduce fused GEMM1 time and made the downstream
handoff less favorable.  It was rejected and v10 was restored.

### Rejected payload `NT_WB` store experiment

The `ntwbv18` experiment changed only the fused payload TDM store cache policy
from `HT=2` to `NT_WB=7`.  CDNA5 defines this as non-temporal in near caches
and write-back/high-priority retention in the far cache.  Random validation
remained byte-exact, but the three-round, 100-iteration interleaved result was:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline control | 85.306 | 57.749 | 197.269 |
| fused `ntwbv18` | 87.175 | 62.913 | 207.416 |

`NT_WB` regressed both the producer and consumer, so the payload store remains
`HT=2` in the retained v10 implementation.

### Rejected reverse-M-group launch experiment

The `mgrevv19` experiment preserved the 16-M-tile group and its N-major order,
but emitted complete M groups in reverse order for full-group shapes.  The goal
was to write low-M payload last because GEMM2 begins by consuming low-M tiles.
Partial-tail shapes retained the original mapping.  Random validation remained
byte-exact.

The first three-round run showed a promising but small relative effect:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline control | 84.717 | 60.639 | 199.687 |
| fused `mgrevv19` | 85.599 | 59.927 | 201.049 |

GEMM2 improved by 0.712 us relative to that control, but GEMM1 lost 0.882 us
and the full pipeline still lost 1.362 us.  A second, independent five-round
run did not reproduce the consumer benefit:

| Case | GEMM1 + quant median (us) | GEMM2 median (us) | MoE median (us) |
|---|---:|---:|---:|
| baseline control | 83.933 | 55.935 | 196.780 |
| fused `mgrevv19` | 87.117 | 60.828 | 207.065 |

The hardware work distributor does not provide a reliable completion-order
guarantee from this software grid permutation, while the reversed expert/M
order weakens GEMM1 locality.  The experiment was rejected and v10 restored.

## Corrected a07-3 Baseline After Workspace Stash

The absolute baseline was remeasured on 2026-10-02 after stashing the fused
implementation.  This measurement supersedes the later approximately 196 us
interleaved-control values as the current-machine absolute baseline.  Those
faster values remain useful only as same-run relative controls because the
machine's performance state changed between runs.

Command:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| Stage | Samples (us) | Median (us) |
|---|---|---:|
| GEMM1 | 80.140, 81.004, 80.334 | 80.334 |
| GEMM2 | 59.941, 59.963, 59.672 | 59.941 |
| MoE e2e | 213.61, 215.38, 215.76 | 215.38 |

All three rounds were bitwise correct for the reported GEMM1, GEMM2, and MoE
hashes.

`run_moe_prefill_switch_ab.sh` originally removed
`AITER_FLYDSL_GEMM1_FUSED_QUANT` as part of its legacy tuning-environment
cleanup, so an external `AITER_FLYDSL_GEMM1_FUSED_QUANT=0` was silently lost.
The script now preserves this public pipeline selector and labels custom-shape
runs as `current-baseline` or `current-fused`.

Post-fix validation command:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=0 ROUNDS=1 \
  bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

The post-fix run reported:

```text
AITER_FLYDSL_GEMM1_FUSED_QUANT=0
gemm1_quant_pipeline=standalone-quant-baseline
mode=current-baseline
GEMM1=81.186 us
standalone quant=11.200 us
GEMM1 + quant=92.386 us
GEMM2=59.178 us
MoE e2e=213.18 us
logits_diff=0
rel_l2=0
```

The profiler contained
`moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2`, confirming
that the original two-kernel GEMM1-to-GEMM2 path was selected.

The script summary now also reports the standalone quant samples and the
combined `GEMM1 + quant` median.  For the stashed three-round baseline above,
the quant samples extracted from the profiler logs were `13.4`, `12.3`, and
`13.6 us`; the per-round combined-stage median was therefore `93.540 us`.

An explicit fused-v10 three-round run in the same current machine state used:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=1 ROUNDS=3 \
  bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| Stage | Fused-v10 samples (us) | Fused-v10 median (us) | Current baseline median (us) |
|---|---|---:|---:|
| GEMM1 pipeline | 93.516, 96.687, 85.506 | 93.516 | 93.540 (GEMM1 + quant) |
| GEMM2 | 66.145, 71.112, 61.213 | 66.145 | 59.941 |
| MoE e2e | 223.06, 229.17, 208.31 | 223.06 | 215.38 |

The fused producer therefore preserved the combined GEMM1-stage time in this
measurement.  The remaining end-to-end regression came from GEMM2 consuming
the fused producer's cache state, consistent with the earlier interleaved A/B
diagnosis.  All three fused rounds were bitwise correct.

## Commands and Results

### Rebooted a07-3 Baseline

The baseline was remeasured after the a07-3 reboot on 2026-10-01.  The
container repository was first restored to the exact worktree base commit:

```text
c0d98474c1a2bc937b07930d35221d7bfc8dd713
```

The three target source files had no tracked differences from that commit.
Host checks before, between, and after the measured runs reported:

```text
GPU use: 0%
VRAM allocated: 0%
KFD processes: none
```

One unrecorded compile/warm-up launch was performed first.  Each recorded round
used the following command:

```bash
docker exec -i hyg_fyd_e2e bash -lc '
  cd /app/aiter
  export ENABLE_CK=0
  export AITER_MOE_EXPERT_BALANCE=true
  export AITER_LOG_MORE=1
  export AITER_USE_GROUPED_GEMM=1
  export AITER_GROUPED_DEBUG=0
  export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
  python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench \
    --data-format a4w4 \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --act silu \
    --no-bias \
    --no-check-aot-cache \
    --iters 20 \
    --const-init 0
'
```

The table uses the profiler's in-e2e kernel averages and eager
`testGraph=False` MoE time.  The standalone quant values are printed by the
profiler at one decimal place.

| Round | GEMM1 (us) | standalone quant (us) | GEMM1 + quant (us) | GEMM2 (us) | MoE e2e (us) | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 80.832 | 11.4 | 92.232 | 60.199 | 213.379 | bitwise equal |
| 2 | 77.201 | 14.2 | 91.401 | 58.952 | 214.226 | bitwise equal |
| 3 | 81.591 | 11.1 | 92.691 | 59.925 | 214.674 | bitwise equal |
| **Median** | **80.832** | **11.4** | **92.232** | **59.925** | **214.226** | **bitwise equal** |

Baseline kernel symbols:

```text
GEMM1:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3

Standalone GEMM1-to-GEMM2 quant:
moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2

GEMM2:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2
```

All three const0 runs reported:

```text
logits_diff=0.0000e+00
rel_l2=0.0000e+00
ref_output_hash128=6bebf6409ef198fe1a0255681f4f784f
moe_output_hash128=6bebf6409ef198fe1a0255681f4f784f
```

Because the machine clocks changed during later experiments, the same baseline
was also rerun through `AITER_FLYDSL_GEMM1_FUSED_QUANT=0` between fused trials:

| Round | GEMM1 (us) | standalone quant (us) | GEMM1 + quant (us) | GEMM2 (us) | MoE e2e (us) |
|---:|---:|---:|---:|---:|---:|
| 1 | 80.945 | 12.6 | 93.545 | 59.546 | 214.999 |
| 2 | 79.107 | 12.7 | 91.807 | 59.730 | 209.224 |
| 3 | 81.410 | 13.4 | 94.810 | 60.087 | 216.896 |
| **Median** | **80.945** | **12.7** | **93.545** | **59.730** | **214.999** |

The interleaved control command is:

```bash
ITERS=20 \
  bash my_code/run_gemm1_fused_apre_quant_validation.sh baseline-const0
```

### Fused-output correctness commands

Compare the fused payload and ScaleA directly with the original standalone
quant producer for the target shape:

```bash
bash my_code/run_gemm1_fused_apre_quant_validation.sh compare-random
bash my_code/run_gemm1_fused_apre_quant_validation.sh compare-const0
```

Repeat random validation with additional seeds:

```bash
SEED=1 bash my_code/run_gemm1_fused_apre_quant_validation.sh compare-random
SEED=2 bash my_code/run_gemm1_fused_apre_quant_validation.sh compare-random
```

The diagnostic records the fused and standalone route maps independently and
compares rows in canonical route order.  This is necessary because the atomic
route scatter may assign different grouped-row numbers on consecutive launches.

Validate the other optimized prefill shapes directly:

```bash
python3 -u my_code/verify_gemm1_fused_apre_quant.py \
  --experts 96 --tokens 16384 --topk 6 \
  --model-dim 7168 --inter-dim 3072 --seed 0

python3 -u my_code/verify_gemm1_fused_apre_quant.py \
  --experts 256 --tokens 16384 --topk 8 \
  --model-dim 7168 --inter-dim 2048 --seed 0
```

Run an interleaved A/B benchmark in one Python process:

```bash
ROUNDS=3 ITERS=100 \
  bash my_code/run_gemm1_fused_apre_quant_validation.sh ab-const0
```

`baseline-const0` and the baseline side of `ab-const0` set
`AITER_FLYDSL_GEMM1_FUSED_QUANT=0`; the default is the fused path.

### Retained source hashes

```text
d2baa8844dfb9a123b292b2eb5d450e13a9e83de6517b013837aa0dade54fed2  aiter/ops/flydsl/grouped_gemm_mxfp4.py
b803747520dc4c712acf2935054de0d3a18fcf8901deec02417fb8067eb27f00  aiter/ops/flydsl/grouped_moe_gfx1250.py
02479aa9f60d6971b284f74339fcd1e600f9233a6ec622a9de83cfc2319ae8fa  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
07fe72f109f574385537fffe8de022c47207319f3a0bd388434c5c977d720c0b  my_code/benchmark_gemm1_fused_apre_quant_ab.py
9747ff59e0f76f6001d6a45ebb6917ce6a60cebf42966b2a86e3886073d12e60  my_code/run_gemm1_fused_apre_quant_validation.sh
6222007e50df27bb9973790f3b73536572cb4b0e1d7465c7abf6f962dfd1a351  my_code/verify_gemm1_fused_apre_quant.py
```

## Post-reboot three-group remeasurement: unfused baseline versus v21

The a07-3 machine was rebooted before this measurement, so both the unfused
pipeline and the retained v21 fused kernel were remeasured under the new machine
state. Each table row is one independent invocation containing three profiler
rounds. Host-side GPU/KFD checks were idle immediately before each invocation.
The checks after each invocation either remained idle or showed an unrelated
process whose reported start time was after the measured command had completed.

Unfused baseline command:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=0 ROUNDS=3 \
  bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

v21 command:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=1 ROUNDS=3 \
  bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

The tested v21 kernel suffix and source hash were:

```text
apreqb16batchs16direct_scaletdmrtv21
9e5def358591ab7e79c251b2d83748c13786fea32702fd19a2535827d2199fd9
```

| Pipeline | Group | Producer samples (us) | Producer median (us) | MoE samples (us) | MoE median (us) | Result |
|---|---:|---|---:|---|---:|---|
| unfused GEMM1 + quant | 1 | GEMM1 80.158, 77.721, 77.906; quant 11.900, 12.500, 11.900 | **90.221** | 212.26, 206.96, 208.33 | **208.33** | bitwise equal |
| unfused GEMM1 + quant | 2 | GEMM1 81.912, 79.569, 78.631; quant 10.900, 11.800, 13.700 | **92.331** | 213.70, 218.96, 213.52 | **213.70** | bitwise equal |
| unfused GEMM1 + quant | 3 | GEMM1 80.549, 79.680, 78.223; quant 12.100, 11.500, 12.400 | **91.180** | 221.95, 212.74, 211.35 | **212.74** | bitwise equal |
| fused v21 | 1 | 92.185, 92.050, 88.835 | **92.050** | 224.24, 220.57, 219.19 | **220.57** | bitwise equal |
| fused v21 | 2 | 99.138, 92.016, 98.847 | **98.847** | 227.49, 231.68, 228.82 | **228.82** | bitwise equal |
| fused v21 | 3 | 97.323, 95.537, 88.786 | **95.537** | 230.23, 225.94, 219.84 | **225.94** | bitwise equal |

Across the three group medians:

| Pipeline | Producer median of group medians (us) | MoE median of group medians (us) |
|---|---:|---:|
| unfused GEMM1 + quant | **91.180** | **212.74** |
| fused v21 | **95.537** | **225.94** |

Under this post-reboot machine state, v21 regressed producer latency by
4.357 us (4.78%) and MoE latency by 13.20 us (6.20%) relative to the unfused
pipeline. The older 83.794/87.193 us v21 measurements therefore must not be
used as the current performance baseline.

All six accepted groups reported `logits_diff=0`, `rel_l2=0`, matching GEMM1
and GEMM2 hashes, and matching MoE/reference output hashes. An earlier baseline
attempt at `20261002T141054Z` hit a GPU page fault during round 2 and is excluded
from every table and aggregate above.

## Retained v49: eight-wave latency hiding with wave-private payload stores

The new campaign baseline used the frozen four-wave v21 kernel. The first
retained change adapts the GEMM2 eight-wave latency-hiding method to fused
GEMM1 while preserving the exact target shape and output formats:

- change only the target fused E64/T1536 GEMM1 workgroup from `w2x2` to `w2x4`;
- halve each wave's N range, accumulator footprint, and SiLU/quant work;
- allow two resident waves per physical SIMD to hide XDL, TRANS, and LDS stalls;
- keep `waves_per_tensor_tdm=2`, giving every wave one input TDM job;
- keep the original two-kernel path at `w2x2`;
- stage each wave's FP4 payload contiguously in LDS and issue one independent
  output TDM per wave;
- write one ScaleA byte per `wave_n`, preserving the original four-byte
  preshuffled scale row exactly;
- leave the target dimensions, arithmetic, BF16 rounding point, MXFP4 packing,
  ScaleA layout, and GEMM2 interface unchanged.

The initial eight-wave collective-output prototype was rejected because a
single eight-wave payload descriptor read the interleaved LDS layout
incorrectly. Its ScaleA result was exact, but the payload and downstream output
were wrong. Repacking temporary LDS as `[wave][wm][256 bytes]` and using one
payload descriptor per wave fixed the issue.

Correctness was verified with random seeds 0, 1, and 2 and with const0. Every
case produced byte-identical payload and ScaleA data, bitwise-identical GEMM2
valid rows, and bitwise-identical MoE output.

Performance command:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

| Version | Group | Fused GEMM1 samples (us) | GEMM1 median (us) | MoE samples (us) | MoE median (us) |
|---|---:|---|---:|---|---:|
| four-wave v21 baseline | 1 | 96.858, 103.192, 93.843 | **96.858** | 225.85, 238.55, 223.74 | **225.85** |
| eight-wave v49 | 1 | 234.540, 83.638, 82.238 | **83.638** | 780.09, 210.99, 207.67 | **210.99** |
| eight-wave v49 | 2 | 83.022, 85.025, 86.761 | **85.025** | 209.91, 210.20, 209.14 | **209.91** |
| eight-wave v49 | 3 | 79.012, 81.059, 84.559 | **81.059** | 197.92, 209.48, 209.95 | **209.48** |

The median of the three v49 group medians is **83.638 us** for fused GEMM1 and
**209.91 us** for MoE. Relative to the campaign baseline, this reduces fused
GEMM1 latency by **13.65%** and MoE latency by **7.06%**. The first sample of
the first retained group was a large machine transient; keeping it does not
change that group's median. An earlier v49 run at `20261002T160815Z` was
discarded because another user's GPU job began at the measurement boundary.

The retained v49 kernel source SHA256 is:

```text
44323518045e800d657758203860140b18189ea455839f99fc35a4f2764bb6f7
```

## Rejected v46 experiment: v21 with tanh chunk 8

This experiment copied the validated v21 source and changed only the fused
optimized SiLU/quant path from a tanh scheduling chunk of 4 to 8. The common
helper gained an optional `chunk_size` argument whose default remains 4, so the
generic launcher and non-quant paths kept their existing schedule. A new kernel
suffix prevented reuse of the v21 JIT artifact:

```text
apreqb16batchs16direct_scaletdmrtv21_tanhc8v46
```

The tested source hashes were:

```text
888c03949b195b26508a73d00976fb3c873f79454b9684189799b6805028c7d0  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
e3819ad372c38e4cd5ae5b1ecd3dae56fbbc66c333aed191c77d3094b6aab8c6  aiter/ops/flydsl/kernels/gemm_common_gfx1250.py
```

Correctness was checked before performance measurement with random seeds 0, 1,
and 2 and with const0. Every run had bitwise-identical quant payload, ScaleA,
GEMM2 valid rows, and MoE output relative to the unfused reference.

Each performance group used:

```bash
AITER_FLYDSL_GEMM1_FUSED_QUANT=1 ROUNDS=3 \
  bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

| Version | Group | Fused GEMM1 samples (us) | Fused GEMM1 median (us) | MoE samples (us) | MoE median (us) |
|---|---:|---|---:|---|---:|
| v21, tanh chunk 4 | 1 | 92.185, 92.050, 88.835 | **92.050** | 224.24, 220.57, 219.19 | **220.57** |
| v21, tanh chunk 4 | 2 | 99.138, 92.016, 98.847 | **98.847** | 227.49, 231.68, 228.82 | **228.82** |
| v21, tanh chunk 4 | 3 | 97.323, 95.537, 88.786 | **95.537** | 230.23, 225.94, 219.84 | **225.94** |
| v46, tanh chunk 8 | 1 | 87.062, 95.146, 93.236 | **93.236** | 211.61, 230.87, 219.11 | **219.11** |
| v46, tanh chunk 8 | 2 | 95.080, 96.837, 98.614 | **96.837** | 232.22, 227.77, 229.35 | **229.35** |
| v46, tanh chunk 8 | 3 | 100.988, 91.397, 96.860 | **96.860** | 238.97, 217.14, 233.03 | **233.03** |

| Version | GEMM1 median of group medians (us) | MoE median of group medians (us) |
|---|---:|---:|
| v21, tanh chunk 4 | **95.537** | **225.94** |
| v46, tanh chunk 8 | **96.837** | **229.35** |

Tanh chunk 8 regressed the fused GEMM1 by 1.300 us (1.36%) and MoE by
3.41 us (1.51%) on the median of the three independent group medians. Only one
of the three paired GEMM1 groups improved, and only one paired MoE group
improved. The change therefore has no stable performance benefit and was
rejected; the active source was restored to v21 after measurement.

## Post-reboot v56 / v111 / v115 comparison and invalid-block trace

The a07-3 machine was rebooted again before this comparison. Every accepted
performance run was preceded by `/data/yanguahe/code/gpu_users.sh` reporting no
GPU/KFD users. The common reproduction command was:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

The compared kernels were:

- v56: persistent four-task pipeline with WMMA `reuseB`;
- v111: v56 with `amdgpu-expert-scheduling-mode` disabled for this launcher;
- v115: v111 with `MMA_GROUP=2` instead of 4.

| Version | Fused GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE samples (us) | MoE median (us) |
|---|---|---:|---|---:|---|---:|
| v56 baseline | 82.844, 81.783, 78.905 | **81.783** | 76.434, 66.843, 64.818 | **66.843** | 216.16, 211.15, 203.88 | **211.15** |
| v111 no expert scheduler | 84.557, 82.596, 80.510 | **82.596** | 71.760, 65.979, 63.468 | **65.979** | 215.30, 210.31, 207.33 | **210.31** |
| v115 no expert scheduler + `MMA_GROUP=2` | 80.950, 80.733, 79.025 | **80.733** | 62.186, 66.011, 68.761 | **66.011** | 203.23, 203.89, 208.40 | **203.89** |

On this post-reboot machine state, v115 is the measured winner. Its GEMM1
median is 1.050 us (1.28%) below v56. The machine remains variable, so the
GEMM2 and MoE differences are retained as observed pipeline measurements rather
than attributed entirely to the GEMM1 change.

Random seed 0 validation for v115 was byte exact against the unfused producer:

```text
payload mismatch=0
ScaleA mismatch=0
GEMM2 valid-row bitwise mismatch=0
MoE output bitwise mismatch=0
logits_diff=3.3849e-06
rel_l2=2.6019e-03
```

The retained v115 kernel source SHA256 is:

```text
e42894e0fb12d842b55a087e8a1bcce9a1758a1be17d9d4a611f6f1a09a501fb
```

The post-reboot all-SIMD v115 ATT capture is stored at:

```text
/app/aiter/my_code/thread_trace_runs/gemm1_fused_v115_att_all_simd_a07_reboot_20261003
```

Four independent captures selected SIMD0, SIMD1, SIMD2, and SIMD3. Across all
32 observed physical slots, the longest slot window was
`SIMD0/SE0/CU1/slot0`:

| Block class | Begin | End | Lifetime cycles | Instructions | WMMA | TDM |
|---|---:|---:|---:|---:|---:|---:|
| valid | 111823 | 246431 | 134608 | 16731 | 2688 | 120 |

This strict maximum slot contains no invalid expert block. Its invalid-block
fraction is therefore exactly `0 / 134608 = 0.000000%` for both active cycles
and the complete slot window.

Because the captured maximum happened to contain only one block, the analysis
also selected the longest slot that actually ran multiple blocks. That slot was
`SIMD0/SE3/CU1/slot1`:

| Block class | Begin | End | Lifetime cycles | Instructions | WMMA | TDM |
|---|---:|---:|---:|---:|---:|---:|
| valid | 101696 | 234523 | 132827 | 16882 | 2688 | 120 |
| invalid expert | 234616 | 236030 | 1414 | 281 | 0 | 0 |

The two active lifetimes sum to 134241 cycles. The invalid block accounts for
`1414 / 134241 = 1.053329%` of active block cycles. The complete slot window is
134334 cycles, including a 93-cycle inter-block gap, so the invalid block
accounts for `1414 / 134334 = 1.052600%` of that slot's wall-clock cycle window.

The invalid-wave classification is structural rather than duration-only: each
invalid wave contains zero `v_wmma_*` and zero `tensor_*` instructions and
terminates after the scalar lookup/control path with `s_wait_tensorcnt 0x0`
followed by `s_endpgm`. A valid wave executes 2688 WMMA instructions and 120 TDM
instructions in the all-SIMD capture.

### v115 critical-slot wait and stall breakdown

The strict critical slot above contains one complete valid wave with a
134608-cycle lifetime. Summing every executed instruction of each requested
class gives:

| Instruction class | Executions | Total latency cycles | Latency / kernel | Stall cycles | Stall / kernel |
|---|---:|---:|---:|---:|---:|
| `s_wait_tensorcnt` | 117 | 1814 | **1.347617%** | 1697 | **1.260698%** |
| `s_wait_dscnt` | 832 | 15010 | **11.150897%** | 14178 | **10.532806%** |
| `s_barrier_wait` | 116 | 30030 | 22.309224% | 29914 | **22.223048%** |
| `v_wmma_*` | 2688 | 32939 | 24.470314% | 8747 | **6.498128%** |
| `s_wait_kmcnt` | 9 | 1844 | 1.369904% | 1835 | 1.363218% |
| `ds_load_*` | 5184 | 6432 | 4.778319% | 1248 | 0.927137% |

Together, all `s_wait_tensorcnt` and `s_wait_dscnt` instructions account for
16824 latency cycles, or **12.498514%** of the critical wave lifetime. Their
pure stall contribution is 15875 cycles, or **11.793504%**.

The dominant individual DS wait is the repeated hot-loop ring-reuse fence:

```text
s_wait_tensorcnt 0x2
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
tensor_load_to_lds ...
```

The `s_wait_dscnt 0x0` at code index 1344 executes 93 times and contributes
11280 stall cycles by itself, or **8.379888%** of the complete critical wave.

The largest tensor waits are:

- next-persistent-task `s_wait_tensorcnt 0x1`: 939 stall cycles;
- final output drain `s_wait_tensorcnt 0x0`: 445 stall cycles;
- repeated steady `s_wait_tensorcnt 0x2`: 257 stall cycles.

Across all 32 complete valid waves from the four SIMD captures, the weighted
stall shares are 10.459015% for `s_wait_dscnt`, 3.837586% for
`s_wait_tensorcnt`, 12.324430% for `s_barrier_wait`, and 8.381495% for
`v_wmma_*`. The critical slot shifts most peer-side tensor latency into its
barrier wait: its tensor wait share is only 1.26%, while its barrier wait share
is 22.22%.

The primary bottleneck is therefore workgroup synchronization plus LDS ring
reuse, not raw TDM completion. The repeated DS drain and workgroup rendezvous
consume 32.76% of the critical wave as stall, and all explicit wait/barrier
classes (`tensorcnt`, `dscnt`, `kmcnt`, `xcnt`, and barrier wait) consume
35.442916%. WMMA arbitration is the next material bottleneck at 6.50% stall.

## MI400 LDS segment-layout experiments: v116 through v135

The next optimization series targeted the dominant hot-loop `s_wait_dscnt 0x0`.
The design was based on the local MI400 hardware documentation rather than on a
generic LDS bank-conflict model:

- `MI400_Shader_Programming#65.txt`, section 4.7.1 (page 169), states that LDS
  has 64 banks of 4 bytes, is physically divided into 64 KiB segments, and uses
  address mapping `{Segment[2:0], SRAM_address[7:0], Bank[5:0],
  ByteInBank[1:0]}`.
- `architecture/system.txt`, section 4.1.3.2, states that the two ports may
  access different segments simultaneously for up to 512 B/cycle load
  bandwidth, while two ports contending for one segment are serialized.
- `HGEMM_Optimizations_And_Recommendations#1.txt`, pages 18-21, recommends
  placing A and B in different segments, interleaving their loads, placing the
  two resident A copies in different segments, and using the physical-to-logical
  wave permutation `(w0,w1,w2,w3,w4,w5,w6,w7) ->
  (w0,w2,w1,w3,w4,w6,w5,w7)`. It also confirms the 16-byte-per-1024-byte input
  padding used by this kernel.

The unmodified v115 all-SIMD trace had 3,981,044 total valid-wave cycles.
`s_wait_dscnt` contributed 416,378 stall cycles (10.459%), and the repeated hot
`s_wait_dscnt 0x0` averaged 122.3 cycles over 93 executions on the critical
wave.

### v116 and v117: separate the resident A halves

v116 moved the two wave-M halves of A into different 64 KiB segments. v117 also
moved each half's ScaleA beside its A data. Both preserved the arithmetic and
the persistent pipeline. v117 measured 77.068 us in its retained three-sample
run and was the better of the two layouts.

| Version | Valid-wave cycles | `s_wait_dscnt` stall | `s_barrier_wait` stall | `s_wait_tensorcnt` stall | `ds_load` stall |
|---|---:|---:|---:|---:|---:|
| v116 A split | 4,176,029 | 8.405% | 15.404% | 4.558% | 1.270% |
| v117 A + ScaleA split | 4,037,165 | 8.344% | 14.991% | 4.971% | 0.929% |

The DS waits improved, but the saved cycles moved into workgroup-barrier wait.
This established that segment placement must also preserve arrival balance
between the two resident wave slots.

### v123: cyclic four-segment packing, retained winner

v123 packs every persistent ring stage into the four existing 64 KiB segments:

- stage `s`, `wave_m=0` A/ScaleA: front of segment `s`;
- stage `s`, `wave_m=1` A/ScaleA: tail of segment `(s+1) mod 4`;
- stage `s` B/ScaleB: middle of segment `s`.

Each physical segment contains one stage's main region and the preceding stage's
second A/ScaleA half. The input footprint remains four segments, and the output
LDS arena and persistent overlap protocol are unchanged. The kernel metadata is
301,056 bytes group LDS, 128 SGPRs, 208 architectural VGPRs, zero accumulator
VGPRs reported separately, and zero scratch/private bytes.

Random seed 0 verification was byte exact:

```text
payload mismatch=0
ScaleA mismatch=0
GEMM2 valid-row bitwise mismatch=0
MoE output bitwise mismatch=0
```

The first retained v123 run was `75.940 / 81.057 / 76.094 us`, median
**76.094 us**. Its all-SIMD trace was:

| Metric | v115 | v123 | Change |
|---|---:|---:|---:|
| Critical-wave cycles | 134,608 | 123,995 | -7.88% |
| Total valid-wave cycles | 3,981,044 | 3,803,374 | -4.46% |
| Valid-wave duration median | not recorded in this comparison | 118,800 | - |
| `s_wait_dscnt` stall | 10.459% | 8.600% | -1.859 pp |
| `s_barrier_wait` stall | 12.324% | 12.401% | +0.077 pp |
| `s_wait_tensorcnt` stall | 3.838% | 5.528% | +1.690 pp |
| Hot `s_wait_dscnt 0x0` mean | 122.3 cycles | 94.3 cycles | -22.90% |

The retained source SHA256 is:

```text
c54a38e5d4bab82453deb9e3a59c3558fece84a04f7602237b2cedd3110eb6b4
```

The trace is stored at:

```text
/app/aiter/my_code/thread_trace_runs/gemm1_fused_v123_acyc4seg_att_all_simd_a07_20261003
```

### Rejected layout and scheduler probes after v123

Each version was copied from its immediate predecessor before modification. A
version was rejected if it changed the result, caused a fault, or failed to show
a stable performance improvement.

| Version | Experiment | Result |
|---|---|---|
| v118 | Increase split-A stage pitch to 80 KiB | 80.646 us; rejected |
| v119 | Change fence-cover WMMA from 8 to 10 | 78.481 us; rejected |
| v120 | Change `MMA_GROUP` from 2 to 1 | 84.391 us; rejected |
| v121 | Raise fixed `wave_m=1` `USER_PRIO` | 79.461 us; rejected |
| v122 | Issue B/ScaleB before A/ScaleA | 79.200 us; rejected |
| v124 | Swap cyclic main/tail ownership | 77.182 us; rejected |
| v125 | Force a large LDS memory clause | random payload mismatch; rejected |
| v126 | Change input pad amount to 8 bytes | random payload mismatch; rejected |
| v127 | Change input pad amount to 32 bytes | correct, 78.348 us; rejected |
| v128 | Shrink the continuous output arena | GPU memory fault; immediately restored without reset |
| v129 | Scatter output into segment tails and reduce to four segments | correct, 76.933 us; rejected |
| v130 | Scatter output while retaining the original allocation | correct, 79.181 us; rejected |
| v131 | Pairwise-XOR cyclic A ownership | correct, approximately 77.663 us; rejected |
| v132 | Add `HT` cache modifier to A TDM | correct, 78.354 us; rejected |

### v133: delta-2 A mapping

v133 placed stage `s`'s second A/ScaleA half in segment `(s+2) mod 4`. This
makes the future-stage TDM-write segment set disjoint from the carry DS-read set
in steady state. Random validation remained byte exact.

The original three-sample run was `82.072 / 76.640 / 74.723 us`, median
76.640 us. A later adjacent machine-state comparison measured v123 at 80.560 us
and v133 at 79.760 us, but the all-SIMD trace showed a structural regression:

| Metric | v123 | v133 |
|---|---:|---:|
| Critical-wave cycles | 123,995 | 131,006 |
| Total valid-wave cycles | 3,803,374 | 3,906,894 |
| Valid-wave duration median | 118,800 | 121,014 |
| `s_wait_dscnt` stall | 8.600% | 7.995% |
| `s_barrier_wait` stall | 12.401% | 14.867% |
| `s_wait_tensorcnt` stall | 5.528% | 5.788% |

The critical-wave hot wait executes 93 times and totals 7,632 latency cycles:
82.1 cycles mean, 87 cycles median, 100 cycles P90, and 127 cycles maximum. The
layout therefore reduced the local DS drain but delayed peer waves enough to add
more barrier and tensor wait than it removed. v133 was rejected.

### v134: put complete B/ScaleB in a third segment

v134 follows the aggressive HGEMM recommendation directly. For logical stage
`s`, the two A/ScaleA halves occupy segments `s` and `s+1`, while the complete
B/ScaleB tile occupies segment `s+2`. Across the four-stage ring, every physical
segment still holds exactly one copy of each region, so group LDS remains
301,056 bytes. Metadata reports 128 SGPRs, 200 architectural VGPRs, and no
scratch. Random payload, ScaleA, GEMM2 valid rows, and MoE output were all
bitwise identical.

The adjacent idle-GPU comparison was:

| Version | GEMM1 samples (us) | Median (us) | GEMM2 median (us) | MoE median (us) |
|---|---|---:|---:|---:|
| v134 third-segment B | 80.315, 78.057, 82.276 | **80.315** | 67.734 | 204.41 |
| v123 control | 75.272, 79.657, 82.632 | **79.657** | 69.459 | 212.94 |

The trace explains why v134 did not win despite improving LDS execution:

| Metric | v123 | v134 | Change |
|---|---:|---:|---:|
| Critical-wave cycles | 123,995 | 128,811 | +3.88% |
| Total valid-wave cycles | 3,803,374 | 3,846,515 | +1.13% |
| Valid-wave duration median | 118,800 | 118,085.5 | -0.60% |
| `ds_load` stall | 1.441% | 0.355% | -1.086 pp |
| `s_wait_dscnt` stall | 8.600% | 8.420% | -0.180 pp |
| `s_barrier_wait` stall | 12.401% | 13.671% | +1.270 pp |
| `s_wait_tensorcnt` stall | 5.528% | 5.808% | +0.280 pp |
| `v_wmma` stall | 8.495% | 8.816% | +0.321 pp |

The LDS change removed roughly 75% of direct `ds_load` stall, but the two-port
wave arrival pattern became less balanced. The added barrier, tensor, and WMMA
stall exceeded the saved DS cycles, so v134 was rejected as the active winner.
Its trace is retained for comparison at:

```text
/app/aiter/my_code/thread_trace_runs/gemm1_fused_v134_a3seg_att_all_simd_a07_20261003
```

### v135 through v152: load ordering, wave mapping, and scheduler follow-ups

v135 added the documented eight-wave permutation to v134. Its first performance
run looked favorable (`77.559 / 77.357 / 77.401 us`, median 77.401 us), but the
all-SIMD trace did not confirm a structural improvement: total valid-wave cycles
rose to 3,965,188, critical-wave cycles rose to 131,594, and barrier stall was
13.986%. The permutation reduced tensor wait to 4.499%, but added instruction and
barrier cost. It was rejected.

v136 and v138 tested B-to-A interleaving and opposite A/B order by resident
wave. Both changed random output and were rejected before performance testing.
The failures show that the generated partial `s_wait_dscnt` sequence is tied to
the operand issue order; source-level reordering is not automatically safe.

v137 applied the documented 0,2,1,3 wave permutation directly to v123. It was
byte exact and measured `77.745 / 80.838 / 75.259 us`, median 77.745 us. Its
trace had 3,924,340 total valid-wave cycles, 8.462% DScnt stall, 11.737% barrier
stall, and 4.593% tensor wait. Although barrier wait decreased, remapping the
full wave id expanded dynamic scalar/vector address work and increased total
cycles by 3.18% versus v123. It was rejected.

v139 copied the exact correct interleave order used by the Yadai kernel:
ScaleA then ScaleB, followed by alternating A then B payload fragments. It was
byte exact. The first run was `74.502 / 75.330 / 78.550 us`, median 75.330 us.
However, an adjacent idle-GPU control measured v123 at 78.843 us and v139 at
78.803 us, only a 0.05% difference. The trace had 3,914,341 valid-wave cycles:
DScnt improved from 8.600% to 7.632%, but barrier rose to 13.045% and tensor
wait rose to 5.831%. The apparent timing gain was not stable, so v139 was not
promoted.

The scheduler follow-ups to v139 were also rejected:

| Version | Change | Correctness | GEMM1 median | Trace conclusion |
|---|---|---|---:|---|
| v140 | Schedule two future DS reads before the first WMMA group | byte exact | 76.737 us | 3,992,970 cycles; barrier 14.607% |
| v141 | First WMMA group 4, later groups 2 | byte exact | 77.996 us | 3,984,488 cycles; barrier 15.797% |
| v142 | Split ready barrier around front WMMA | payload mismatch | not run | unsafe LDS reuse ordering |
| v144 | Split barrier with explicit `s_wait_dscnt 0` before signal | payload mismatch | not run | still unsafe with the current issue point |

Two further third-segment and wave-mapping variants remained correct but did not
improve the full trace:

| Version | Change | GEMM1 samples / median | Total valid-wave cycles | DScnt / barrier / tensor stall |
|---|---|---:|---:|---|
| v145 | Move complete B/ScaleB from `s+2` to `s+3` | 79.246, 81.612, 77.859 / **79.246 us** | 3,935,055 | 8.786% / 13.093% / 5.655% |
| v146 | Remap only logical N with 0,2,1,3 permutation | 74.123, 79.619, 76.018 / **76.018 us** | 4,026,979 | 8.165% / 16.168% / 5.680% |
| v147 | XOR the second resident wave's N half | 80.815, 74.850, 79.153 / **79.153 us** | 3,866,449 | 8.353% / 12.906% / 5.965% |

v146's low timing median was contradicted by a 5.88% increase in traced total
cycles and therefore was treated as machine variance rather than a retained
gain.

The remaining probes failed the byte-exact gate and were not performance-tested:

| Version | Change | Failure |
|---|---|---|
| v148 | Align all segment regions to 128-byte starts | payload mismatch |
| v149 | Interleave only A/B payload, preserving legacy scale order | payload mismatch |
| v150 | Combine third-segment B with exact A/B interleaving | payload mismatch |
| v151 | Reduce fence-cover WMMA from 8 to 6 | payload mismatch |
| v152 | Enable the gfx1250 `coexec` scheduler strategy | payload mismatch |

These failures are important: LLVM scheduling hints and apparently benign LDS
offset changes can alter the generated partial DScnt waits. Const0 is not a
sufficient gate for these variants; random byte-level payload validation must
run before any timing result is accepted.

v153 corrected the earlier split-barrier race. It performs
`s_wait_tensorcnt`, drains DScnt, signals the workgroup barrier, executes only
FRONT WMMA operations whose operands are already in registers, waits for every
wave, and only then issues the TDM overwrite and reads the next LDS stage. This
ordering passed all random byte-exact checks. Its three samples were
`75.443 / 82.476 / 76.518 us`, median 76.518 us. The trace, however, had
3,964,952 total valid-wave cycles: DScnt stall 8.528%, barrier stall 14.169%,
and tensor wait 6.661%. Delaying the next TDM issue until after the barrier made
the reuse safe but exposed more downstream TDM latency than the FRONT WMMA could
hide, so v153 was rejected.

After this sweep, v123 remains the retained source locally and in
`hyg_fyd_e2e:/app/aiter`. The remaining exposed time is coupled: reducing DS
stall alone repeatedly shifts more time into the all-wave reuse barrier or
per-wave TDM completion. A larger gain now requires changing the synchronization
topology, such as proven named sub-group barriers with matching producer/consumer
sets, rather than another local address permutation or scheduler hint.

Reaching 60 us from the retained 75-79 us range requires roughly another
20-24% reduction. The v123 trace assigns about 26.5% of aggregate valid-wave
time to DScnt, tensorcnt, and workgroup-barrier stall together, so that target
cannot be reached by removing one local wait or changing one address phase. The
barrier currently couples all eight waves because each stage is jointly produced
by four TDM owner groups: A, B, ScaleA, and ScaleB. Any producer may overwrite a
ring segment only after every consumer of the data sharing that segment has
finished.

The next credible architecture is a producer/consumer protocol with named DATA
and FREE barriers per reusable LDS region. It must initialize named barriers,
join every producer and consumer before a generation can complete, use separate
FREE barriers where producer membership differs, and keep the four-buffer ring
from reusing a barrier generation too early. The existing C++ Opus pipeline has
a proven `DECLARE_NAMED_BARRIERS` implementation, but the current FlyDSL kernel
has no working named-barrier allocation helper; the nearby gfx1250 FMHA code
also leaves its `_named_barrier_pair` implementation as a TODO/no-op. Attempting
this as another local reorder would risk a completion-before-join deadlock or an
LDS overwrite race. Optimization is paused at v123 until that synchronization
support is implemented and validated independently.

### Latest v123 control remeasurement

The retained v123 source was restored locally and in the a07-3
`hyg_fyd_e2e:/app/aiter` tree, with SHA256
`c54a38e5d4bab82453deb9e3a59c3558fece84a04f7602237b2cedd3110eb6b4`.
The host GPU/KFD check was idle both before and after the run.

| Metric | Samples (us) | Median (us) |
|---|---|---:|
| Fused GEMM1 | 77.737, 78.422, 75.234 | **77.737** |
| GEMM2 | 68.991, 65.263, 66.022 | **66.022** |
| MoE e2e | 209.12, 203.86, 202.98 | **203.86** |

Const0 GEMM1, GEMM2, and MoE hashes matched their references. The run artifacts
are stored at:

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T151433Z
```

### Same-session comparison: v123, v140, v146, and v153

All four versions were measured sequentially on a07-3 with the same command and
an idle host GPU/KFD check before and after every completed case. The forward
pass completed for all four versions:

| Version | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|
| v123 control | 82.309, 78.985, 77.112 | **78.985** | 73.410, 67.810, 64.437 | **67.810** | 213.10, 207.73, 201.18 | **207.73** | const0 hashes match |
| v140 interleave + DS-first-2 | 76.899, 76.590, 76.767 | **76.767** | 63.734, 66.024, 71.841 | **66.024** | 202.12, 203.40, 211.37 | **203.40** | const0 hashes match |
| v146 logical-N remap | 76.758, 82.369, 78.487 | **78.487** | 63.045, 64.089, 71.932 | **64.089** | 198.01, 208.02, 211.69 | **208.02** | const0 hashes match |
| v153 safe split barrier | 75.556, 81.253, 78.405 | **78.405** | 59.716, 68.892, 64.439 | **64.439** | 192.21, 208.90, 203.26 | **203.26** | const0 hashes match |

Relative to the v123 median in this forward pass, v140 was 2.81% faster in
GEMM1 and 2.08% faster in MoE. v146 and v153 did not provide a consistent
GEMM1/MoE improvement despite their lower GEMM2 medians.

The reverse-order stability pass produced complete second groups for v153 and
v146:

| Version | GEMM1 samples / median (us) | GEMM2 samples / median (us) | MoE samples / median (us) |
|---|---|---|---|
| v153 | 79.716, 79.385, 75.510 / **79.385** | 64.449, 68.004, 61.913 / **64.449** | 203.67, 210.75, 194.81 / **203.67** |
| v146 | 76.153, 78.763, 77.833 / **77.833** | 63.933, 61.542, 73.024 / **63.933** | 200.24, 203.24, 214.18 / **203.24** |

The reverse v140 run completed two valid rounds before the third round caused a
GPU memory fault:

| Round | GEMM1 (us) | GEMM2 (us) | MoE e2e (us) |
|---:|---:|---:|---:|
| 1 | 79.412 | 63.720 | 206.23 |
| 2 | 78.177 | 64.066 | 202.15 |

The third round exited with code 134 and:

```text
Memory access fault by GPU node-2. Reason: Page not present or supervisor privilege.
```

No further GPU work was submitted after the fault. The reverse v123 run was not
started. The active source was restored to v123 without resetting or restarting
the GPU, host, or container. Because v140 failed the repeated-run stability gate,
its lower forward-pass median is not considered an acceptable performance win.

Run directories:

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T153414Z  # v123
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T153538Z  # v140
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T153645Z  # v146
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T153803Z  # v153
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T154031Z  # v153 reverse
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T154139Z  # v146 reverse
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261003T154310Z  # v140 reverse, round 3 fault
```

The common performance reproduction command for this series is:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

## 2026-10-04 four-version remeasurement

The four requested versions were rerun sequentially on a07-3. The complete host
GPU/KFD check was idle before and after every case. Every run used three fresh
processes and the common command above. The active source was restored to v123
after the comparison.

| Version | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|
| v123 control | 78.203, 78.306, 84.100 | **78.306** | 64.607, 63.283, 71.814 | **64.607** | 202.59, 201.74, 216.41 | **202.59** | const0 hashes match |
| v140 interleave + DS-first-2 | 79.588, 79.929, 82.301 | **79.929** | 65.425, 65.868, 73.089 | **65.868** | 212.09, 208.27, 224.64 | **212.09** | const0 hashes match |
| v146 logical-N remap | 81.420, 78.204, 80.541 | **80.541** | 66.313, 66.794, 67.698 | **66.794** | 209.94, 206.55, 209.27 | **209.27** | const0 hashes match |
| v153 safe split barrier | 76.216, 75.674, 80.572 | **76.216** | 65.428, 60.964, 66.775 | **65.428** | 202.97, 195.75, 209.17 | **202.97** | const0 hashes match |

Relative to v123 in this measurement, v153 reduced GEMM1 by 2.090 us (2.67%),
but GEMM2 increased by 0.821 us and MoE increased by 0.38 us. Thus the isolated
GEMM1 gain did not translate into an end-to-end improvement. v140 and v146 were
slower in both GEMM1 and MoE. v123 remains the retained end-to-end control.

Run directories:

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T022540Z  # v123
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T022703Z  # v140
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T022828Z  # v146
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T022942Z  # v153
```

### 2026-10-04 one-command comparison rerun

The four-version runner was added as:

```text
my_code/run_gemm1_fused_variant_compare.sh
my_code/gemm1_fused_apre_quant_variants/v123.py
my_code/gemm1_fused_apre_quant_variants/v140.py
my_code/gemm1_fused_apre_quant_variants/v146.py
my_code/gemm1_fused_apre_quant_variants/v153.py
```

It verifies each source SHA256, waits for an idle GPU in six-second intervals,
runs every requested case, extracts GEMM1/GEMM2/MoE timing, writes one combined
Markdown report, and restores the kernel source that was active when the script
started. The validated invocation was:

```bash
ROUNDS=3 bash my_code/run_gemm1_fused_variant_compare.sh
```

The resulting same-run comparison was:

| Version | GEMM1 samples (us) | GEMM1 median (us) | GEMM2 samples (us) | GEMM2 median (us) | MoE e2e samples (us) | MoE median (us) | Correctness |
|---|---|---:|---|---:|---|---:|---|
| v123 control | 75.217, 77.191, 79.272 | **77.191** | 61.173, 65.177, 62.923 | **62.923** | 197.61, 202.11, 202.78 | **202.11** | const0 hashes match |
| v140 interleave + DS-first-2 | 81.422, 80.437, 78.812 | **80.437** | 69.779, 67.504, 60.612 | **67.504** | 209.26, 208.93, 199.24 | **208.93** | const0 hashes match |
| v146 logical-N remap | 76.098, 79.906, 74.102 | **76.098** | 68.173, 70.730, 63.496 | **68.173** | 205.44, 207.97, 198.33 | **205.44** | const0 hashes match |
| v153 safe split barrier | 79.632, 77.009, 79.609 | **79.609** | 65.497, 61.980, 71.013 | **65.497** | 206.19, 201.55, 214.32 | **206.19** | const0 hashes match |

In this run v146 had the lowest isolated GEMM1 median, 1.42% below v123, but
its GEMM2 median was 8.34% slower and its MoE median was 1.65% slower. v123 had
the lowest GEMM2 and MoE medians and therefore remains the end-to-end winner.

The generated report is:

```text
/app/aiter/my_code/gemm1_fused_variant_compare_runs/20261004T024422Z/summary.md
```

## 2026-10-04 post-GEMM2-port repeated stability run

The committed persistent GEMM2 port at `0df63659cb75` was measured in three
independent groups, with three fresh processes in each group. The host GPU/KFD
check was idle before and after every group. All nine const0 runs reported
`logits_diff=0`, `rel_l2=0`, and matching GEMM1, GEMM2, and final MoE hashes.

The reproduction command for each group was:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

| Group | GEMM1 samples / median (us) | GEMM2 samples / median (us) | MoE e2e samples / median (us) |
|---|---|---|---|
| 1 | 73.385, 73.402, 72.246 / **73.385** | 49.522, 53.909, 52.574 / **52.574** | 179.56, 185.41, 180.16 / **180.16** |
| 2 | 74.063, 73.376, 74.945 / **74.063** | 51.889, 51.050, 51.166 / **51.166** | 184.90, 176.32, 181.72 / **181.72** |
| 3 | 80.089, 74.325, 73.231 / **74.325** | 53.981, 50.761, 51.375 / **51.375** | 192.63, 183.92, 187.27 / **187.27** |

The aggregate statistics across all nine samples were:

| Metric | Median (us) | Mean (us) | Standard deviation (us) | CV | Min--max (us) | Group-median span |
|---|---:|---:|---:|---:|---:|---:|
| GEMM1 | **73.402** | 74.340 | 2.287 | 3.08% | 72.246--80.089 | 0.940 us / 1.27% |
| GEMM2 | **51.375** | 51.803 | 1.467 | 2.83% | 49.522--53.981 | 1.408 us / 2.74% |
| MoE e2e | **183.92** | 183.54 | 4.81 | 2.62% | 176.32--192.63 | 7.11 us / 3.91% |

The central GEMM1 and GEMM2 results were repeatable across groups, although
individual samples still showed roughly 3% coefficient of variation. The third
group contained the slowest GEMM1 and MoE samples (`80.089 us` and `192.63 us`),
which raised its MoE median. There was no fault, hang, correctness failure, or
pre-existing GPU workload during these measurements.

The matching const0 hashes in every run were:

```text
gemm1_ref_output_hash128 = c281c06c980fd4ca89d84615b26083b9
gemm1_output_hash128     = c281c06c980fd4ca89d84615b26083b9
gemm2_ref_output_hash128 = 8435f663d2aae0fe93d109c485cd9265
gemm2_output_hash128     = 8435f663d2aae0fe93d109c485cd9265
ref_output_hash128       = 6bebf6409ef198fe1a0255681f4f784f
moe_output_hash128       = 6bebf6409ef198fe1a0255681f4f784f
```

Run directories:

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T055532Z
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T055616Z
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T055659Z
```

## 2026-10-04 FlyDSL-guided persistent-kernel cleanup

The two dedicated persistent modules were reviewed against FlyDSL
`rocm/main@1941889400621f1fc3d8f03bada1a4e7380cdf7e`, using the
`flydsl-kernel-authoring`, `flydsl-tile-programming`, `kernel-code-cleanup`,
`llvm`, `gemm-optimization`, `lds-optimization`, `prefetch-data-load`, and
`api-stability` guidance.

The cleanup deliberately preserved the tuned launch ABI, kernel names, LDS
layout, TDM issue order, wait/barrier protocol, LLVM options, and persistent task
schedule. It made these source-only changes:

- Removed the unreferenced generic `launch_gemm_a8w4_tdm` copy and its compile
  hints from each dedicated persistent module. Repository-wide caller analysis
  found that only `launch_gemm_a8w4_tdm_fused_persistent` and
  `launch_gemm_a8w4_tdm_gemm2_persistent` are imported from these modules.
- Removed imports, constants, and helpers that became dead with those generic
  launchers. The two files lost approximately 3.7k lines of duplicate code.
- Replaced the GEMM2 module's remaining internal `vector.extract` use with
  `Vec(...)[sub]`.
- Localized the raw `flydsl._mlir.dialects.llvm` import to the exact
  `llvm.amdgcn.s.setreg` boundary. This raw intrinsic remains intentional because
  the available convenience helper writes a different SCHED_MODE bit.
- Retained `SharedAllocator().allocate(...)._ptr` with an explicit explanation:
  replacing it with `peek().ptr` changes address lowering for the t192
  specialization and previously caused a segmentation fault.

AST comparison of both retained launchers showed no executable differences
outside the unused-value removal, local import, and the GEMM2 vector-extract API
migration. `py_compile`, `ruff check`, and `git diff --check` all passed. The
unfused `AITER_FLYDSL_GEMM1_FUSED_QUANT=0` fallback also compiled and passed its
const0 hash checks.

Random validation after the cleanup used:

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh \
  e2e-random \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

It passed with `logits_diff=3.3849e-06` and `rel_l2=2.6019e-03`. The captured
GEMM2 reference and output hashes were both
`8cbca379ad3bb16f00c22b4cb1dd528f` in this run.

Three independent post-cleanup groups, each using the standard `ROUNDS=3`
command, produced:

| Group | GEMM1 samples / median (us) | GEMM2 samples / median (us) | MoE e2e samples / median (us) |
|---|---|---|---|
| 1 | 72.839, 75.696, 71.443 / **72.839** | 52.005, 52.125, 49.252 / **52.005** | 182.82, 186.30, 180.06 / **182.82** |
| 2 | 74.797, 75.154, 73.773 / **74.797** | 53.597, 52.159, 56.446 / **53.597** | 186.48, 184.54, 187.85 / **186.48** |
| 3 | 78.959, 73.003, 76.332 / **76.332** | 53.620, 52.098, 51.685 / **52.098** | 193.33, 184.97, 187.03 / **187.03** |

The median of the three group medians was `74.797 us` for GEMM1, `52.098 us`
for GEMM2, and `186.48 us` for MoE e2e. Across all nine samples, the medians
were `74.797 us`, `52.125 us`, and `186.30 us`, respectively. Every run was
bitwise exact for const0.

Because these measurements occurred later than the pre-cleanup stability batch,
an adjacent ABBA control was also run with one fresh process per sample:

| Version | GEMM1 samples / median (us) | GEMM2 samples / median (us) | MoE e2e samples / median (us) |
|---|---|---|---|
| Pre-cleanup control | 71.449, 75.000, 74.133 / **74.133** | 50.380, 53.024, 52.298 / **52.298** | 179.83, 182.29, 183.92 / **182.29** |
| Cleanup | 71.283, 75.805, 70.895 / **71.283** | 51.409, 53.987, 50.688 / **51.409** | 181.15, 191.10, 179.26 / **181.15** |

The adjacent comparison shows no performance regression: the cleanup medians
were 3.84% lower for GEMM1, 1.70% lower for GEMM2, and 0.63% lower for MoE e2e.
These differences are treated as normal machine variance rather than an
optimization because the retained specialization bodies and generated kernel
symbols are unchanged.

Post-cleanup three-group run directories:

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T063851Z
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T063942Z
/app/aiter/my_code/moe_prefill_switch_ab_runs/20261004T064030Z
```

The adjacent comparison logs are stored under:

```text
/tmp/flydsl_refactor_ab_20261004/
```
