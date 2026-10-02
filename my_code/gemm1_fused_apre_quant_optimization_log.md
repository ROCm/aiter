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
