# FP8 softmax and H128 experiments, 2026-10-21

Selected candidate: `fma`, based on commit
`c3ac80f73c182858547392fdec25f9131570c9fa`.
The retained change and current performance are recorded below. Rejected
experiments remain outside the production kernel, with source snapshots,
test logs and measurements in the remote results directory.

## Base-2 softmax

The previous kernel already used `__builtin_amdgcn_exp2f` and supplied
`softmax_scale * log2(e)` once to the attention body. Its scale/subtract
compiled into separate `v_mul_f32` and `v_sub_f32` because the translation
unit uses `-ffp-contract=off`. The new explicit `__builtin_fmaf` produces
one `v_fma_f32` followed by one `v_exp_f32` per score. `MUFU.EX2` is the
NVIDIA instruction name, not the MI355X instruction name.

```cpp
// alpha2 is precomputed; m2 is the online maximum in base-2 units.
p = __builtin_amdgcn_exp2f(__builtin_fmaf(score, alpha2, -m2));
```

For an N32 tile, each wave has eight scores. This removes eight VALU
instructions per wave per KV block. It does not remove the row max/sum,
O rescaling, conversions, KV traffic, barriers or matrix instructions.
The QK and PV precision remains scaled FP8 and BF16 respectively.
FMA changes rounding, so byte identity with the old kernel is not an
accuracy requirement for this candidate. Existing oracle tolerances are
unchanged. The maximum six-case output relative L2 change after output
quantization is 4.6934452e-05.

## Full-path measurements

MI355X GPU 6, serial runs, unlocked clocks, microseconds. T is batch times
query tokens per request; H is query heads; S is long-cache scope length.
All KV is FP8. K is 640 for H64 and 1152 for H128.

| T | H | S | Previous paged | Selected paged | Previous row | Selected row | PR full row | Selected row / PR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 560 | 64 | 8192 | 129.00 | 128.31 | 133.19 | 132.65 | 113.03 | 1.174 |
| 560 | 64 | 32768 | 132.65 | 132.10 | 136.36 | 136.63 | 115.41 | 1.184 |
| 960 | 64 | 8192 | 161.06 | 160.67 | 164.59 | 164.25 | 172.72 | 0.951 |
| 960 | 64 | 32768 | 164.19 | 163.75 | 167.12 | 166.62 | 180.57 | 0.923 |
| 320 | 128 | 8192 | 208.19 | 207.59 | 213.21 | 212.27 | 153.69 | 1.381 |
| 320 | 128 | 32768 | 220.36 | 215.94 | 235.97 | 235.89 | 157.08 | 1.502 |

Native PR #6675 uses row-packed KV, so compare it with the row columns.
Full timing includes Q RoPE, group32 Q quantization, attention, inverse
RoPE and output quantization. PR full also includes Q preparation. The
fused kernel writes max/LSE; PR timing does not request optional LSE.
Allocation, metadata construction, cache layout conversion, cache writer
and GEMM are excluded. Timing uses balanced graph orders and poisoned
output replay checks. Small differences around one percent are sensitive
to the unlocked clocks; instruction-count reductions alone do not prove
a corresponding end-to-end speedup.

## Architecture work considered

| Direction | State before this round / evidence |
|---|---|
| Producer and compute wave specialization | No dedicated producer-only wave in the retained baseline. H128 shared-load CTA experiments did not improve full-path timing. |
| Multiple LDS slots | Retained baseline has two raw FP8 K slots and one BF16 V slot, 67,840 bytes total. |
| Next KV load overlapped with current compute | Already present as raw register prefetch overlapping QK. Direct global-to-LDS variants were also tested this round. |
| Matrix QK and PV/SV | Already present: scaled FP8 QK MFMA and BF16 PV MFMA. |
| Online softmax | Already present, including base-2 state and output rescale. |
| Resident O | Already present, explicitly constrained to VGPR across QK/PV. |
| External one-time KV dequantization | A separate gather/dequant plus BF16 fused path was implemented in the earlier decode/prefill work. Its value depends on reuse and the number of dequantized rows. |
| Cross-block QK/PV interleaving | Added as an experiment this round; see candidate test and timing records below. |

Candidate families:

- `shared128`: one eight-wave CTA for 128 heads; passed correctness but slower.
- `cross4`, `cross4pipe`, `cross8pipe`: exchange P across waves and partition V
  channels. All passed correctness. Streaming reduced the register excess,
  but these variants were slower overall than the retained baseline.
- `overlap`: interleave current V dequantization with QK. Passed correctness,
  but did not improve the measured full paths.
- `dma2`: `global_load_lds_dwordx4`, 64-bit global addresses and a blocked raw
  K layout. Passed correctness; lowered metadata VGPR to 221, but did not
  provide a general H128 improvement. The S32768 row case improved about 6%.
- `dmacross`: DMA plus cross-wave P exchange and V reuse; passed correctness,
  241 VGPR, with similar limited benefit and no general H128 improvement.
- `n64`: double the KV tile to halve softmax/barrier frequency. Its first
  version failed the independent oracle and is not eligible for selection.
- `fma`, `fmabatch`: explicit FMA; the second forces a boundary between all
  FMAs and all exponents. Both passed; scheduling fence had no stable advantage.
- `interleave` and any subsequent variant: compute QK(k+1) among PV(k)
  matrix instructions. The first partial-wait schedule failed the max-logit
  oracle. It is retained only as a failed experiment. Later variants have
  separate source, fresh JIT directories, test logs and timing records.

The machine-readable `candidate_summary.json` records the actual test
status, binary hashes, resources and same-run timings for all candidates.
No performance number is reported for a candidate that failed correctness.

## Validation and reproduction

- 253 native regressions pass, including all 34 FP8 QK cases.
- Full fixtures check quantized-Q dense oracles, max and LSE, both cache
  layouts, internal/prepared Q consistency and graph replay poisoning.
- Six supplemental shared-cache fixtures cover T1/T64/T1024 and H64/H128.
- ISA and metadata for all selected specializations are archived. This
  kernel-level validation is not a model-level accuracy evaluation.
- The API remains opt-in through `fp8_qk=True`; default BF16 QK is unchanged.

Selected binary and measured source are recorded in `provenance.json`.
Use a fresh `AITER_JIT_DIR` for each source candidate: the existing extension
binary is not rebuilt automatically after a header change.

```bash
HIP_VISIBLE_DEVICES=6 GPU_ARCHS=gfx950 MAX_JOBS=8 \
AITER_JIT_DIR=/tmp/dsv41-fp8-new-build \
AITER_CK_DIR=/home/jun_chen2_qle/yzhou_aiter/aiter/3rdparty/composable_kernel \
PYTHONPATH=$PWD:$PWD/op_tests OMP_NUM_THREADS=4 \
/opt/venv/bin/python3 -m pytest \
 op_tests/test_fused_dsv41_mxfp8.py op_tests/test_fused_dsv41_paged.py \
 op_tests/test_fused_dsv41_bf16.py op_tests/test_fused_dsv41_fp8_qk.py -q
```

Benchmark adapters depend on the pinned fixture and PR helper checkouts
under `/home/jun_chen2_qle/yzhou_aiter`. They are evidence and reproduction
adapters for this environment, not standalone portable benchmarks.
PR comparison commit is `b92f3cc9eeff2b5a5c56741a15da003dc5640d2e`.
No branch is pushed by this work.
