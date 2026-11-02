# mHC through attention reduction: Flash C1

Implemented as `full_prefix`, one kernel launch with 256 CTAs. The complete measured scope includes mHC learned projection and Sinkhorn, stream mixing, input RMSNorm/quantization, QA/KV projections and split-K reduction, weighted RMSNorm, latent quantization, QB projection, Q RMSNorm, Q/KV RoPE, persistent cache write, split-KV attention, and final attention reduction. All cases use split-KV 4.

The unfused control is a new 17-launch WMMA/TDM control using the same arithmetic and tiling, not the historical vector prototype and not native ATOM/AITER serving. The three-region arm launches mHC, QKV, and attention separately. All learned projections use WMMA; TDM tiled loads and QB/cache L2 prefetch remain enabled.

## Whole prefix, second round

GPU-event elapsed time per complete prefix, 128 calls per captured graph, 5 warmups, 40 alternating-order samples, median. The first round is retained but ignored. Includes device gaps within the graph.

| Context / weights | Unfused, 17 launches (us) | Three regions, 3 launches (us) | Fully fused, 1 launch (us) |
|---|---:|---:|---:|
| 128 tokens, reused | 38.425 | 37.145 | 37.326 |
| 1024 tokens, reused | 44.841 | 48.258 | 48.391 |
| 128 tokens, 16 rotating copies | 39.975 | 41.034 | 40.171 |

Rotating copies include QA/KV, QB, and mHC projection weights (632 MiB total); identical values at different addresses, not actual different model layers. Scale and norm arrays and the 128-token cache remain shared/reused.

Single-launch fusion reduces the reused 128-token measurement by 2.9%, is approximately tied/slightly slower with rotating weights, and is slower at 1024 tokens. This is not a consistent whole-prefix speedup. The three-region arm remains competitive. No serving throughput or model-quality claim.

## Group table

The first three rows are newly measured at 128 tokens, split-KV 4. These are second-round profiler sums of kernel GPU durations (median of 30 one-invocation graph replays), matching the earlier group-table convention. They exclude inter-kernel gaps and use different replay/cache conditions than the batched GPU-event prefix measurements above; do not add them to predict prefix latency. Raw batched per-group event measurements are also retained in timings.json.

| Unfused kernels / operations | Fused region | Unfused summed GPU time (us) | Fused GPU time (us) |
|---|---|---:|---:|
| mHC WMMA projection → split reduction + input RMS/scale/bias → Sinkhorn → stream mixing → weighted RMSNorm | mHC input | 13.315 | 15.062 |
| Input quant → QA/KV WMMA → split-K reduce → RMSNorm → latent quant → QB WMMA → Q RMSNorm → Q RoPE + KV RoPE | QKV | 16.675 | 11.456 |
| Cache store → split-KV attention partials → attention reduce | Attention + reduce | 8.410 | 10.535 |
| Inverse RoPE → quant → OA → quant → OB | Attention output projections — next | Not rerun | Not rerun |
| mHC post → mHC projection → Sinkhorn → stream mixing → RMSNorm | Attention residual + FFN input | Not rerun | Not rerun |
| Router + shared expert + routed experts | MoE | Not rerun | Not rerun |
| Expert sum + shared add → mHC post | Final residual | Not rerun | Not rerun |

Input quant now belongs to the QKV row; the earlier table counted it in mHC. Prior remaining-group measurements are preserved in ../full-layer-c1/regions/REPORT.md and ../full-layer-c1/all-wmma/REPORT.md, with their original vector controls and arithmetic limitations. They are not comparable fresh WMMA/TDM measurements. The next pipeline group is attention output projections; within the measured prefix, mHC is an important optimization target.

## Validation and resources

- 64 cases: nine checkpoint embedding inputs and 23 synthetic residual inputs at each of 128 and 1024 tokens. Fused and three-region intermediates/output match the stage-separated control within the unchanged 0.001 relative RMS gate.
- 9,216 changing-input graph invocations with deliberately stale current-token cache rows pass; maximum output discrepancy from split control is zero.
- Independent FP64 mHC projection reference maximum relative RMS: 4.374184641164902e-07.
- Independent FP64 attention reference maximum relative RMS: 0.00011608856584643945.
- Numerical equivalence to this control does not establish full-layer native ATOM equivalence. The prior all-WMMA full-layer downstream rounding issue is not resolved by this experiment.
- Trace confirms 17 / 3 / 1 launches. Fully fused ASM retains TDM, WMMA, and L2 prefetch.
- Fully fused resource metadata: {"t128_fused": {"vgpr": 419, "lds": 131584, "spills": 0}, "t1024_fused": {"vgpr": 419, "lds": 131584, "spills": 0}}.

## Reproduction

Run `bash run.sh` inside the supplied gfx1250 container with this directory mounted as /work/mhc-through-attention, after checking GPU0 is idle. Standalone fixtures, code, skills, upstream MIT source/license, raw first/second-round samples, traces, assembly, LLVM, and checksums are included. No server, image, git branch, or serving script was changed.
