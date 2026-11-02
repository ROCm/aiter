# Q/KV preparation follow-up tuning

Applied the earlier QA lessons to QB: direct WMMA operand layouts, packed scaled FP8->BF16 conversion and.ca were already present; this change explicitly unrolls the Kloop to expose more independent weight loads and WMMA work, and retunes output tile size with that schedule. QA N128/split16 atomic protocol is retained in the recommendation.

Second-round whole-chain timings (microseconds), same-pass comparisons:

| Working set | Previous QA1/QB64U1 | QB128U2 | Recommended QB128U4 | QA2/QB128U8 |
|---|---:|---:|---:|---:|
| 1 copies | 10.393 | 9.689 | 9.377 | 9.226 |
| 16 copies | 12.566 | 12.030 | 11.225 | 11.426 |

Recommended four-way unroll:9.78%lower latency for one38MiB weightcopy,10.68%lower for16copies/608MiB. Exact timings are in final-benchmark/timings.json. Weightcopies have identical values at different addresses, not distinct layers. Scales/gamma remain shared. No forced-cold-cache claim.

Single-copy fastest is fullunroll8 with QAunroll2, but it uses827QB VGPRs versus273forunroll4 and is slower at16copies. Therefore default isolated two-launch API uses unroll4 and keeps QAunroll1. The single-launch API remains the prior unroll1 implementation; no high-register persistent-grid residency assumption was introduced.

All268input cases x3replays x4candidates passed independent FP64-projection reference checks, including changing rotary tables. Unchanged0.1%relative-RMS gate for Q/KV separately. Recommended and fullunroll8 variants each also passed10240checked changing-input stress invocations. Standalone API and graph path checked. See final-validation/, stress/, api-validation.json.

Compiled QB resources: prior N64U1=86VGPR, unroll2N128=142, unroll4N128=273, fullunroll8N128=827. All zeroSGPR/VGPRspills, zero private scratch,2048bytesLDS. QA remains112VGPR/8192LDS for recommendation; QAunroll2uses145VGPR. Saved AMDGCN/LLVM in final-benchmark/. Static WMMA instruction sites4/16/32/64respectively; firstthree retain Kloop, fullunroll8has no Kloop. Static counts are not executed-operation totals. Four-wave mapping: recommended256CTAs x4waves x(32sites x2loopiterations)=same65536WMMA instruction executions as prior512CTAs x4waves x(4sites x8iterations). Improvement comes from scheduling/work distribution, not a reduction in useful matrix arithmetic. Counter increments halve512->256, and per-head output fragments halve8->4. No global_wb in these QB dumps. Larger unroll provides more overlap but costs registers; no hardware-counter measurement is claimed.

## Remaining opportunities, prioritized

1. QB pipeline tuning:32MiB of38MiB projection weights are QB; original standalone QB+norm+RoPE was6.065us vs3.929us for extendedQA. Explore intermediate unroll/prefetch schedules to retain overlap with fewer registers. Dedicated L2prefetch/TDM staging was not tested in this follow-up; earlier QA-only prefetch/TDM attempts lost, so do not assume a win.
2. Remove unused QA FP32 normalized diagnostic stores (1536floats=6KiB). Keep latent FP8 and finalKV; normal production consumer needs these, not diagnostic Y. Not yet measured; likely modest total bytes but on finishingCTA critical path.
3. Store QB completed projection in BF16 scratch instead of FP32 containing BF16-rounded values, reducing scratch128KiB->64KiB. Semantically plausible because current code already rounds; must verify and benchmark. Not yet implemented.
4. Repeated initial input quantization is12x per128elementblock across outputtiles. Prior separate-quant3launch experiment was slower, so replacing recomputation with a global rendezvous is not an established improvement. Producer quantization reuse or preceding-region fusion needs its own A/B.
5. For one-launch follow-up, consider readiness tied to Qlatent rather than full-grid rendezvous, possibly overlap QB weight prefetch with QA. Requires bounded polls, device ordering and proved residency; not implemented. Do not move the827VGPRvariant blindly into a persistent grid.

No reason yet to reintroduce intermediate sumsq mailboxes. QA/KV atomic sums must be complete before squaring; squares of Kpartials omit cross terms. Current finishing CTA owns norm and following quant/rope, eliminating another synchronization phase. See FLOW.md for full table and ASCII diagram.

This is Flash C1 singleGPU isolated Q/KV preparation. No attention, cachewrite, MoE or servingintegration; no TPOT/TTFT/tok/s claims. No image/branchpublication.
