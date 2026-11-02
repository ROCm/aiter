# TDM for all QKV GEMM weight tiles

The previous kernel inherited direct weight loads from the WMMA prototype. The earlier first-tile experiment did not isolate early versus late TDM. This revision converts **all QA, KV and QB GEMM weight tiles** to TDM and adds a matched early/late QB comparison. It remains the isolated Flash C1 fully fused QKV preparation region, not full attention/MoE or serving.

## Implementation

- QA/KV:192 producer CTAs,128 Q +64 KV, split16, N128/K128. Each stages both FP8[128,128] tiles through TDM into separate padded LDS slots, then reads LDS for BF16 upcast and WMMA. Each slot has16KiB payload/18KiB padded allocation.
- QB:256 CTAs, N128,K1024. All eight K128 tiles now use TDM, not just block zero. Two padded LDS slots form a pipeline. Preload blocks0/1; for each block wait for the oldest transfer, read LDS, barrier before recycling that slot, issue block+2 when present, and perform upcast/WMMA. Wait zero for the final block. No direct global weight loads survive in this specialization.
- Early variant starts QB blocks0/1 after producer publication/arrival ticket but before last-arriver RMSNorm and before other CTAs poll. The64 CTAs with no QA/KV work stage before polling directly. Late variant starts the same two transfers after polling. Both use the same LDS layout, tile sizes, pipeline, consumer and arithmetic. This is a source-level placement ablation, not a claim of byte-identical scheduling.
- L2 prefetch hints are disabled for all reported variants, separating the TDM issue-timing experiment.
- Direct accesses remain for genuinely small operands:128-element input and latent-vector blocks, scalar E8M0 scales, norm/rotary vectors, vector outputs, fixed-order split-K reduction vectors, and synchronization. The logical C1 activation operand is one row, broadcast for WMMA; it is not a dense16-row matrix in memory. Split-K partial vectors are still reduced directly; this does not convert every scratch or output access to TDM.
- Fixed-order FP32 split-K, BF16 WMMA, numerical rounding boundaries and device-scope load-poll barrier are unchanged.

## Results (second timing round only)

| Path | Reused weights (us) | 16 rotating copies (us) |
|---|---:|---:|
| global_control | 10.8337 | 12.6613 |
| tdm_late | 10.5954 | 13.0469 |
| tdm_early | 10.3488 | 13.1569 |

With reused weights, early TDM is about4.5% faster than the old direct-load control, and2.3% faster than late TDM. With16 rotating copies, early TDM is about3.9% slower than direct loads and0.8% slower than late TDM. Thus early staging helps the reused-weight test but does not establish a larger-working-set win. No hardware-counter attribution of the regression was performed.

The default development API now uses TDM for all GEMM weights as requested, with early QB staging. The original global-load code remains an explicit benchmark control. This choice reflects the requested design direction, not a claim that TDM is currently faster in every regime.

Method: one and16 identical weight copies at distinct GPU addresses,38MiB/608MiB total QA+QB weights;128 calls per graph,5 graph warmups,40 alternating-order samples,2 rounds. Only round2 medians reported. This is not an actual model layer sequence or serving run. Read raw timings for variation. Device was idle before execution and after completion.

## Verification and resources

All three paths passed268 varied cases ×3 replays and10240 changing-input graph invocations each, including the known quantization-threshold counterexample. Relative-RMS acceptance threshold0.001 unchanged; health and arrival counters clean. API replay checks cover all three modes. Exact metrics in scratch-validation/summary.json, scratch-stress/results.json and api-validation.json.

TDM early and late both compile to343VGPR,73856 bytes LDS (72.125KiB), zero spills/private bytes. Direct control:289VGPR,8192 bytes LDS, zero spills. Assembly shows tensor_load_to_lds for weight transfers and no instructions attributed to either direct-weight-load source line in TDM specializations. Existing direct small-vector/scalar loads remain intentionally. See isa-verification.json and complete assembly/LLVM IR in scratch-benchmark.

## Reproduce

Run ./run.sh in /work/qkv-all-tdm inside vigokhal_dsv4_tp4, with the GPU idle and no concurrent persistent grids. Fixture includes necessary weights and inputs. The archive includes sources, fixture, raw timings, validation, ASM/LLVM and environment. No checkpoint download, server restart, git push or image change was performed.
