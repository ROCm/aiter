# mHC optimization: gfx1250 Flash C1

Completed parallel experiments on four otherwise idle GPUs. Final performance table below is a matched GPU0 comparison, second measurement round only. The user-requested scope is the standalone mHC group and the one-launch mHC-through-attention-reduction prefix; this is not native ATOM serving integration.

## Accepted implementation

- Keep 256 WMMA/TDM projection producers, with original 24x16384 FP32 checkpoint weights split into BF16 high/low operands. No accepted weight repacking or split-K change.
- Before projection, the two tail CTAs issue TDM loads for the 32 KiB residual and 8 KiB input norm weights. These data are ready when the projection partials arrive.
- Reduce four internal mHC grid barriers to one. After publication of the partials, CTA0 computes the four pre coefficients, BF16 stream mix, and weighted input RMSNorm. CTA1 independently computes the persistent pre/post/comb coefficients and Sinkhorn. Duplication of the small pre/RMS calculation removes dependent global handoffs.
- Use AITER-style hardware reciprocal multiplication for the 19 repeated Sinkhorn row/column normalization iterations. Initial normalizations retain division.
- Preserve the exact original residual RMS reduction order: reshape each 4096-element stream into four 1024-element chunks; reduce each chunk using the original tree; accumulate the sixteen scalar sums sequentially. Avoid expensive 4096-to-1024 gathers. Preserve BF16 mixed-residual and normalized-input boundaries.
- The existing mHC-to-QA publication barrier remains in the complete prefix. QA/KV, QB, attention, and split-KV reduction retain WMMA/TDM and the existing QB/cache L2 prefetch.
- For tested T=128, `opt_both` additionally prefetches all QA/KV weights into L2 and stages each producer's two QA/KV weight tiles into LDS before mHC. For tested T=1024, `opt` is faster and omits only this additional QA prefetch/staging. This is an explicit experiment selection, not an untested general dispatch heuristic.

## mHC only, GPU0, second round

Batched graph GPU-event timing: 256 calls per graph, 5 warmups, 40 alternating-order samples per round, median. Rotating uses 256 copies of the mHC learned weights (384 MiB); residual/norm/scale inputs reused.

| Weights | Unfused, 5 launches (us) | Original fused (us) | Optimized fused (us) | Reduction vs original fused |
|---|---:|---:|---:|---:|
| Reused | 12.737 | 14.568 | 5.821 | 60.05% |
| 256 rotating copies | 13.098 | 15.116 | 6.403 | 57.64% |

Independent GPU1 check (128 calls/graph, second round) measured 12.960 / 15.007 / 6.222 us for unfused / old fused / optimized. GPU3 reproduced approximately 5.85 us reused and 6.40 us rotating. These corroborating runs use different devices/replay batches and are not mixed into the primary GPU0 table.

## Full prefix: mHC through attention reduction

Split-KV 4 throughout. GPU0 second round, 128 calls per captured graph, 5 warmups, 40 alternating-order samples. Rotating weights include mHC, QA/KV, and QB, 16 identical-value copies at different addresses (632 MiB); this is a cache-pressure proxy, not different actual model layers.

| Context / weights | Unfused, 17 launches (us) | Original single launch (us) | Optimized single launch (us) | Mode |
|---|---:|---:|---:|---|
| 128 tokens, reused | 38.310 | 37.414 | 29.522 | opt_both |
| 1024 tokens, reused | 45.567 | 50.679 | 40.008 | opt |
| 128 tokens, 16 rotating copies | 39.973 | 40.796 | 32.770 | opt_both |

The fused prefix includes persistent cache update and split-KV reduction; profiling confirms `optimized_prefix` is one launch. The unfused path is a matching WMMA/TDM development control, not the native ATOM/AITER launch sequence. No TTFT/TPOT/tokens-per-second result is implied.

## Additional QA prefetch ablation

| Mode | T128 reused (us) | T128 rotating16 (us) | T1024 reused (us) |
|---|---:|---:|---:|
| opt | 30.590 | 33.381 | 40.008 |
| opt_l2 | 30.299 | 33.545 | 41.034 |
| opt_both | 29.522 | 32.770 | 44.165 |

`opt`: optimized mHC with early residual/norm TDM. `opt_l2`: also QA/KV L2 hints. `opt_both`: also early QA/KV weight TDM staging. Existing QB/cache L2 prefetch is enabled in every arm.

## Correctness

- 521 GPU0 cases and 521 independent GPU3 cases (different random seeds; embedding cases overlap) passed the unchanged 0.001 relative-RMS gate for the complete prefix.
- Another 521 cases validate `opt_both`; 64 additional cases validate selected T1024 `opt`.
- 10,240 changing-input full-prefix graph checks each for `opt` and `opt_both`, poisoning Mix/Raw/X and the current cache slot before every call. Health/counters are clean. Raw/X/Q/KV/output/cache match bitwise in stress; only the post/Sinkhorn coefficient outputs have tiny FP32 differences (maximum relative RMS 1.47e-07).
- Final prefix has 419 VGPRs, 131584-byte LDS, zero spills. Standalone optimized mHC has 267 VGPRs, 56496-byte LDS, zero spills (independent compiled check).
- Assembly confirms WMMA, TDM, device-scope prefetch, and three fewer global atomic-add sites (the removed grid barriers): original10, optimized7.
- An initial ~6 us version reassociated the residual RMS sum; it failed a larger prefix test because a tiny coefficient change crossed a BF16/FP8 threshold. It is rejected. The selected reshape implementation restores the original reduction tree/order, and passes the larger test.
- These checks establish this prefix against the matching development control. They do not resolve prior full-layer/native ATOM numerical equivalence or validate later use of all Mix coefficients in MoE/residual stages. No GSM8K was run here.

## AITER findings and parallel experiments

Studied the installed gfx1250 Gluon fusions/mhc.py, HIP wrapper/kernel, and preserved their exact source, MIT license and commit in upstream-aiter-mhc/. Relevant ideas: early TDM with scheduler pinning, separate Sinkhorn work from apply/RMSNorm, producer partial sums, prepacked high/low weights, and reciprocal Sinkhorn iterations. AITER's fused-norm path squares the FP32 mixed residual, whereas this control squares its BF16-rounded form; that arithmetic difference was not silently adopted.

- GPU0: early QA/KV L2/TDM experiments, full-prefix integration, final acceptance and final performance.
- GPU1: merged/parallel tail, reciprocal Sinkhorn, exact-order diagnosis, native AITER diagnostic, graph stress. Native AITER timing is diagnostic only because norm rounding differs.
- GPU2: split-K/N-tile sweep, producer RMS partials, high/low weight packing, global/local L2 hints. Smaller producer grids did not help the fixed 256-CTA prefix. Packing/hints provided small variable benefits; not selected. Producer RMS reassociation is not accepted.
- GPU3: early residual/norm TDM, precise reshape reduction, independent prefix validation, rotating-weight tests.

## Reproduce and resume

Run `bash run.sh` in this directory inside the existing gfx1250 container with GPU0 idle. Fixtures are included; no checkpoint access is needed. setup.State.run(R, "opt_both") selects the measured short-context variant; `opt` selects the measured longer-context variant. `fused` deliberately remains the original fused control and `unfused` the 17-launch control. Copy the folder before rerunning because scripts overwrite measurements. Sources, fixtures, upstream licenses, validation, raw first/second-round samples, traces, assembly/LLVM, experiment scripts, and hashes are bundled. Experiments/ contains rejected/exploratory snapshots clearly separated from selected source. No model-serving script, image, or git branch was changed.
