# Flash C1 regional microbenchmark

Scope: checkpoint-backed layer 0, single GPU, concurrency 1, 128-row controlled KV window, grid 256. These are split-control kernels derived from the prototype, **not the native AITER serving baseline**. Native AITER comparisons remain necessary before selecting production regions. All routed experts and shared expert are included. HERD is not used for this hash-routed C1 layer.

Timing: second microbenchmark round only; median of 30 graph replays per measurement. Column 3 sums actual GPU kernel durations within each complete unfused group, then takes the median across replays. Column 4 measures one regional kernel. Input restoration copies, host overhead, and gaps between launches are excluded. Both arms restore the same pre-region inputs each replay. First round is retained but ignored. These are not serving clients or end-to-end latencies.

| Unfused operations (prototype helper names) | Fused region | Unfused summed GPU time (µs) | Fused GPU time (µs) |
|---|---|---:|---:|
| mhc_mix → sinkhorn → mhc_pre → norm → quant | Attention input mHC + norm + quant | 14.76 | 16.42 |
| matvec(QA, KV) → norm(Q, KV) → quant(Q) → matvec(QB) → norm(Q) → rope(Q, KV) | Q/KV projections + norm + RoPE | 31.84 | 33.59 |
| KV cache store → attention | Cache update + attention | 14.58 | 14.56 |
| rope(inverse) → quant → matvec(OA) → quant → matvec(OB) | Attention output projections | 52.03 | 53.66 |
| mhc_post → mhc_mix → sinkhorn → mhc_pre → norm | Attention residual + FFN input | 19.16 | 20.23 |
| quant(FP8, MXFP8) + router matvec → route + shared gate/up → activation → quant → shared down + routed gate/up → activation → quant → routed down | Routing + shared expert + routed MoE | 158.91 | 164.99 |
| expert reduction + shared add → mhc_post | Expert sum + final residual | 3.56 | 3.96 |

Each unfused launch is named `split_stage` with compile-time STAGE specialization; each fused launch is `region_kernel` with REGION specialization. The operation names above are actual helper names, not fabricated native AITER kernel names. mapping.json contains the complete source mapping for all 33 stages.

Findings:
- MoE dominates: its shared-down plus routed gate/up stage alone is 77.24 µs in the individually timed stage diagnostic; routed down is 47.71 µs. The prototype uses packed vector matvecs, not native WMMA GEMMs.
- Attention output projection is next: individually timed OA is 30.04 µs and OB is 23.15 µs. Prioritize the GEMM implementations before adding more global fusion.
- Fusion provides no material win for any measured group. Internal global synchronization and reduced scheduling freedom remain costs; this experiment does not isolate their individual contributions.
- The independent-stage timing sum differs from back-to-back group timing due to execution/cache context. Use the grouped numbers above. Raw JSON retains both.

Optimization trial: distribute routed expert output tiles across CTAs rather than walking six experts sequentially. Every recorded intermediate still exactly matches the original split control. MoE changed from 164.99 to 164.20 µs (0.47%); its contemporaneous grouped split control is 159.07 µs. This sub-1% change is not accepted as a meaningful performance improvement. Sources and traces are in expert-tiles/.

Validation: all intermediates, final output, and cache compare exactly to the original split control; semaphore health is zero. regions-trace.json verifies seven regional kernel launches. Existing independent-reference validation applies to original arithmetic, but this experiment adds no native ATOM equivalence or GSM8K evidence.

Next optimization work: establish matching native AITER region measurements with actual layouts/quantization; replace the routed MoE and attention output vector matvecs with optimized gfx1250 WMMA stages; then remeasure fusion. Keep separate regions wherever synchronization costs exceed launch savings.

Reproduce: start the existing idle vigokhal_dsv4_tp4 container after checking GPU availability, then run:
```sh
docker exec -e HIP_VISIBLE_DEVICES=0 -e TRITON_CACHE_DIR=/work/full-layer-c1/triton-cache vigokhal_dsv4_tp4 bash -lc "cd /work/full-layer-c1/regions; python bench.py --grid 256"
```
The benchmark overwrites fixed output filenames; copy the directory before rerunning. Sources, generators, timings, raw profiler traces, and emitted ISA are preserved. Timed CUDA graph events were unavailable in this ROCm build (external events disallowed; internal events returned invalid resource handles), so the final harness uses profiler GPU durations. The container is stopped after this experiment.
