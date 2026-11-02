# GPU3 final fixed-address microbenchmarks

Primary method: `attention/graphbatch.py:FlatGraphBatch`. Capture one callable invocation, retain graph and buffers, topologically sort its kernel nodes, clone unchanged kernel parameters128times into a flat serial graph. No per-repeat allocation and no child-graph nodes. Empty nodes may be omitted; any other nonkernel node is rejected.128repeats,5warmups,40alternating samples, second-round median GPUevent time. Old `attention/`, `prefix-gpu3/` use distinct per-repeat native buffers and are diagnostics. `*-samebuffers/` used childgraphs and include child scheduling overhead; do not use as primary.

| Scope | Actual ATOM kernels us | Candidate us |
|---|---:|---:|
| mHC through attention, same shuffled residual input | 54.7057 | 31.2727 |
| QKV through attention, same BF16input and one cache write | 43.9061 | 20.2342 |
| Attention only vs candidate attention plus cache write | 11.1278 | 8.5286 |

Prefix native10kernels includes native mHC residual-unshuffle helper; candidate2kernels includes inputadapter+optimized_prefix. Core-only candidate starts plain residual and is supplemental. Native QKV+attention7kernels vs candidate1 are exactly matching stage boundaries. Standalone attention row differs in cachewrite boundary and should remain diagnostic.

Validation: prefix native-vs-candidate relativeRMS0.0022118, QKV+attention0.00202988; prior0.001gate fails. Timing is NOT proof of an accepted ATOMreplacement. GPU health clean. Both methods use BF16KV; trueSWA128, native indexcapacity16384 produces128splits, candidate4splits.

Graph execution validated: prefix unifiedprofiler captures1664kernels=128*(10+2+1); QKV+attention captures1024=128*(7+1). Individual first profiler sessions sometimes drop all nativeevents, so use all-batch-trace.json. Explicit nodecounts and NaNpoisonoutput replay show finite bitwise-identical to immediatecall outputs. Do not claim coldcache or native serving speedups; inputs are synthetic/checkpointfixtures and warmed static buffers.
