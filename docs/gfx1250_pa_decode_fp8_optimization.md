# gfx1250 D256 FP8 PA decode optimization

On the current fixed-length benchmark, the optimized kernel reaches **54.91% of
peak physical read bandwidth** on B64/64K, versus 50.84% before. Across the four
large fixed workloads, latency falls **6.9–8.0%**. **The requested 60% target has
not been reached.**

This comparison starts from the dirty workspace kernel captured locally in
`benchmark-results/fp8-bandwidth60/baseline.py`, on branch
`enable-gfx1250-pa-decode`, HEAD
`fae4e8c61ebc7e41d205c6764240e13027cc8160`. It includes the earlier D128
optimization. The baseline is not a clean checkout of HEAD.

## Kernel change

The measured specialization is per-token FP8, D256, transposed V, page128, full
attention, and one 16-row query tile. It now alternates two packed K/V LDS
buffers, reuses the page lookup for K/V and scales, and copies full-page FP32
scales directly from global memory to LDS with asynchronous 128-bit loads.
At consumption it waits for both TDM and asynchronous copies before the CTA
barrier. Partial pages retain exact TDM extents and masked scale loads.
Q/P quantization and online softmax arithmetic are unchanged.

Each context partition and KV head launches one 128-thread workgroup: four
wave32 waves. Partitions cover ranges of 256-token plan tiles, processed in
128-token compute tiles for this specialization. With B64/64K and Hkv8,
budget512 gives one 65536-token partition per sequence; budget2048 gives four
16384-token partitions per sequence. The eight KV heads launch separate
workgroups for every partition.

## Fixed-length results

BF16 Q/O, FP8 E4M3FN K/V, FP32 per-token scales, Hq64/Hkv8, D256, QL1,
page128, transposed V, window0, seed0. Q/K/V are initialized from U(-0.5,0.5);
each sequence owns unique physical pages, shuffled across the cache pool.
Peak bandwidth is 23.347 TB/s, with the memory clock at 1900 MHz.

| Batch | Context | Budget before → after | Latency before → after (µs) | Latency reduction | Physical TB/s before → after | Peak usage before → after |
|---:|---:|:---:|:---:|---:|:---:|:---:|
| 64 | 65536 | 512 → 512 | 1409.224 → 1304.835 | 7.41% | 11.870 → 12.820 | 50.84% → 54.91% |
| 128 | 65536 | 512 → 2048 | 2853.034 → 2627.892 | 7.89% | 11.728 → 12.738 | 50.23% → 54.56% |
| 64 | 131072 | 2048 → 2048 | 2851.137 → 2624.174 | 7.96% | 11.739 → 12.754 | 50.28% → 54.63% |
| 128 | 131072 | 512 → 2048 | 5718.136 → 5322.011 | 6.93% | 11.701 → 12.574 | 50.12% → 53.86% |

Budgets 512/1024/2048 were compared on identical inputs in the same process,
with candidate order shuffled across two passes. The table selects each
implementation’s best median latency across those passes. Raw results also
contain every comparison at the same budget. B128 budgets 512/1024 produce
the same plan capacity. Using budget 2048 changes B128 capacity 128 → 256.

Timing covers decode plus reduce in a 32-call graph, seven rounds of twelve
replays per measurement. Initialization, planning, reference checks, JIT and
scratch allocation are excluded. Each timing result checks four sampled
sequences against FP32 attention with atol=rtol=0.005.

Physical bandwidth is measured with a separate single-pass rocprofv3
`FETCH_SIZE` capture: five warm dispatches followed by ten samples of each
decode/reduce kernel. FETCH_SIZE is converted from KiB to bytes; mean reads
from both kernels are divided by the independently measured graph latency.
All eight captures exited zero and contain ten samples per kernel.
rocprofv3 emitted queue-finalization timeout warnings after workload completion.

For B64/64K, physical reads are 16.727833 GB before and 16.727968 GB after
(a 0.00081% difference). The speedup does not come from increasing reads.
60% requires 14.0082 TB/s, or **1194.155 µs** for those reads. The measured
1304.835 µs needs a further **8.48% latency reduction**.

## Trace and resources

The matched ATT runs both use B64/64K and budget 512, the same 64-slot plan,
four waves per workgroup, CU 1, SIMD 0, and SE mask 0xf. Sampled stalls fall
11.20M → 9.06M cycles (about 19%). The baseline’s hottest instruction is
`s_wait_loadcnt 1` at 5.45M sampled stall cycles. After optimization, the
hottest instruction is `s_wait_tensorcnt 0` at 6.50M cycles, about 71.7% of
remaining stalls. This identifies exposed K/V TDM completion as the next
bottleneck. Single-CU trace totals are diagnostic samples, not whole-GPU timing.

The FlyDSL skill’s shipped hotspot analyzer misidentifies gfx1250 as gfx942.
Its CDNA occupancy estimates and classification of split waits as “other”
are not used. Instruction stall values and final ISA metadata are used.

| Resource | Before | After |
|---|---:|---:|
| VGPRs |223|240|
| SGPRs |102|93|
| Static LDS bytes/workgroup |73792|140352|
| VGPR/SGPR spills |0/0|0/0|
| Scratch bytes |0|0|

The shipped ISA-resource tool flags the register/LDS growth as a resource
regression. The second buffer deliberately increases LDS, and live overlap
increases registers. Measured latency improves, with no spills or scratch.
Final ISA contains `global_load_async_to_lds_b128` and both split wait counters.

## Other variants evaluated

The isolated pilots also tested larger/smaller token tiles, one/two/eight-wave
workgroups, independent wave partitions, deeper DMA rings, early V register
loads, direct global loads, mixed TDM/global loads, collective page descriptors,
scale-copy wave distribution, page lookahead, cache hints, LDS placement, P
swizzling, and LLVM scheduling hints. None produced a sustained improvement
beyond the retained candidate. The source candidates and their paired pilot
results remain in the local evidence directory; they were not integrated.

## Correctness and variable-length controls

`pytest op_tests/test_flydsl_pa_decode.py -q -x`: **88 passed in 58.05s**.
The existing packed-page boundary test now covers D128 and D256, BF16/FP16
queries and QL1/2, including NaN padding, empty contexts, varying scales,
poisoned scratch, graph replay, and refreshed plans. Tolerances were preserved.
All fixed and variable-length benchmark reference checks passed. The known
native-BF16/page16 fault workload was not run.

B16/max16K variable-length controls use identical random inputs:

| Shape/distribution | Budget | Before µs | After µs | Speedup |
|---|---:|---:|---:|---:|
| D128/log-uniform | 512 | 12.274 | 12.261 | 1.001× |
| D128/log-uniform | 2048 | 20.168 | 20.144 | 1.001× |
| D256/log-uniform | 512 | 23.740 | 20.155 | 1.178× |
| D256/log-uniform | 2048 | 35.081 | 36.397 | 0.964× |
| D256/bimodal | 512 | 55.497 | 50.819 | 1.092× |
| D256/bimodal | 2048 | 64.503 | 62.919 | 1.025× |

D256/log-uniform at budget 2048 regresses 3.8%. The specialization is therefore
not a universal improvement for every work plan. D128 controls are unchanged.

## Reproduce

Run inside the gfx1250 container from `/workspace/aiter`:

```bash
PYTHONPATH=/workspace/aiter \
AITER_JIT_DIR=/home/dev/.cache/aiter-pa-decode \
FLYDSL_RUNTIME_CACHE_DIR=/home/dev/.cache/flydsl-bandwidth60 \
/home/dev/.venvs/aiter-pa-decode/bin/python \
  -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 64 128 --min-context-len 65536 --max-context-lens 65536 \
  --distributions uniform --head-dim 256 --page-size 128 --kv-dtypes fp8 \
  --query-heads 64 --kv-heads 8 --query-length 1 --query-dtype bf16 \
  --v-layout transposed --kv-scale per-token --data-init uniform --seed 0 \
  --workgroup-budgets 512 1024 2048 --warmup 3 \
  --rounds 7 --replays 12 --graph-iterations 32 --check-sequences 4 \
  --output /workspace/benchmark-results/fp8-bandwidth60/reproduce64k.json
```

For 128K, set both context bounds to 131072. Matching minimum and maximum
produces fixed-length contexts. Run twice for independent repeats.

The exact paired comparison used the local `final_timings.py` driver.
`capture_final.py` captured physical counters; `capture_att_final.py` captured
matched ATT and fresh ISA; `summarize_final.py` validated sample counts. Those
drivers, source snapshots, and raw traces remain in
`/home/sixifang/gfx1250-dev/benchmark-results/fp8-bandwidth60/`; they are not
included in this repository. The command above reproduces current-kernel
timing, rather than the saved baseline comparison or physical counters.

Committed evidence: [bandwidth JSON](validation/gfx1250_pa_decode_fp8/bandwidth-summary.json),
[bandwidth CSV](validation/gfx1250_pa_decode_fp8/bandwidth-summary.csv),
[resource diff](validation/gfx1250_pa_decode_fp8/resource-diff.json), and
[evidence index](validation/gfx1250_pa_decode_fp8/README.md). The index also
links the paired timing samples, variable-length controls, counter capture
status, and sampled instruction hotspots.

The original commit’s 70% harness is unavailable; these measurements use the
current reproducible benchmark and do not establish why that report differs.
