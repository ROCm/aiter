# Sequential gfx1250 PA decode experiments, 2026-10-12

Two changes were retained: direct output for eligible one-partition plans, and
four query splits for transposed page 128 native BF16 D1024/G16/MTP4 full
attention. Separate K/V waits and shorter V operand lifetimes did not show a
repeatable streaming improvement. The default partition budget remains unchanged.

All decision timings use `op_tests/op_benchmarks/flydsl/bench_pa_decode.py`
directly. Each source snapshot runs in a separate process, with seed 0,
U(-0.5,0.5) inputs, separate shuffled physical pages, seven timing rounds,
32 decode calls per graph, twelve replays per round, and the benchmark's
existing FP32 reference checks and tolerances. Baseline repeats bracket
important comparisons. These are sequential measurements, not same-process
paired timings. Physical read counters were collected separately on the final
kernel after the optimization decisions; see the bandwidth measurements below.

Hardware/software: gfx1250, 256 CUs, PyTorch
2.11.0+rocm10.1.0a20260811-bkc.20260820, HIP 7.16.26315,
FlyDSL 0.3.3.dev886. GPU tests and ISA compilation did not overlap timing.

## Experiments

| Experiment/workload | Baseline µs | Candidate µs | Decision |
|---|---:|---:|---|
| Separate K/V waits, FP8 D256 B64/64K, budget 512 | 1092.14; repeat 1099.20 | 1091.04 | Reject: within drift |
| Same, budget 2048 | 1097.35; repeat 1104.80 | 1102.14 | Reject: within drift |
| Preload one V fragment, same streaming shape, budget 512/2048 | 1092.14 / 1097.35 | 1095.36 / 1102.38 | Reject |
| Preload two V fragments, same shape, budget 512/2048 | 1092.14 / 1097.35 | 1101.16 / 1112.97 | Reject |
| Direct output, FP8 D128 MQA B4/4K, cap 1 | 56.78; repeat 57.00 | 53.09; repeats 52.97 / 53.04 | Retain: 6.5–7.1% reduction |
| Direct output, FP8 D128 G8 MTP4/window 257, boundary contexts, cap 1 | 99.45; repeat 103.57 | 51.94; repeat 53.02 | Retain: about 48–49% reduction |
| Two query splits, FP8 D256 G8 MTP4 B16/4K, budget 512 | 82.98; repeat 74.92 | 109.17 | Reject automatic selection |
| Four query splits, same FP8 shape, budget 512 | 82.98; repeat 74.92 | 66.74 | Leave fused by default: no independent candidate repeat |
| Two query splits, BF16 D1024 G16 MTP4 B4/1K, budget 512 | 299.77; repeat 292.31 | 73.79 | Four splits faster |
| Four query splits, same BF16 shape, budget 512 | 299.77; repeat 292.31 | 58.34; retained repeats 58.03 / 58.24 | Retain: approximately 5× speedup |

The boundary contexts are `[0,1,127,128,129,255,256,257,1027]`; all nine
sequences were reference-checked. Other benchmarks check four sequences.
The boundary BF16 direct-output control also passes at 52.68 µs.

The direct-output benchmark uses `--max-partitions 1` for both implementations.
For B4/1K wide BF16, combining direct output with splitting measured 204.60 µs,
versus 58.03 µs with a larger partitioned plan. A one-partition plan is therefore
not recommended for that workload. This comparison changes the plan as well as
the epilogue, and does not establish a direct-output regression at the same plan.

## Budget and historical baseline controls

B16/max 16K FP8 D256 budget sweeps include 128/256/512/1024/2048/4096.
Budget 512 is best for both sampled distributions:

| Distribution | Budget 512 decode+reduce µs | Budget 2048 µs | Budget 512 including plan µs | Budget 2048 including plan µs |
|---|---:|---:|---:|---:|
| log-uniform | 82.06 | 119.25 | 83.46 | 123.84 |
| bimodal | 62.00 | 75.12 | 64.84 | 77.77 |

No tuning entries were installed. Earlier exploratory custom-driver results
are retained locally, but do not determine these decisions.

Clean `fae4e8c61` measured 1329.20 µs at budget 512 and 1102.52 µs at 2048 for
D256 B64/64K. HEAD measured 1092.14–1099.20 µs at 512. This control does not
reproduce the older reports' apparent 71%→55% bandwidth regression. It measures
latency only and does not resolve the earlier harness differences.

## Final bandwidth measurements

The final streaming kernel was remeasured with the official benchmark. Useful
KV bandwidth counts K and V once per context token. Physical read estimates use
`rocprofv3 --pmc FETCH_SIZE`, whose unit is KiB, including memory/cache effects.
The last ten dispatches of each PA kernel are averaged and summed, then divided
by the separately measured unprofiled median latency. Profiler timings are
serialized and are not used to calculate rates. Percentages use the reported
23.347 TB/s peak as the reference; these are read rates, not combined read/write
bus utilization.

| Final workload | Median µs | Useful KV TB/s | Useful % peak | Physical read TB/s | Read % peak |
|---|---:|---:|---:|---:|---:|
| FP8 D256 H64/KV8 QL1 B64/64K, budget 512 | 1110.19 | 15.475 | 66.28% | 15.066 | 64.53% |
| FP8 D256 H64/KV8 QL4 B64/64K, fused queries, budget 512 | 2177.35 | 7.890 | 33.80% | 7.688 | 32.93% |
| BF16 D1024 H16/KV1 QL4 B4/1K, four splits, budget 512 | 58.24 | 0.288 | 1.23% | 0.338 | 1.45% |
| FP8 D128 H16/KV1 QL1 B4/4K, direct output, cap 1 | 53.04 | 0.079 | 0.34% | 0.080 | 0.34% |

The streaming call fetches 16.726 GB on average, including its reducer. The two
small workloads do not approach streaming bandwidth: direct MQA has only four
active workgroups, while wide BF16 is dominated by per-workgroup execution and
resource constraints. Four query splits issue repeated logical KV reads, but
cache reuse keeps measured memory fetches well below four complete KV copies.
All timing and profiling runs pass the benchmark's existing reference checks.
Counter collection changes dispatch serialization, so physical rates are
estimates using the independently measured unprofiled timing.

For streaming FP8 QL4, queries remain fused. Its median latency is 1.961 times
QL1, with 544.34 µs amortized per query position and 117,574 query tokens/s across
the batch. Query throughput is 2.040 times QL1 in these sequential measurements.
Mean memory fetches are 16.738 GB per call, nearly unchanged from QL1, showing
that QL4 retains KV reuse while doing additional attention work. The decode
counter record reports 136 VGPRs, zero scratch, and 87,552 bytes of LDS.

The QL1 streaming counter capture logged queue-sync warnings during profiler
finalization, after the benchmark completed. Each capture contains the
expected 17 dispatches per PA kernel (initial launch, five warmups, one graph
replay, ten measured replays); the final ten have valid counters. The direct
MQA capture contains only the decode kernel, confirming that its reducer is
skipped.

See the [bandwidth evidence](validation/gfx1250_pa_decode_fp8/sequential-20261012-bandwidth.json)
and [streaming QL4 evidence](validation/gfx1250_pa_decode_fp8/sequential-20261012-stream-ql4.json)
for samples, workload settings, counter definitions, and source paths.
`reproduce-final-bandwidth.sh` in the local evidence directory records commands
for independent timing and counter collection using the official benchmark.

## Implementation and validation

Direct output is selected on gfx1250 when the plan cap is one, sinks are absent,
and query/output strides match. The launch maps sequences through `reduce_info`
to packed task slots. Empty sequences write zeros, including after graph refresh.
The kernel writes final values through the existing output argument and skips
the reducer; provided scratch buffers retain their existing validation.
Other architectures and sink workloads retain reduction.

Query splitting preserves the full MTP length for query addressing and causal
masking, computes a per-CTA query offset, and writes each split into the original
unsplit scratch layout. Automatic selection is restricted to native BF16,
D1024/G16/MTP4, transposed V, page 128, full attention. Explicit compiler overrides
accept 1/2/4 splits dividing the query length.

Fresh single-specialization ISA dumps with runtime caching disabled report:

| BF16 D1024/G16/MTP4 resource | Fused | Four splits |
|---|---:|---:|
| VGPR count | 1024 | 698 |
| SGPR count | 106 | 106 |
| Private bytes/work-item | 2208 | 0 |
| Static LDS bytes/workgroup | 315968 | 315968 |

The final FP8/BF16 suite passes 149 tests in 235.99 seconds. Added coverage
includes direct packed NaN tails, direct BF16 graph refresh, and split BF16 query graphs with multiple
KV heads, changing empty/nonempty sequences, poisoned scratch, all-empty
batches, and matching odd query/output strides. Black, Ruff and whitespace
checks pass. A test-reference dtype mistake in an earlier run was corrected;
the final complete suite passes with unchanged tolerances.

## Reproduce

From the ROCm development environment, run the repository benchmark:

```bash
python3 -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 4 --min-context-len 1024 --max-context-lens 1024 \
  --head-dim 1024 --query-length 4 --query-heads 16 --kv-heads 1 \
  --kv-dtypes bf16 --page-size 128 --v-layout transposed \
  --workgroup-budgets 128 512 --rounds 7 --replays 12 --graph-iterations 32 \
  --output benchmark-results/wide-bf16.json

python3 -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 4 --min-context-len 4096 --max-context-lens 4096 \
  --head-dim 128 --query-heads 16 --kv-heads 1 --max-partitions 1 \
  --workgroup-budgets 128 --rounds 7 --replays 12 --graph-iterations 32 \
  --output benchmark-results/direct-mqa.json

python3 -m pytest op_tests/test_flydsl_pa_decode.py \
  op_tests/test_flydsl_pa_decode_bf16.py -q -x
```

Repository evidence: [timing summary](validation/gfx1250_pa_decode_fp8/sequential-20261012-summary.json)
and [resource summary](validation/gfx1250_pa_decode_fp8/sequential-20261012-resources.json).
Full source snapshots, commands, JSON/CSV samples, logs, and ISA dumps remain at
`/home/sixifang/gfx1250-dev/pa-gfx1250-validation/sequential-20261012/`.
