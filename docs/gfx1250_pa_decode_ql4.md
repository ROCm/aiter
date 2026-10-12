# gfx1250 FP8 D256 QL4 optimization, 2026-10-12

FP8 D256/G8/QL4 with transposed page128 caches and per-token scales now keeps
both query tiles fused while using the packed 128-token path. B64/64K at budget
512 improves from 2178.59 µs to 1085.78 µs, with an independent candidate repeat
at 1087.17 µs and baseline repeat at 2180.62 µs. The default budget stays 512.
Final post-validation runs measured 1113.13 and 1120.87 µs, maintaining about
49% lower latency than the baseline.

The baseline is `db4b45de8`, including direct output and wide native BF16 query
splitting. All decision timings run
`op_tests/op_benchmarks/flydsl/bench_pa_decode.py` directly in separate processes.
Each run uses seed 0, U(-0.5,0.5), BF16 queries, FP8 E4M3 per-token K/V scales,
separate shuffled physical pages, seven timing rounds, 32 decode+reduce calls
per graph, and twelve replays per round. The benchmark's existing FP32 reference
checks and tolerances are unchanged. GPU tests do not overlap timing.

## Sequential experiments

| B64/64K D256/G8/QL4 experiment | Budget 512 µs | Budget 2048 µs | Decision |
|---|---:|---:|---|
| Baseline padded 64-token path | 2178.59 | 1919.05 | Control |
| Packed 128-token path, fused queries | 1085.78; repeat 1087.17 | 1098.23; repeat 1098.95 | Retain |
| Keep both query operands resident | 1087.36 | 1102.76 | Reject: flat/slower |
| Two query splits using packed pages | 1310.35 | 1352.33 | Reject: slower |

Budget 1024 measured 2276.80 µs on the baseline and 1098.07 µs on the packed
candidate. The retained default-budget candidate also exceeds the baseline's
best measured budget, 2048, by about 1.77 times.

The old packed-layout condition required one 16-row query tile. QL4/G8 has
32 rows, so it fell back to padded 64-token staging. Extending the condition
to the measured D256/G8/QL4/page128 full-attention specialization activates
128-token packed TDM copies, K128 QK/PV WMMA, asynchronous scale copies, and the
existing rescaled running output. K/V stay shared between the two query tiles.
BF16, scalar KV scales, plain V, windows, and other multi-query specializations
keep their existing selection.

## Shorter and variable contexts

The following official benchmark sweep uses minimum context 4096 and a uniform
distribution up to each maximum. The 4K rows have fixed context; the larger
rows contain varied, unrounded context lengths.

| Batch | Maximum context | Baseline µs | Packed µs | Speedup |
|---|---:|---:|---:|---:|
| 16 | 4096 | 104.34 | 98.67 | 1.057× |
| 16 | 16384 | 114.73 | 74.00 | 1.550× |
| 16 | 65536 | 413.67 | 235.13 | 1.759× |
| 64 | 4096 | 191.04 | 137.60 | 1.388× |
| 64 | 16384 | 566.51 | 290.51 | 1.950× |
| 64 | 65536 | 2050.50 | 872.77 | 2.349× |

Added tests cover D256 QL4 with both query dtypes, both plan caps (direct and
partitioned), changing contexts after graph capture, poisoned scratch, empty
and short causal rows, and NaN K/V/scales beyond partial pages. The generic
QL4 case also checks sinks and alternative V/scale modes.

The complete FP8/BF16 suite passes 161 tests in 250.94 seconds. Black, Ruff,
and whitespace checks pass. New NaN-tail coverage is restricted to D256; an
exploratory D128 QL4 test exposed NaNs in the unchanged padded path and was
excluded from this specialization's coverage.

## Final throughput and physical reads

| B64/64K D256/G8/QL4, budget 512 | Baseline | Retained |
|---|---:|---:|
| Median decode+reduce latency µs | 2177.35 | 1113.13 |
| Useful KV bandwidth TB/s | 7.890 | 15.434 |
| Useful % of reported 23.347 TB/s peak | 33.80% | 66.11% |
| Physical read bandwidth TB/s | 7.688 | 15.037 |
| Read % of reported peak | 32.93% | 64.41% |
| Mean memory fetches per call GB | 16.738 | 16.738 |

Physical reads use `rocprofv3 --pmc FETCH_SIZE`, converting KiB to bytes.
The final ten warmed decode and reducer dispatches are averaged and summed,
then divided by the independently measured unprofiled latency. Profiling
serializes dispatches, so these rates are estimates; writes are excluded.
Traffic remains nearly unchanged, and bandwidth improves through lower runtime.
The retained decode reports 176 VGPRs, zero scratch, and 144,896 bytes of LDS,
versus 136 VGPRs, zero scratch, and 87,552 bytes of LDS on the baseline.

The adjacent QL1 control measured 1084.68 µs. QL4 takes 1.026 times that latency
for four query positions, giving 229,982 query tokens/s across the batch and
3.898 times the QL1 query throughput in these sequential measurements.

## Reproduce

From the repository root in the ROCm development environment:

```bash
python3 -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 64 --min-context-len 65536 --max-context-lens 65536 \
  --head-dim 256 --query-heads 64 --kv-heads 8 --query-length 4 \
  --kv-dtypes fp8 --workgroup-budgets 512 1024 2048 \
  --rounds 7 --replays 12 --graph-iterations 32 \
  --output benchmark-results/ql4-stream.json

python3 -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 16 64 --min-context-len 4096 \
  --max-context-lens 4096 16384 65536 \
  --head-dim 256 --query-heads 64 --kv-heads 8 --query-length 4 \
  --kv-dtypes fp8 --workgroup-budgets 512 \
  --rounds 7 --replays 12 --graph-iterations 32 \
  --output benchmark-results/ql4-context-sweep.json

python3 -m pytest op_tests/test_flydsl_pa_decode.py \
  op_tests/test_flydsl_pa_decode_bf16.py -q -x
```

The [measurement summary](validation/gfx1250_pa_decode_fp8/ql4-20261012-summary.json)
contains all timing samples and the final physical-read counter samples.
Full official JSON/CSV reports and source snapshots are saved under
`/home/sixifang/gfx1250-dev/pa-gfx1250-validation/ql4-20261012/`.
