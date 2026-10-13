# PA decode merge validation, 2026-10-13

Merged main `cc709933512a056e04c596877086f92be9067df8` into
`enable-gfx1250-pa-decode`, previously `6dcdcbeee8d3125ffc4074f15111cd3207505da8`.
`aiter/ops/flydsl/kernels/pa_decode_plan.py` is identical to main: the Triton
planner is replaced by main's FlyDSL block-wide scan and packed-task writer.
The reusable plan still contains 256-token tile ranges, packed work records,
per-sequence reduction records, and cleared padding. Refresh uses the current
stream and can be captured in a graph.

The gfx942/gfx950 decode refactor and native BF16 implementation come from main.
The gfx1250 compute backend is unchanged. Architecture dispatch preserves its
wave32 reducer, eligible one-partition direct output, wide BF16 query splitting,
and packed FP8 D256/G8/QL4 path. The wrapper infers compute from the KV cache when
omitted. Main's native `prepare_pa_decode_plan` API and tuning CLI support
gfx1250 FP8/BF16 and retain the default workgroup budget of twice the CU count.

Validation used the pinned FlyDSL **0.3.4.1** wheel, installed in a separate
target directory, with PyTorch 2.11.0+rocm10.1, HIP 7.16.26315, and a gfx1250 GPU
with 256 CUs. GPU tests and benchmarks ran sequentially.

- `op_tests/test_flydsl_pa_decode.py` and `op_tests/test_flydsl_pa_decode_bf16.py`:
  **183 passed, 4 skipped in 281.98 seconds**. This includes six planner cases
  with up to 4,096 sequences, empty and near-int32-limit lengths, windows,
  graph refresh, FP8 QL4 page tails, sinks, direct output, wide addresses, and
  BF16 split-query graphs. Skips are gfx9 BF16 D192 cases and rejection cases
  for FP16 queries/plain BF16 V, which gfx1250 accepts.
- Native plan preparation checked FP8 and BF16 default-cache misses, candidate
  search, saved-cache reload, and captured planner/decode replay against the
  FP32 attention reference. The CLI also passed BF16 tuning with FP16 queries
  and plain V caches. Tuning files were saved outside the repository.
- FP8 and BF16 MTP4 tile, compact reducer, and FlyDSL planner compilation passed
  for gfx942 and gfx950 with `COMPILE_ONLY=1`; these targets were not tested
  numerically on hardware.
- Black, Ruff, and the merge-resolution whitespace check against main passed.

All timings use `op_tests/op_benchmarks/flydsl/bench_pa_decode.py`, BF16 queries,
transposed page128 caches, budget512, seven rounds, twelve graph replays per
round, and 32 calls per graph. The FP8 workloads use per-token scales.

| Workload | Timed operations | Median µs | Useful KV TB/s |
|---|---|---:|---:|
| FP8 B64/64K, D256, H64/KV8, QL4 | Decode + reduction | 1085.70 | 15.82 |
| FP8 B64/64K, D256, H64/KV8, QL4 | Plan + decode + reduction | 1089.87 | 15.76 |
| FP8 B4/4K, D128, H16/KV1, QL1, cap1 | Plan + direct decode | 56.90 | 0.074 |
| BF16 B4/1K, D1024, H16/KV1, QL4 | Plan + split decode + reduction | 64.02 | 0.262 |

Every benchmark passed its accuracy check. Useful bandwidth counts each logical
K/V byte once and does not measure physical memory traffic. Replays reuse the
same inputs. QL4 is within the previous 1.086–1.121 ms range; the previous runs
used FlyDSL 0.3.3.dev886, so this comparison does not isolate planner or compiler
effects. The two planner-inclusive controls also use a different timed operation
set from the earlier decode-only measurements.

For the requested workload:

```bash
python3 -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 64 --min-context-len 65536 --max-context-lens 65536 \
  --head-dim 256 --query-heads 64 --kv-heads 8 --query-length 4 \
  --kv-dtypes fp8 --workgroup-budgets 512 \
  --rounds 7 --replays 12 --graph-iterations 32 --output ql4.json
# Add --include-plan to time captured FlyDSL plan refresh as well.
```

Raw benchmark metadata and samples are in
[the merge summary](validation/gfx1250_pa_decode_fp8/merge-main-20261013-summary.json).
Full local logs and smoke-check scripts are under
`/home/sixifang/gfx1250-dev/pa-gfx1250-validation/merge-main-20261013/`.
