# MiniMax-M3 PA benchmark

This benchmark isolates the MiniMax-M3 TP4 attention shapes optimized by
[PR 7079](https://github.com/ROCm/aiter/pull/7079). Each GPU has 16 query heads,
one KV head and head dimension 128. Queries and outputs are BF16; KV is either
BF16 or E4M3 FP8 with separate FP32 per-token K/V scales. It targets gfx1250 and
requires the checkout's PyTorch, AITER and FlyDSL dependencies.

The benchmark calls `aiter.ops.flydsl.pa_decode.pa_decode` directly. It measures
the complete attention call, including any partition reduction. Planning, KV
quantization, sparse block-table construction, compilation, allocations and
reference checks happen before timing. These are synthetic attention inputs;
this does not measure model inference, index scoring or top-k selection.

## Cases

| Suite | Requests / PA rows | Query length | Context tokens | Physical page |
| --- | --- | --- | --- | --- |
| Dense decode | 1, 4, 8, 32 requests | 1 | 8192 | 128 |
| Dense decode | 4 requests | 1 | 32768 | 128 |
| Dense decode | 3 requests | 1 | 225280 (220 Ki tokens) | 128 |
| Dense verify | 3 requests | 4 | 8192, 32768, 225280 | 128 |
| Dense verify | 8 requests | 4 | 32768 | 128 |
| Sparse decode | 4, 64, 128, 256 rows | 1 | 2048 | 16 |
| Sparse verify | 3 requests × 4 tokens = 12 rows | 1 per PA row | 2048 | 16 |
| Sparse prefill | 1024, 4096, 8192 rows | 1 | `min(row + 1, 2048)` | 16 |
| Optional cluster boundary | 239, 240, 241 rows | 1 | 2048 | 16 |

Dense verify uses four causally masked queries sharing one request's block
table. Sparse verify uses one independent block-table row per query token and
`query_length=1`. Every sparse table selects 16 logical 128-token blocks from a
shared 4096-page KV pool. Each selected block retains its eight consecutive
physical page16 entries. Prefill rows have increasing visible lengths.

There are 18 default shapes, run with both KV dtypes (36 measurements per
profile). The optional cluster boundary cases exercise PR 7079's resident
cluster limit on a 256-CU gfx1250: 240 rows can merge within clusters, while
241 rows use the reducer. The JSON records the observed output mode rather
than assuming a particular kernel dispatch.

## Profiles

`--profile atom` is the default and reproduces the ATOM calling conventions
described in PR 7079: dense block tables are padded to 256K tokens and decode
receives `max_context_length=None`; sparse decode and verify use
`get_recommended_splits(rows, 1)` with `max_context_length=2048`. Sparse prefill
uses a one-partition cap. On the 256-CU target, the sparse caps are 8 at 4–64
rows, 4 at 128 rows, 3 at 239–241 rows, and 2 at 256 rows. The optional cluster
boundary suite pins a two-partition cap in both profiles to isolate the
240-cluster residency limit. This profile models the PA call; ATOM's routing
and plan caching are outside its scope.

`--profile pr` uses exact-sized dense tables, a tight live context bound and
the planner's default partition cap, matching the PR's main performance table.
Both profiles use the default planner workgroup budget (`2 × CU count`), unless
`--workgroup-budget` overrides it. `--profile both` runs both configurations.
For 8192-row prefill, the harness raises the planner's `max_batch_size` to 8192
when that parameter is available; its default is 4096 on the latest PR head.

## Run

From the repository root:

```bash
# List the matrix without importing GPU libraries.
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py --list

# Full ATOM profile, BF16 and FP8, with correctness checks.
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --output /tmp/m3-atom.json

# Both profiles, or one selected shape/suite.
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --profile both --output /tmp/m3-both.json
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --case 'dense-verify-b3-q4-*' --dtype fp8 --output /tmp/m3-verify.json
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --suite cluster-boundary --output /tmp/m3-clusters.json
```

Select a visible PyTorch device with `--device N`. Run comparisons on the same
idle GPU, serially, with the same environment and benchmark settings. Small
differences should be confirmed with alternating baseline/candidate runs.

The default timer captures ten complete decode calls per CUDA graph. Each
sample times five replays with GPU events and divides by 50 calls. It takes the
minimum of five samples per round and reports the median of three round minima,
following the PR's measurement procedure. JSON retains every sample. Use
`--graph-calls`, `--replays`, `--samples`, `--rounds` and `--warmup` to adjust it.

Correctness checks compare eager and replayed outputs against independent
FP32 attention over dequantized KV. BF16 uses `rtol=atol=5e-3`; FP8 requires
relative L2 error at most 5% for **every query row**. FP8 Q/P arithmetic has
about 2–3% relative error on this input distribution even on main; short
prefill rows can exceed a fixed 0.005 absolute tolerance. The JSON records
global and maximum row errors, maximum absolute error and the check criterion.
A further replay negates Q and poisons output with NaNs before checking again.
This verifies that the
captured graph reads current inputs and writes all rows. `--no-check` skips
these checks and marks results as unchecked.

## Compare checkouts

Use the same benchmark file for both checkouts. `--source` selects which
checkout's AITER implementation is imported, and verifies its actual path.
The source checkout must provide the planned PA API used by PR 7079.

```bash
git worktree add --detach /tmp/aiter-pa-base <baseline-commit>
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --source /tmp/aiter-pa-base --profile both --label baseline \
  --output /tmp/m3-base.json
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --profile both --label pr7079 --compare-to /tmp/m3-base.json \
  --output /tmp/m3-pr7079.json
```

Comparison joins by case, profile and KV dtype. It requires matching GPU,
PyTorch/FlyDSL versions, seed, harness, timing settings, environment and input
geometry. The ratio is **baseline latency / candidate latency**, so values
above one indicate a speedup. Plan capacity and output mode may differ between
implementations and are recorded for review.

Each run writes JSON plus a CSV with the same basename, saving completed rows
after every case. Results include source commit and dirty status, GPU identity,
software versions, timing samples, actual plan cap/capacity, scratch size,
table width, output mode and reference error. Keep result files outside the
source checkout when comparing clean revisions.

## Validation

Validated on 2026-11-02 against PR 7079 head `15214dad7`, with a 256-CU gfx1250,
PyTorch 2.11.0+rocm10.1 and FlyDSL 0.4.1:

- All 84 measurements (both profiles, both KV dtypes, including cluster
  boundaries) passed eager and CUDA-graph checks with changed queries.
- At a two-partition cap, 239 and 240 rows used cluster merges; 241 rows used
  the reducer, for both KV dtypes.
- The maximum FP8 row relative L2 error was 0.03445, below the 0.05 limit.
  FP8 1024-row sparse-prefill output matched main `12ef6d652` bit for bit.
- Sixteen representative measurements also passed comparison against main
  `12ef6d652`, using the same harness and timing settings in separate processes.
- CPU matrix/lazy-import checks, Black, Ruff and `git diff --check` passed.
