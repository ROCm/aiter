# gfx1250 TP4 PullQ4 compressed-load prefetch

Experimental opt-in: `AITER_PULL_Q4_PREFETCH=1`. Unset/0 retains the existing
kernel; TP2 and TP8 retain it even when the variable is 1. The host dispatcher
reads the flag once, before graph capture.

The original reduction interleaves one remote payload/scale read with decoding
and accumulation for every atom. The candidate first issues the eight independent
compressed atom/leader-scale reads for one peer, then decodes them in order.
It does not change INT4 encoding, scale sharing, BF16/FP16 casts, fixed-rank
half-precision addition, output rounding, publication, epoch parity, or IPC
allocation. It uses more registers to overlap peer-read latency.

This is a peer-read scheduling experiment, not a new quantization scheme.

## Reproduction

Build `module_quick_all_reduce` from these sources for gfx1250. Headers must
actually be recompiled (do not compare against a prebuilt stale module).

```bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
pytest -q op_tests/multigpu_tests/test_quick_all_reduce_gfx1250.py \
  -k test_pull_q4_prefetch_tp4

AITER_PULL_Q4_PREFETCH=0 python \
  op_tests/multigpu_tests/bench_pull_q4_prefetch.py baseline 19301
AITER_PULL_Q4_PREFETCH=1 python \
  op_tests/multigpu_tests/bench_pull_q4_prefetch.py candidate 19302 \
  --compare baseline.json
```

The benchmark checks full output hashes between ranks and between old/new paths.
Timings include quantize, publish and reduce; reported values are the slowest
rank's median over five graph-replay timings. Two independent processes per side
were tested on C5 (four gfx1250 GPUs).

| Input size | Original median us | Prefetch median us | Speedup |
|---|---:|---:|---:|
| 8 MiB | 131.29 | 47.22 | 2.78x |
| 32 MiB | 163.51 | 113.22 | 1.44x |
| 38 MiB | 174.86 | 133.96 | 1.31x |
| 64 MiB | 315.18 | 193.03 | 1.63x |

All eight tested sizes, including unaligned tails, were bit-identical between
ranks and between old/new kernels. Existing eager/tail checks and 1000 alternating
graph replays with rank skew passed on both paths. This does not relax the
one-stream, same-collective-order contract.

A mixed MiniMax-M3 TP4 trace motivated this work. Profiler durations were much
higher than standalone GPU-event timings and are not an estimate of serving
savings. Serving A/B is recorded separately; operator speedup alone does not
establish TTFT/TPOT/throughput improvement.

Tests were invoked 2026-11-11 UTC on the control host. C5's clock was behind,
dating its logs 2026-11-07; no clock settings were changed.


## Recipe screening run

Same C5 node, TP4, original 2026-11-10 MiniMax-M3 recipe, EAGLE3 K3 with real
acceptance, FP8 KV, TG=1, MONO=0, max-seqs=16; no profiler during measurements.
Both sides use the same newly built binary, differing only in the opt-in flag.
The gather experiment is NOT included.

MLPerf Endpoints commit 4b385eb, original dataset SHA256
1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae.
Concurrency 8, 3-minute issue cap; exploratory, not a full official submission.

| Metric | Flag 0 | Flag 1 |
|---|---:|---:|
| System tok/s | 988.46 | 984.57 |
| TTFT p50, ms | 109.70 | 105.28 |
| TPOT p50, ms | 4.155 | 4.141 |
| TPOT mean, ms | 4.110 | 4.016 |
| Turns / failed | 373 / 0 | 349 / 0 |
| Mean output length | 494.1 | 527.1 |
| Window EAGLE acceptance | 18.82% | 17.52% |
| Inline accuracy (different sampled turns) | 53.91% | 49.84% |

This screening run does NOT establish an AgentX throughput improvement. Sampled
turns, output lengths and acceptance differ. The inline scores are not a matched
accuracy comparison and no GSM8K validation has been run for this patch.


## Actual HTTP serving replay (prefill-heavy, not full AgentX)

Three cold-prefix sets, 96 real first-user turns per set, concurrency 8, same
request hashes on each side, same endpoint/model/8192-token cap, temperature 0,
real EAGLE3 acceptance and no profiler. Prompt tokens per set: 2,535,349;
prefix-cache hits per set: 12,288 (0.485%) on both sides. An initial A->B
48-request comparison was followed by this B->A 96-request comparison with
fresh servers and matching per-set salts.

| Set | System tok/s old -> new | TTFT p50 ms old -> new | TPOT p50 ms old -> new | Output tokens old -> new |
|---|---|---|---|---|
| 1 | 506.1 -> 606.1 | 403.9 -> 339.9 | 12.002 -> 10.032 | 17,802 -> 16,763 |
| 2 | 474.9 -> 571.1 | 422.5 -> 337.3 | 13.326 -> 10.816 | 16,370 -> 15,672 |
| 3 | 475.1 -> 640.8 | 426.0 -> 339.5 | 12.882 -> 9.571 | 16,539 -> 17,842 |

All 576 measured requests succeeded. Whole-set wall time fell 20.0-21.4%,
TTFT p50 fell 15.9-20.3%, and TPOT p50 fell 16.4-25.7%. Output tok/s rose
19.8-34.9%, but outputs are not deterministic across dynamic batching:
token totals changed -5.8%, -4.3%, +7.9%. Thus this is observed serving
improvement for this prefill-heavy workload, not an exact fixed-output speedup
nor evidence of a full multi-turn AgentX throughput gain.

The replay harness emits per-request usage, TTFT, TPOT, content hashes, and
canonical responses. It measures first nonempty content/reasoning/tool delta;
TPOT is (stream completion - first delta)/(completion_tokens - 1).
It runs only first turns, without AgentX tool-delay/multi-turn scheduling.

Use `op_tests/multigpu_tests/bench_m3_pull_q4_replay.py TAG 96 SALT`
against each fresh server. Set M3_REPLAY_DATASET to the recipe JSONL path,
M3_REPLAY_OUTPUT to a local output directory, and optionally M3_REPLAY_API to
the base URL (default http://127.0.0.1:18890). Use the same salts 1,2,3 on both
sides, each exactly once per server; do not mix these runs with profiler or
other GPU benchmarks. Requires aiohttp. This harness is separate from and
does not modify the official Endpoints client.

Additional regression: TP2 INT4 and TP2 legacy INT4 passed with the flag set to 1 (100 graph replays). Other architectures and TP8 were not executed on this four-GPU node.
