# aiter.bench: unified kernel benchmarking

One timing interface for every aiter backend (FlyDSL, CK, ASM, HIP, Triton).
Every result records which mode, cache policy and environment produced it.
Design and roadmap: `op_tests/dump_data/docs/aiter_unified_benchmarking_plan.md`.

`run_perftest` in `aiter/test_common.py` is unchanged. `aiter.bench` sits
beside it.

## Quick start

```python
from aiter.bench import MeasureConfig, run_bench, time_all, time_fn

# Same calling convention as run_perftest, but returns the full result.
out, result = run_bench(my_kernel, x, w, out_buf, measure=MeasureConfig(mode="graph"))
print(result.median_us, result.cache_policy, result.graph_calls_per_replay)

# Any callable. Pass inputs through `args` so cache="rotate" can copy them.
result = time_fn(my_kernel, MeasureConfig(mode="device"), args=(x, w, out_buf),
                 flops=2 * m * n * k, nbytes=(m * k + n * k + m * n) * 2)
print(result.to_json())

# All three modes on the same inputs.
results = time_all(my_kernel, args=(x, w, out_buf))
host_overhead_us = results["eager"].median_us - results["graph"].median_us
```

`MeasureConfig` is the only place timing is configured. Pass the same object
to every call you want to compare.

## Timing modes

| Mode | Measures | How | Use it for |
|---|---|---|---|
| `device` (default) | GPU busy time per call: the sum of every kernel, memcpy and memset the call launches | torch.profiler hardware timestamps. Each call runs in its own profiler range, and a GPU operation belongs to the call whose range issued its launch (matched by correlation id, not kernel name). | Kernel work in isolation; per-kernel breakdown (`per_kernel_us`); comparable to `run_perftest` |
| `graph` | Steady-state time per call when replayed from a HIP graph, as graph-based serving runs it | K calls recorded into one graph; each sample is one replay ÷ K. K is chosen so one replay takes at least `graph_min_replay_us` (default 1000 µs). Fails with `EmptyGraphCaptureError` when nothing was captured. | **PR verdicts** (graph median); production-style A/B |
| `eager` | Per-call time including host launch overhead | GPU events around every call; one sync at the end (`do_bench` pattern). `eager_sync_each=True` drains every call, adding the full host round trip. | Non-graph serving; launch-overhead questions |

Reading them together:

| Observation | Meaning |
|---|---|
| `eager` ≫ `graph` | Host launch overhead dominates at this shape. |
| `device` `span_us` > `median_us` | Gaps between the call's kernels (`span_us − median_us`). |
| `graph` > `device` for kernels under ~2 µs | Graph replay pays a per-kernel GPU dispatch cost (~0.5 µs on MI350P) that profiler timestamps leave out. Real in serving. |
| `graph` < `device` for kernels of ~2–10 µs | Back-to-back kernels in a graph overlap dispatch with the previous kernel's tail; device mode times each kernel separately. |
| `graph` ≈ `device` above ~20 µs | Fixed per-kernel effects are negligible. |

So `device` and `graph` are different measurements, not two estimates of one
number. Compare like with like: same mode, cache policy and `MeasureConfig`.

### Why graph mode records K calls

A `g.replay()` costs the host about 6 µs and the GPU a further 5–10 µs of
fixed setup. With one call per graph, every kernel faster than that reads as
the replay cost (a 1.2 µs kernel measured 5.9 µs on MI350P). Spreading the
fixed cost over K calls bounds its share at roughly
`fixed_cost / graph_min_replay_us`, about 1% at the default.

## Cache policies

| Policy | Effect | Default for |
|---|---|---|
| `warm` | Same inputs every call; they stay cache-resident. | `graph` |
| `rotate` | Deep copies of the inputs, cycled so each call reads data that is not cache-resident (sized from the last-level cache, 256 MB Infinity Cache on gfx942/gfx950). Needs inputs passed as arguments. | `device` (matches `run_perftest`) |
| `flush` | Overwrite a buffer twice the last-level cache size before each call, outside the timed span. Not available in graph mode (use `rotate`). | `eager` (matches `do_bench`) |
| `auto` | Probe warm vs flushed calls; go cold (`flush`, or `rotate` in graph mode) when cold is more than 5% slower. The measured difference is in `cache_probe_delta`. | — |

`cache=None` (the default) picks the per-mode default above. The policy that
actually ran is always in `result.cache_policy`. A `rotate` that could not fit
enough copies adds a "partly warm" note.

## Results

`BenchResult` (`to_dict()`, `to_json()`, `from_dict()`, versioned by
`schema_version`):

- **Timing:** `samples_us`, `median_us`, `mean_us`, `min_us`, `max_us`,
  `std_us`, `outliers` (IQR count; outliers are reported, never dropped), and
  `us`: the statistic chosen by `MeasureConfig.metric` (`median` by default,
  `mean` to match `run_perftest` and the tuners).
- **Mode details:** `cache_policy`, `rotate_copies`, `graph_calls_per_replay`,
  `per_kernel_us`, `span_us`, `kernel_count`, `all_kernels_counted`.
- **Cost:** `compile_ms` (wall time of the first call, including JIT).
- **Throughput:** `tflops`, `gb_s` when `flops` / `nbytes` are given.
- **Context:** `system` (GPU, gfx, CU count, HIP/torch/Triton/FlyDSL versions,
  aiter git SHA), `gpu_state_before` / `gpu_state_after` (clocks, performance
  level, activity, power, temperature), `telemetry`, `notes`.

`notes` carries anything that weakens a number: a GPU already busy before
timing, a varying kernel count, operations the profiler could not attribute,
a partly warm rotation.

## Environment and telemetry

Nothing in `aiter.bench` changes GPU settings. Clocks and the performance
level are only recorded: from the `amdsmi` Python binding when it imports,
otherwise from the `amd-smi` CLI (`record_gpu_state=False` skips this).
`telemetry=True` samples clocks, power and activity on a background thread
during the timed window; it needs the `amdsmi` binding.

## Not yet implemented

Tracked in the plan: rocprofiler-sdk as an alternative device backend,
per-arch peaks (`tflops_peak_ratio`, `gbps_peak_ratio` are `None`), and the
case-level API (`bench`, `sweep`, `compare`, `run_candidates`, CLI).
