# Record and replay aiter op calls

Kernel microbenchmarks use synthetic inputs (uniform routing, exact token
counts). Frameworks call aiter with real routing, padded batches and their own
environment, so the two can rank kernels differently. `AITER_RECORD` captures
the calls a framework makes; the replay scripts rerun them standalone, on the
same aiter dispatch path, without the framework.

Supported ops: `fused_moe`.

## 1. Record (in the framework)

Use an aiter build that includes the recorder, run the framework **in eager
mode** and set:

| Variable | Meaning |
|---|---|
| `AITER_RECORD=<dir>` | turn recording on; one bundle per process under `<dir>` |
| `AITER_RECORD_OPS=fused_moe` | ops to record (default: all supported) |
| `AITER_RECORD_SKIP_CALLS=N` | ignore the first N calls of each op (warm-up) |
| `AITER_RECORD_MAX_CALLS=N` | stop after N recorded calls per op (default 100000) |
| `AITER_RECORD_EVERY=N` | record one call in N |

Example with vLLM:

```bash
export VLLM_ROCM_USE_AITER=1                 # vLLM only uses aiter MoE when set
export AITER_RECORD=/data/rec/minimax_m3_tp4 AITER_RECORD_OPS=fused_moe
vllm serve <model> --tensor-parallel-size 4 --enforce-eager
# run the workload (e.g. vllm bench serve ...), then stop the server
```

SGLang: `SGLANG_USE_AITER=1` and `--disable-cuda-graph`.

Why eager mode: in graph mode the framework replays captured graphs and no
Python runs per step, so only the capture-time calls (dummy routing) would be
recorded. Those are still written, flagged `capturing`, and skipped by replay
unless `--include-capture` is given.

Each bundle (`<dir>/rank<r>-<host>-pid<pid>/`) holds:

- `session.json`: aiter commit, GPU, `AITER_*` env, framework patches of aiter
- `calls.jsonl`: one line per call: argument shapes/dtypes/strides/attributes,
  scalars, and the kernels aiter chose
- `payloads/`: contents of routing tensors (`topk_ids`, `topk_weight`, masks)
- `configs/`: the tuned config files that were in effect

Weights and activations are not saved; replay fills them synthetically.
Recording copies routing tensors to the host on every call, so the run is
slower than usual; its own performance numbers are not meaningful.

To see what a bundle contains (GPU, aiter commit, patches, configs, and the
calls grouped by weight shape, M and kernels):

```bash
python op_tests/replay/inspect_bundle.py <bundle-dir>
```

## 2. Replay (any machine with the same GPU, no framework needed)

```bash
python op_tests/replay/replay_fused_moe.py /data/rec/minimax_m3_tp4/rank0-... \
    --pad-to-graph-sizes 1,2,4,8,16,24,32,48,64 --out results/
```

Before importing aiter the script restores the recorded `AITER_*` environment
and points `AITER_CONFIG_FMOE` at the bundle's tuned config, so `fused_moe`
selects kernels as it did in the framework. Each call is checked against the
recording (exit code 1 if any check fails):

- kernels: the kernels aiter selects must be the recorded ones
- layout: every rebuilt tensor must have the recorded shape, strides, storage
  offset, dtype and attributes (`is_shuffled`, ...). Data-pointer alignment is
  not reproduced; lower alignment than recorded is reported as a note only

`--pad-to-graph-sizes` pads each call's token count up to the batch sizes the
framework captured graphs for, as graph mode would. Padding rows reuse the
routing of real rows.

Timing: each call is run `--iters` times, with caches flushed between
iterations by a read-only pass over a 1 GB buffer (`--no-flush` to keep them
warm). Two numbers are reported (`--timer both|kernel|wall`):

- `kernel_us`: sum of the device time of every kernel the call launches
  (torch profiler). This is what aiter op_tests (`run_perftest`) and rocprof
  report, so compare with those.
- `wall_us`: GPU time from the call's first to its last kernel (CUDA events,
  with a GPU busy-wait queued in front so launch overhead before the call is
  hidden). It adds the gaps between the call's kernels and any time the GPU
  waits for the host inside the call. `wall_us - kernel_us` is that overhead.
- `aiter_us` (`--timer aiter` or `--timer all`): aiter's own
  `run_perftest`, called exactly as `op_tests` call it (5 iterations, 2
  warm-up, inputs rotated through deep copies; `--aiter-iters`,
  `--aiter-warmup`, `--aiter-rotate`). Use it to match op-test numbers
  exactly. The rotation copies all inputs, weights included, several times,
  so it needs much more memory on large recordings.

Every layer keeps its own weight buffer (`--shared-weights` to use one set).

Outputs:

- `replay_calls.csv`: per call: M, padded M, bucket, recorded vs replayed
  kernels, layout check, kernel/wall/aiter times, and a per-kernel breakdown
  (`kernel_breakdown`, median us per kernel name)
- `replay_summary.csv`: per M bucket: number and share of calls, M values
  seen, kernels, kernel-match and layout-match rates, median and total times

`--check-only` verifies kernels without timing.

## Limitations

- The replay uses stock aiter. If the framework patched aiter (reported in
  `session.json` and by the replay), its path is not reproduced.
- `fused_moe` calls with `stage2_scatter` (EP scatter handles) are not
  replayable and are reported as skipped.
- Replay times a call in isolation; interference from other kernels in the
  framework is not reproduced. Compare with an in-framework trace when needed.
