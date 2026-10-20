# Timing Modes

GPU kernel timings answer different questions depending on where the timer sits. Treat `graph`, `eager`, and `device` as separate views of the same kernel launcher. FlyKAT defaults to `graph` because graph replay is the fair production-style steady-state comparison for kernels that must run inside graph-based serving stacks. The production question is:

> If a kernel change moves timing mode X by delta, what does that mean for end-to-end inference?

## Delta To End-To-End Impact

`graph` is the default fair-to-production number. It captures the full launcher once, then times graph replay. A `graph` delta reflects changes in captured device work: kernel body quality, memory access, parallelism, and contractual GPU side effects. In graph-based serving, that captured work is what production replays.

`eager` is the per-call number for a serving stack that calls the kernel launcher for each inference step. That path pays device work, host dispatch, runtime launch round trips, and any per-call tensor preparation the launcher performs. If `eager` improves by `10 us`, a non-graph inference loop saves `10 us` per call to that operation.

The core mechanism is the host bubble. A timed eager sample records a GPU start event, calls the launcher once, records a GPU end event, and synchronizes. The start event fires on the GPU before the host has finished enqueuing the launcher's kernels. The GPU then sits idle until the host completes Python, dispatch, and launch work and the first queued kernel can begin. That idle gap sits inside the GPU-event span, so eager timing captures device work plus host dispatch and launch round-trip cost.

`device` is the hardware floor from dispatch timestamps. For a single-kernel operation, it is the isolated hardware duration of that kernel. For a multi-kernel operation, the contract-complete device number sums the hardware time across all non-noise contractual dispatches. A single selected dispatch is a microscope on one kernel, not the operation's latency authority.

The practical relationship is:

```text
eager ~= graph + host_overhead_us
```

Interpret deltas through that split:

| Observation | Meaning |
|---|---|
| `eager` moves by about the same amount as `graph` | Device work changed; host cost stayed roughly flat. |
| `eager` moves more than `graph` | Host-side cost changed too, such as an added or removed dispatch. |
| `graph` moves but `eager` barely moves | Device work changed, but host overhead dominates the production call at this shape. |
| `device` moves but `graph` does not | One dispatch changed, but total captured device work did not move the same way. Check other contractual dispatches and measurement noise. |
| `graph` moves but `device` does not | Captured work outside the inspected dispatch changed, or the device measurement is summing a different dispatch set than graph replay captures. |

Report `median_us` as the authoritative latency. An idle, clock-gated GPU intermittently runs a sample slow, which inflates `mean_us` but not the median. FlyKAT derives `tflops`, `gb_s`, sweep ordering, and baseline speedups from median latency. Keep `mean_us` and standard deviation as spread and noise context.

## Mode Summary

| Mode | What it measures | Contract-complete? | Requires hardware dispatch timestamps? | Production equivalent | Primary use |
|---|---|---|---|---|---|
| `graph` | A serial window of captured launcher graph replays, divided by replay count, including every captured GPU kernel and GPU side effect. | Yes on the device side. It captures the full launcher. | No | Yes, for graph-based serving stacks. | Default comparison for fair production-style steady-state latency. |
| `eager` | One full launcher call bracketed by GPU events, including device work plus the host bubble caused by per-call dispatch and launch round trips. | Yes. It runs the full launcher. | No | Yes, for a serving stack that invokes the launcher per inference step. | Estimate non-graph per-call impact and host overhead. |
| `device` | Hardware timestamp duration for GPU dispatches after framework/runtime noise filtering. | Yes when it sums all contractual dispatches. A single-dispatch report is complete only for single-kernel operations. | Yes | No | Analyze the hardware floor for one dispatch or a contract-complete dispatch set. |

## Contract Completeness

An operation contract includes every GPU side effect the launcher must produce. A timing mode is contract-complete when it includes all GPU work required by that contract.

Contracts can include work outside the main kernel body: output zeroing, output buffer initialization, scratch-space setup, reductions, scans, copies, or multi-phase dispatch sequences. Contract completeness asks whether the timing includes that required work, not just the kernel most interesting to optimize.

- `eager` runs the full launcher, so it includes contractual GPU side effects and host cost.
- `graph` captures the full launcher, so its replay includes the contractual GPU side effects on the device.
- `device` is contract-complete when it sums hardware time across all non-noise dispatches required by the contract.

Treat this as a measurement contract, not a profiler detail. If the operation's contract requires multiple dispatches, report their summed hardware time for the operation's device cost. Report one dispatch only when the operation has one contractual GPU dispatch or when the question is specifically about that dispatch.

## Mode Details

### `eager`

Each eager sample records a start event, runs one complete launcher call, records an end event, synchronizes, and stores the elapsed GPU-event time in microseconds.

This includes:

- GPU execution time for all work the launcher enqueues.
- The host bubble between the start event and the first queued kernel beginning execution.
- Host dispatch, language runtime, and launch round-trip cost that keeps the GPU idle inside the measured span.
- Any host-side tensor preparation done inside the launcher's per-call path.
- Extra per-call launch cost from implementations that split work across multiple dispatches.

Keep eager per-call. Do not bracket many back-to-back launcher calls under one event pair and divide. That batching keeps the queue full, lets the host race ahead, removes the GPU idle gap between calls, and erases the host bubble that eager exists to measure.

For production-style inference calls that invoke the launcher each step, `eager` maps directly to per-call latency.

### `graph`

Graph timing warms up the launcher, captures one complete launcher call into a CUDA/HIP graph, then times graph replay with GPU events. Warmup must happen before capture because capture forbids allocation, JIT, and most host work. The runner that owns the captured kernels must stay alive for the replays to remain valid.

The correct measurement brackets a window of back-to-back graph replays under one event pair and divides by the replay count. The replays run on one stream. A stream is a FIFO, so the replays serialize; this is serial back-to-back device time, not a throughput or overlap measurement.

Dividing by the replay count removes two host-side costs graph mode is meant to strip:

- Per-replay event-record and synchronization overhead, which can dominate small replays.
- The GPU idle bubble between replays where the GPU waits for the host to issue the next replay.

This leaves the cleanest production-style steady-state number for the full captured launcher. It includes every captured contractual GPU kernel and side effect, but excludes normal host dispatch and launch round trips. Use it to compare kernel bodies and launcher device work under the operation's contract. Do not use it as the authority for one kernel's isolated hardware duration; use `device` for that.

Empty capture is a hard error. A FlyDSL launcher that omits a trailing `stream: fx.Stream` parameter runs on the HIP NULL stream; `torch.cuda.graph` captures a side stream and drops that launch, leaving an empty graph. FlyKAT raises `EmptyGraphCaptureError` instead of returning a meaningless ~1 microsecond replay. Pass `torch.cuda.current_stream()` at launch time and forward it to `.launch(..., stream=stream)` so graph capture records the work.

### `device`

Device timing runs the launcher under a profiler that exposes hardware dispatch timestamps, such as rocprofiler-sdk, then reads start and end timestamps for GPU dispatches. The measurement filters framework and runtime noise such as tensor fills, runtime copies, and allocator or library setup work.

This includes:

- Per-dispatch hardware execution time.
- Summed hardware time across all non-noise dispatches required by the operation's contract.
- No host dispatch time.
- No runtime launch round-trip time.

Use `device` when the question is about hardware behavior: occupancy, memory bandwidth, instruction mix, instruction scheduling, or the isolated duration of a single dispatch. Use summed contractual dispatches when the question is the operation's hardware floor.

## Measurement Hygiene

- **Warmup and steady state.** First calls pay one-time costs: JIT or AOT compilation, autotuning, kernel-cache population, allocator growth, and first-touch page faults. Discard warmup iterations and measure steady state; run warmup on the same stream and execution path as the measured work.
- **GPU clocks and thermal drift (DVFS).** GPU clocks ramp under load and throttle under heat, so back-to-back measurements drift. For reproducibility, lock clocks when the platform allows, or warm long enough for clocks to settle and keep runs short enough to avoid thermal throttling; a faster result can mean a higher boost clock.
- **Asynchronous execution and synchronization.** GPU work enqueues asynchronously. A CPU wall-clock timer, such as a host monotonic clock, around a launch measures enqueue time unless it synchronizes, and synchronizing every call adds host overhead and serializes work; GPU timing events recorded on the stream measure the GPU-side span, which is why the modes above use GPU events instead of host wall-clock timing.
- **Cache state across iterations.** Repeated back-to-back iterations leave L2 cache and TLB state warm, so a tight timing loop can report cache-warm numbers that a cold serving stack call would not see. If the production access pattern is cold, flush or pollute cache state between samples; if production is warm, leave it warm and state that choice.
- **Input-data dependence.** Data-dependent kernels, including sorting, sparsity, early exit, reductions over masked data, and dynamic routing, time differently per input. Fix the seed and input distribution so samples are comparable, and measure representative inputs rather than degenerate all-zero or all-equal cases.
- **Observer effect.** Collecting hardware dispatch timestamps or performance counters can serialize dispatches and add overhead, so profiling perturbs timing; device-mode hardware-counter measurements are not free of measurement effect even though they exclude host cost. Graph capture also changes execution from the normal per-call launch path, so cross-check modes instead of trusting one in isolation.
- **Sample count and statistical stability.** Take enough samples that `median_us` stabilizes. Treat a wide or multi-modal spread as a signal of contention, throttling, or data dependence rather than a clean kernel measurement; outliers usually indicate interference from other GPU work, CPU scheduling, or throttling.
- **Compare like-for-like.** Compare only measurements taken with the same mode, warmup, sample count, cache policy, input distribution, and clock state. A delta between runs that differ in any of these inputs is not attributable to the kernel.

The three modes answer what changed; measurement hygiene makes each number trustworthy enough to attribute that change to the kernel.

## Reading The Three Together

Start with `graph` for the fair production-style answer. Use `eager` to measure non-graph per-call host overhead. Use `device` to explain hardware behavior once the device-side delta matters.

For small shapes, host overhead can dominate. In that case, a strong `device` improvement can disappear in `eager` until the launcher shape changes or the workload grows enough for device work to dominate. For larger shapes, `eager` and `graph` converge when host overhead becomes a small fraction of total time.

For multi-kernel operations, compare contract-complete `graph` with contract-complete summed `device` before interpreting one dispatch. A change that improves one kernel but adds or slows another contractual dispatch can make that dispatch look better while `graph`, summed `device`, and `eager` get worse.
