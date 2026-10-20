# Kernel Measurement Reference

## Verified Sources

- `src/flykat/tools/bench/models.py`: `TimingStrategy = Literal["eager", "graph", "device"]`, `parse_timing_strategy`, `MeasureConfig`, `TimingStats`, `BenchOutput`.
- `src/flykat/api/bench.py`: `bench`, `bench_many`, `bench_all`, `BenchAllOutput`, `bench_provider`.
- CLI help: `flykat --help`, `flykat bench --help`, `flykat run --help`, `flykat exec --help`.
- `src/flykat/api/models.py`: `PlanMeasureOutTriple.measure`, `Sweep.measure`, `[bench]` parsing for spec files.
- `src/flykat/api/sweep.py`: sweep objectives and candidate sorting.
- `docs/architecture.md` and `AGENTS.md`: FlyKAT is stateless; timing config flows through `MeasureConfig`.

## MeasureConfig

`MeasureConfig` is the single timing object:

- `warmup_iters: int = 3`, validated `>= 0`
- `bench_iters: int = 20`, validated `>= 1`
- `graph_replay_iters: int = 100`, validated `>= 1`
- `trials: int = 1`, validated `>= 1`
- `strategy: "eager" | "graph" | "device" = "graph"`

`parse_timing_strategy(value)` accepts `eager`, `graph`, or `device`; unknown values raise `ValueError`.

Use the same `MeasureConfig` before and after a kernel change. If a wrapper needs timing options, thread the object through `measure=...` instead of adding separate `warmup_iters`, `bench_iters`, `trials`, or `strategy` kwargs.

## Timing Strategies

- `graph`: the default fair production-style strategy. It warms the launcher, captures one full launcher call into a CUDA/HIP graph, then times serial graph replays with events.
- `eager`: runs warmups, then records each full launcher call with `torch.cuda.Event` timing and synchronizes per sample.
- `device`: runs rocprofiler-sdk in a child process and reads hardware dispatch timestamps from kernel-trace CSV output. It is the hardware-floor reference, target/tooling dependent, and can be unavailable.

Graph timing raises `EmptyGraphCaptureError` when capture records zero nodes. For FlyDSL kernels, that usually means the `@flyc.jit` launcher omitted the trailing `stream: fx.Stream` parameter and launched on the HIP NULL stream, outside the side stream captured by `torch.cuda.graph`.

Use `bench_all(plan, measure=..., per_strategy=...)` in Python when comparing all strategies. The code supports per-strategy overrides; use fewer device iterations when rocprofiler overhead dominates the measurement loop.

## CLI Commands

`flykat bench`:

```bash
flykat bench <operation> --kernel <backend:kernel> --set key=value --tunable key=value --mode graph --warmup-iters 3 --bench-iters 20 --trials 1 --format json
```

Verified options: `--kernel`, `--set`, `--tunable`, `--warmup-iters`, `--bench-iters`, `--trials`, `--graph-replay-iters`, `--mode`, `--impl`, `--format`.

`--mode` accepts `eager`, `graph`, `device`, or `all`, and defaults to `graph`. `--mode all` returns `BenchAllOutput` with `eager`, `graph`, and `device` fields when available; unavailable strategies are `None`.

`flykat run`:

```bash
flykat run <operation> --kernel <backend:kernel> --set key=value --config config.json --timing graph --warmup-iters 3 --bench-iters 20 --seed 0 --out artifacts/run --format json
```

Verified options: `--kernel`, `--set`, `--tunable`, `--config`, `--timing`, `--warmup-iters`, `--bench-iters`, `--seed`, `--rtol`, `--atol`, `--out`, `--name`, `--format`. `--timing` defaults to `graph`. `run` calls `inspect`, so it verifies and benchmarks in one report path.

`flykat exec` for sweep specs:

```bash
flykat exec path/to/sweep.toml
```

`exec` auto-detects sweep TOML by file shape. Put timing in the spec:

```toml
[bench]
warmup_iters = 10
bench_iters = 100
timing = "graph"
```

`PlanMeasureOutTriple` and `Sweep` parsers read `warmup_iters`, `bench_iters`, `trials`, `graph_replay_iters`, and `timing` from `[bench]`. Omitted `timing` defaults to `graph`.

## BenchOutput Fields

Read these fields from JSON or Python results:

- `samples_us`: all timed samples across trials.
- `median_us`: authoritative aggregate latency in microseconds.
- `mean_us`, `min_us`, `max_us`, `std_us`: aggregate latency and spread in microseconds.
- `timing_strategy`: selected strategy.
- `compile_ms`: first launch timing, including JIT/compile plus first run.
- `tflops`, `gb_s`: derived from operation flop/byte models and `median_us` when available.
- `tflops_peak_ratio`, `gbps_peak_ratio`: fraction of known target peak when FlyKAT recognizes the GPU arch; otherwise `None`.
- `arch`: detected GCN arch string when peak lookup succeeds.
- `trial_mean_us`, `trial_std_us`: statistics over per-trial median latency when `trials > 1`; otherwise `None`.

For before/after comparison, keep the same command shape and compare `median_us`, `samples_us`, `mean_us`, `trial_mean_us`, and `trial_std_us`. A lower `min_us` with worse median or high `trial_std_us` is weak evidence.

## Sweep Reading

`sweep` verifies each candidate before timing by default, records unsupported candidates with a reason, and sorts passing candidates by objective. Verified objectives are `median_us`, `min_us`, `mean_us`, `gb_s`, and `tflops`; the default objective is `median_us`.

Each `Candidate` contains:

- `kernel`
- `config`
- `tunables`
- `supported`
- `reason`
- `verify_ok`
- `bench`

## Cut Source Material

The external FlyDSL optimization notes were conceptual only. Cut these items because they were not verified FlyKAT measurements:

- External repo paths, helper scripts, setup commands, and provider-specific benchmark tooling.
- Hardcoded TFLOPS, bandwidth, latency, peak-ratio targets, and recorded gfx950/gfx942 results.
- CK DSL and CK Tile command patterns, result schemas, and framework-specific optimization claims.
- Architecture-specific LDS/VGPR formulas as general FlyKAT expectations.
- Any instruction to treat upstream performance numbers as promotion evidence.

Use FlyKAT measurements from the same config, strategy, and GPU as promotion evidence. Prefer graph median for the decision number and device timing for hardware-floor analysis.
