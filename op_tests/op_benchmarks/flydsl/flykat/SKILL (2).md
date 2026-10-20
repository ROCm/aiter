---
name: kernel-measurement
description: Guides trustworthy FlyKAT kernel timing with MeasureConfig, eager/graph/device strategies, bench/run/exec commands, and BenchOutput interpretation. Use when validating whether a kernel optimization actually improved performance.
disable-model-invocation: true
---

# Kernel Measurement

## Workflow

1. Verify correctness before timing. A faster broken kernel is not promotion evidence.
2. Measure the old kernel or tunables first, then measure the changed version with the same operation config, input seed when available, warmup count, bench count, trials, and timing strategy.
3. Keep timing configuration in `MeasureConfig`: `warmup_iters`, `bench_iters`, `trials`, and `strategy`. Do not add parallel timing kwargs or TOML fields that bypass it.
4. Start with the default `graph` strategy for fair production-style comparison. Use `eager` to inspect non-graph host overhead, and `device` when hardware dispatch timestamps or the hardware floor are needed.
5. Increase `--warmup-iters` until setup/JIT/cache effects no longer move the timed samples. Increase `--bench-iters` and `--trials` when `samples_us`, `trial_mean_us`, or `trial_std_us` show noisy results.
6. Treat absolute latency, TFLOPS, bandwidth, and peak ratios as target-dependent; measure on your GPU.
7. Compare `median_us` first, then `mean_us`, spread, and `samples_us`. Prefer JSON output when recording before/after evidence.

## Commands

Use the live CLI help before quoting syntax in documentation or plans:

```bash
flykat --help
flykat bench --help
flykat run --help
flykat exec --help
```

Common timing commands:

```bash
flykat bench <operation> --kernel <backend:kernel> --set key=value --warmup-iters 10 --bench-iters 100 --trials 5 --format json
flykat bench <operation> --kernel <backend:kernel> --set key=value --mode all --warmup-iters 10 --bench-iters 100 --trials 3 --format json
flykat run <operation> --kernel <backend:kernel> --set key=value --timing graph --warmup-iters 10 --bench-iters 100 --format json
flykat exec path/to/sweep.toml
```

Read [reference.md](reference.md) for verified fields, strategy semantics, sweep-spec timing, and source material that was cut.
