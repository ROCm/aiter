# DSR1 EP4 preset for 9216 global tokens

This workload-specific preset reduces full-layer latency from **565.81 us to
536.96 us** (5.10%) on gfx1250 with 256 CUs. The control is the previous M128
winning preset, remeasured in the same sweep. The corresponding fixed-workload
throughput increase is 5.37%; this is a MoE microbenchmark, not whole-model
throughput.

The sweep used source `b30555b6b128e14f98451cf8069b2c1cabf44e14`, tested 34
configurations in 42 successful runs, and repeated the three finalists and the
control three times in interleaved, reversed order. All runs passed correctness.
The frozen DSR1 baseline and shared default tuning table are unchanged.

## Configuration

| Setting | Control | Selected |
|---|---|---|
| GEMM1 / GEMM2 tile | 128x256x256 | 128x256x256 |
| GEMM1 / GEMM2 buffers | 3 / 3 | 3 / 3 |
| GEMM1 / GEMM2 wave grid | 1x4 / 1x4 | 1x4 / 1x4 |
| GEMM1 / GEMM2 cluster | 2x2 / 1x4 | 1x1 / 1x1 |
| Persistent workers | 512 | 512 |
| Next-stage prefetch | On | Off |
| Route-parallel dispatch | 256 blocks x 16 waves | 512 blocks x 16 waves |
| MOC / A preshuffle | On / On | On / On |
| Scatter overlap | Off | Off |

Both configurations use B hint 0, rolled drain 0, split-K 1, two TDM waves,
and LDS load interleaving. M128 does not support the current scatter-overlap
path. The one-row CSVs are intentional overrides for this fixed workload:
9216 global tokens = EP4 x TPR2304, with effective CSV token key 16384,
expert=65, topk=-1, and ep_fused=1. Runtime routing uses top-k8 plus one fused
shared expert, with 256 routed experts, hidden dimension 7168, and intermediate
dimension 2048.

The token=16384 bucket also serves other sizes, including TPR4096. These results
only validate TPR2304, so the preset is opt-in rather than a shared table update.

## Reproduce

From the repository root, use an environment with the repository's requirements
(including the pinned FlyDSL 0.5.3) installed:

```bash
bash op_tests/multigpu_tests/presets/dsr1_global9216/run.sh selected
bash op_tests/multigpu_tests/presets/dsr1_global9216/run.sh control
```

`DSR1_PYTHON` selects the interpreter (default `python3`). `DSR1_CACHE_ROOT`
selects the compilation-cache directory (default `/tmp/aiter-dsr1-global9216-$UID`).
The runner resolves repository and CSV paths relative to itself. It uses
`/usr/local/bin/gpu-lock` when installed and checks `rocm-smi --showpids`
immediately before launching. It exits with code 75 while GPUs are busy.

The fixed benchmark command uses norm initialization, uniform routing, seed 0,
one layer, 10 warmups, 50 timed graph replays, 10 native profiler replays, and
accuracy verification. The runner prints the revision, import paths, environment,
and native timing/profile tables. Run from an environment without additional
GEMM tuning overrides to reproduce the CSV settings.

## Measured results

All values are microseconds. Each table entry is a median across three runs.
Full-layer latency is the benchmark's average across four ranks of per-rank
host-synchronized graph-replay medians. It includes benchmark RMSNorm and residual
work. Kernel values are per-call rank-average native profiler timings; their sum
is not full-layer latency.

| Metric | Control | Selected |
|---|---:|---:|
| Full layer | 565.815 | 536.957 |
| GEMM1 | 171.790 | 164.145 |
| GEMM2 + scatter | 168.877 | 158.303 |
| Dispatch | 50.226 | 42.129 |
| Compact plan | 24.202 | 23.406 |
| A preshuffle | 18.420 | 18.272 |
| Combine | 24.168 | 23.909 |
| Combine sync | 9.120 | 7.899 |

Selected per-run layer times: **535.909, 537.439, 536.957**.
Control per-run layer times: **567.390, 565.815, 565.215**.
The 1024-worker finalist measured 538.077 us; the mixed-wave finalist measured
538.959 us. Their small differences from the winner should not be interpreted as
large independent worker/wave-layout gains.

All 42 runs passed with logits_diff=0.0215067122 against tolerance=0.0327352085.
This verifies the microbenchmark against its reference, not model accuracy.
Actual kernel symbols confirmed the selected tiles, buffers, waves, clusters,
workers, MOC, preshuffle, prefetch, and dispatch geometry.

`results.json` records all 34 candidate configurations as changes from
`control.csv`, their environment overrides, aggregates, and all 42 per-run native
summaries and relevant kernel timings. Single-run cases are screening results.
NB2, alternative N/K tiles, mixed M128/M256 tiles, B hint 1, rolled drain, and the
tested compact-plan variants did not improve this workload. Split-K was excluded
because the MOC placed-weight path does not support it.

The packaged runner was additionally executed with the one-row selected CSV:
537.491 us, correctness PASS, and matching kernel symbols. This smoke validation
is recorded separately in `results.json`; it is not included in the sweep's
three-run medians.
