# MiniMax-M3 decode monokernel (experimental)

A persistent decode monokernel for MiniMax-M3 on MI355X (gfx950, TP4). It runs inside vLLM's native M3 AMD model and has two backends:

- **FlyDSL**: ATOM's FlyDSL M3 mono, vendored as `vllm.models.minimax_m3.amd.mono` (`m3mono/`). The pristine ATOM sources (HEAD 67c54da) are in `vendor_src/`. The vendored `kernels/post_attn.py` adds a stage mask, `VLLM_M3_FLY_STAGES`, used for ablation.
- **Gluon**: a new backend in `m3mono/gluon/`, with one launch per layer, 256 CTAs × 512 threads, and tagged-pair mailbox hand-offs.

See `docs/DESIGN.md` for the design.

## Status (2026-10-28)

The Gluon backend now matches or beats FlyDSL at the AgentX shape: 57 sparse layers, TP4, EAGLE3 verify with
4 rows per request. Numbers are µs per decode step from `bench/bench_agentx.py`, min of 10 steps, same run for
both backends:

| concurrency | context | FlyDSL | Gluon | Gluon / FlyDSL |
|---|---|---|---|---|
| c=1 (4 rows) | 2K | 3245 | 3233 | 0.996 |
| c=1 (4 rows) | 320K | 3506 | 3498 | 0.998 |
| c=2 (8 rows) | 2K | 4618 | 4659 | 1.009 |
| c=2 (8 rows) | 320K | 5128 | 5206 | 1.015 |

Correctness: with pristine KV and index caches before each backend's first step (`--check` now restores them),
the 4-layer c=2 output is bit-exact against FlyDSL at 1K, 2K and 320K. The c=1 residual against FlyDSL is a FlyDSL
defect on speculative row 2 at selection-reuse layers. `bench/ref_attn.py` checks both backends against a
kernel-pipeline-faithful attention reference: Gluon is 5e-3 from it on that row, FlyDSL 5e-2.

An own-key stale-row race in the Gluon split attention was found and fixed (knob `VLLM_M3_GLUON_TAIL_ACQ`, default
8: drain, CTA barrier, L1 invalidate, barrier after the wave-0 flag polls). It costs ~25 µs per step. See
`docs/OWN_KEY_RACE_HANDOFF.md` for root cause, evidence and repro commands.

MiniMax-M3.1-Flash at the AgentX MI355X c=1 shape (DSpark 7 drafts, 8 rows, 60 layers), current kernels on
synthetic M3 weights: Gluon / FlyDSL 1.005 at 2K, 1.017 at 320K. The M3.1-specific changes (MXFP8 shared expert,
load-time attention requant) are not ported yet.

## Layout

- `m3mono/`: the vLLM package, covering both backends, the runtime and the vLLM runner.
- `bench/`
  - `bench_layer_tp.py`: layer microbench, with `--timeline` for in-kernel stamps.
  - `ablate.py`: per-region replay ablation, with `--dump` for ISA.
  - `isa_mix.py` and `isa_lines.py`: ISA comparison.
  - `tl_*.py`: timeline analysis.
  - `bench_agentx.py`: AgentX-shape step bench (`--reqs`, `--ql`, `--ctx`, `--layers`, `--check` against FlyDSL from
    pristine caches). `bax2.sh` / `ab2.sh` wrap it per TP4 GPU set with a lock; `freeset.sh` picks a free set.
  - `debug_layer_spec.py`: per-stage hand-off comparison of one or more layers (speculative rows, `M3_NB`).
  - `ref_attn.py`: per-row attention reference (float, fp8, kernel-pipeline emulation) against both backends.
- `m3mono/gluon/aot.py`: loads offline-compiled `m3_step_w{S}_q{QL}.hsaco` binaries (`VLLM_M3_GLUON_AOT_DIR`).
- `docs/OWN_KEY_RACE_HANDOFF.md`: the own-key race write-up.
- `integration/`: the vLLM model patch, install script, and serve/bench scripts.
- `tests/`: compile checks and probes.

Benchmarks run under torchrun with 4 ranks, for example:

    torchrun --nproc-per-node 4 bench/ablate.py --rows 1
