# MiniMax-M3 decode monokernel (experimental)

A persistent decode monokernel for MiniMax-M3 on MI355X (gfx950, TP4). It runs inside vLLM's native M3 AMD model and has two backends:

- **FlyDSL**: ATOM's FlyDSL M3 mono, vendored as `vllm.models.minimax_m3.amd.mono` (`m3mono/`). The pristine ATOM sources (HEAD 67c54da) are in `vendor_src/`. The vendored `kernels/post_attn.py` adds a stage mask, `VLLM_M3_FLY_STAGES`, used for ablation.
- **Gluon**: a new backend in `m3mono/gluon/`, with one launch per layer, 256 CTAs × 512 threads, and tagged-pair mailbox hand-offs.

See `docs/DESIGN.md` for the design.

## Status (2026-10-16)

The Gluon output is bit-exact against FlyDSL at 1 and 2 rows. At 4 rows the nRMSE is about 2e-5.

16-layer microbench (`bench/bench_layer_tp.py`, TP4, ctx 1536), µs per layer:

| rows | FlyDSL | Gluon | ratio |
|---|---|---|---|
| 1 | 37.1 | 47.7 | 1.29x |
| 2 | 46.8 | 61.3 | 1.31x |
| 4 | 57.6 | 89.9 | 1.56x |

Per-region ablation at 1 row (`bench/ablate.py`), µs:

| region | FlyDSL | Gluon |
|---|---|---|
| qkv | 2.80 | 4.75 |
| split | 4.34 | 5.15 |
| merge+o | 4.87 | 7.25 |
| router | 2.90 | 3.44 |
| moe | 11.28 | 12.21 |

The largest remaining gap is merge+o at 2 or more rows. FlyDSL gives each token its own merge task, while every Gluon o CTA merges all the tokens itself.

## Layout

- `m3mono/`: the vLLM package, covering both backends, the runtime and the vLLM runner.
- `bench/`
  - `bench_layer_tp.py`: layer microbench, with `--timeline` for in-kernel stamps.
  - `ablate.py`: per-region replay ablation, with `--dump` for ISA.
  - `isa_mix.py` and `isa_lines.py`: ISA comparison.
  - `tl_*.py`: timeline analysis.
- `integration/`: the vLLM model patch, install script, and serve/bench scripts.
- `tests/`: compile checks and probes.

Benchmarks run under torchrun with 4 ranks, for example:

    torchrun --nproc-per-node 4 bench/ablate.py --rows 1
