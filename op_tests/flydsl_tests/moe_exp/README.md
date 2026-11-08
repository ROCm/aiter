# FlyDSL MoE kernel timing on gfx1250 (`flydsl_moe_exp`)

Times the FlyDSL grouped MoE kernels of the DSR1 prefill MoE, gemm1 by default (`launch_gemm_a8w4_tdm` in
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`): 2048 tokens, 64 experts, top 8, model dim 7168,
inter dim 2048, SiLU, a4w4 (MXFP4). Every dispatch's kernel time comes from `rocprofv3 --kernel-trace`.

| File | What it does |
|---|---|
| `bench.sh` | runs the processes: one compile process per variant, then the timed processes, interleaved |
| `bench_driver.py` | one process: runs `test_flydsl_grouped_gemm.py` and replaces its timing with rounds of dispatches |
| `summarize.py` | kernel time per round from the trace, per process and per variant; accuracy; output comparison |
| `with_agt.sh` | optional power-management log during a process (`agt_internal`, needs sudo) |
| `shapes_tuned_grouped_fmoe.csv` | the tuned 2048-token gemm1 config for three more shapes (see Experiment switches) |
| `results/` | runs made with an earlier version of this harness (torch-profiler and CUDA-graph timers) |

## Running

From the repository root, with `PY` set to a python that has torch, AITER and FlyDSL, and `ROCM_PATH` set to that
python's ROCm (`$(rocm-sdk path --root)` with the ROCm Python wheels):

```bash
op_tests/flydsl_tests/moe_exp/bench.sh [-p PROCS] [-r ROUNDS] [-d DISPATCHES] [--no-profiler] [-o OUT] [VARIANT ...]
```

| Option | Meaning | Default |
|---|---|---|
| `-p PROCS` | timed processes per variant | 5 |
| `-r ROUNDS` | rounds per process | 50 |
| `-d DISPATCHES` | dispatches per round (a multiple of `MOE_EXP_ROTATE`) | 128 |
| `--no-profiler` | run the timed processes with no profiler | `rocprofv3 --kernel-trace` |
| `-o OUT` | output directory | `moe_exp_runs/<UTC stamp>` |
| `VARIANT` | `name` or `name:ENV=VAL,ENV=VAL,...`, added on top of the default environment | `base` |

On a shared node add `LOCK=gpu-lock IDLE_CHECK=1` (the node's lock command, and an idle check inside the lock before
each process). `bench.sh --help` lists the other settings: tokens, timed kernels, weight rotation, IR dumps, a
variant's own FlyDSL cache, the power log.

## What a run does

1. Per variant, one untimed compile process `p0` (no profiler, one round) compiles the kernels into the variant's
   FlyDSL cache.
2. Timed processes `p1` to `pPROCS`, interleaved across variants (`p1: a b c`, `p2: a b c`, ...), load the kernels
   from that cache (`FLYDSL_RUNTIME_RUN_ONLY=1`), so nothing compiles under the profiler.
3. In each process, the test builds the inputs and runs the MoE once for its accuracy check. For each timed kernel the
   driver then launches once on each of the 4 rotated weight copies, captures DISPATCHES launches as one CUDA graph,
   idles the GPU for 1 s, and runs ROUNDS rounds. A round is one replay of the graph between two CUDA events, then one
   synchronize. The launches in the graph cycle through the weight copies, so each one reads its weights from HBM.
4. The graph keeps the host out of the rounds. Launching gemm1 from Python costs the host about 34 us, so a launch loop
   would leave the GPU idle between kernels shorter than that (gemm1 with 256 or fewer tiles takes about 15-22 us),
   and the time per dispatch would be the launch interval. For longer kernels the graph and a launch loop give the same
   time.

## Output

- `OUT/summary.txt`: per variant and kernel, the kernel time (median over rounds of the per-round mean, then median
  over processes; for all rounds and for the second half), the CUDA-event time per dispatch, TB/s from a byte model,
  the kernel names, the accuracy line, and a bit-exact comparison of the saved outputs.
- `OUT/rounds_<variant>_<kernel>.csv`: kernel time and event time per round, medians over the processes.
- `OUT/<variant>_p<k>.log`: each process's log, with `[moe-exp] gemm1: event_us = ...` and
  `[sanity a4w4 silu] kernel: logits_diff=...`.
- `OUT/trace-<variant>_p<k>/`: the rocprofv3 kernel trace; `OUT/out/<variant>_p<k>.pt.rounds.json`: the host
  timestamps of every round (rocprofv3's clock), which split the trace into rounds.
- `OUT/meta.txt`: node, versions, firmware, settings.

## Experiment switches

Environment variables for variants (`name:ENV=VAL,...`), all off by default. They are for experiments only.

- `AITER_CONFIG_GROUPED_FMOE=$PWD/op_tests/flydsl_tests/moe_exp/shapes_tuned_grouped_fmoe.csv`: AITER reads its
  grouped-MoE configs from this CSV, which gives four shapes the tuned 2048-token gemm1 config: 64 experts x 2048
  tokens top 8 (the default shape), 16 experts x 512 tokens top 8, and 1 expert x 256 tokens top 1 with inter dim 131072 or 32768 (dense
  256 x 262144 and 256 x 65536 GEMMs). Set the shape with `MOE_ARGS`, for example
  `MOE_ARGS="--scenario kernel --data-format a4w4 --experts 1 --tokens 256 --topk 1 --model-dim 7168 --inter-dim 32768
  --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform --scale-init uniform"`.
- `AITER_GROUPED_UNIFORM_M=1` (`grouped_moe_gfx1250.py`): sizes M for uniform routing, experts x align_up(tokens x
  topk / experts, tile), instead of the CUDA-graph-safe worst case, so the gemm1 and gemm2 grids have no all-padding M
  tiles. The host checks that every expert got exactly tokens x topk / experts rows and raises otherwise; EP routing is
  refused.

## Notes

- `ROCM_PATH`: FlyDSL finds `ld.lld` through it and AITER's JIT finds `hipcc`. Under rocprofv3, a linker from another
  ROCm fails with `lld invocation failed`. `bench.sh` falls back to `/opt/rocm` when it is unset.
- `TMPDIR`: AITER merges its tuned configs into `$TMPDIR/aiter_configs/`. If another user owns `/tmp/aiter_configs`,
  AITER misses the tuned CSV and compiles a slower default gemm1; logs then show `no grouped CSV config matched`.
- Ubuntu's `libamdhip64-dev` (`/usr/include/hip`, HIP 5.7) shadows ROCm's headers in AITER's JIT build. Set
  `HIPCC_COMPILE_FLAGS_APPEND="-isystem $ROCM_PATH/include"`.
- Never import `aiter` with the repository root on `sys.path`: the source tree shadows the installed package.
  `bench.sh` is safe.
