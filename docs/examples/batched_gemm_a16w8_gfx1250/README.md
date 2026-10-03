# batched_gemm_a16w8 (gfx1250) experiment suite

This is the reproduction kit for
[`../../batched_gemm_a16w8_gfx1250_roofline.md`](../../batched_gemm_a16w8_gfx1250_roofline.md).

## Prerequisites

- **A checkout that contains the kernel.** The suite needs
  `aiter/ops/triton/gemm/batched/batched_gemm_a16w8.py` and its configs, which
  live on `satya/temp_delete`, not `main`. Point `AITER_ROOT` at that
  checkout, and run from a copy of this directory.
- **gfx1250 GPUs.**

## One GPU process per GPU

Every GPU command goes through `./gpurun.sh <gpu> <cmd...>`. It holds a
per-GPU `flock` (directory set by `AITER_GPU_LOCK_DIR`) for the whole command
and sets `HIP_VISIBLE_DEVICES`. Never start GPU work any other way: two
concurrent jobs on one GPU can take the machine down.

## Run

```bash
export AITER_ROOT=/path/to/aiter-with-kernel
HERE=$PWD   # this directory; gpurun.sh cds to $AITER_ROOT, so use absolute paths
OUT=/tmp/a16w8_runs && mkdir -p $OUT

# Replicated decomposition: the same set on every GPU, one process at a time per GPU
for g in 0 1 2 3; do ./gpurun.sh $g bash $HERE/run_replicated.sh > $OUT/rep$g.log 2>&1 & done; wait

# Shader clock on one GPU (anchors the cycle normalisation), then the table
bash ./build_shader_clock.sh && ./gpurun.sh 0 $HERE/shader_clock
python3 analyze_replicated.py --logs $OUT --ref-gpu 0 --ref-mhz <measured MHz>
```

Other experiments:

| Command | What it measures |
|---|---|
| `./gpurun.sh 0 python3 $HERE/exp_bench.py --check --dobench` | Tuned-config baseline, plus wrapper `do_bench` |
| `./gpurun.sh 0 python3 $HERE/exp_bench.py --module $HERE/batched_gemm_a16w8_variant.py --configs-file $HERE/configs/tiles.jsonl --check` | A config or variant sweep (keys prefixed `X_` are variant constexprs) |
| `... --fullcheck 5` | Bitwise comparison of the variant's whole output against the unmodified kernel |
| `./gpurun.sh 0 python3 $HERE/wmma_peak.py` | Raw bf16/fp8 WMMA throughput |
| `./gpurun.sh 0 python3 $HERE/host_overhead.py` | Host-side cost per wrapper call, by stage |
| `./gpurun.sh 0 python3 $HERE/bandwidth_calibration.py` | Plain `zero_`/`copy_` bandwidth |

To reproduce the committed table from the committed data:
`python3 analyze_replicated.py --logs results --ref-mhz 65`.

## Files

| File | Purpose |
|---|---|
| `gpurun.sh` | Per-GPU lock runner |
| `exp_bench.py` | Launches the kernel (repo or variant) with config overrides. Times it with CUDA-graph replay over cold input copies, the same method as the tuning harness |
| `batched_gemm_a16w8_variant.py` | Kernel copy with ablation knobs (`ABL_LOAD`, `ABL_MMA`, `ABL_STORE`) and fix knobs (`EPI_WAIT`, `EPI_N`, `C_BUFS`) |
| `configs/*.jsonl` | Experiment sets: one JSON object per line, merged over the tuned config |
| `run_replicated.sh`, `analyze_replicated.py` | Per-GPU replicated set and its clock-normalised summary |
| `wmma_peak.py`, `shader_clock.cpp`, `build_shader_clock.sh` | WMMA throughput and true shader clock |
| `host_overhead.py`, `bandwidth_calibration.py`, `sustain_load.py` | Host cost, bandwidth, and a sustained load for sampling clocks |
| `results/` | Raw `RESULT` records from the runs, the machine state, and an ISA summary of the baseline kernel |
