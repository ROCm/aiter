# SonicMoE on AITER Gap Plan

This branch keeps the current SonicMoE-compatible Python API and uses AITER's
existing `fused_moe` implementation as the ROCm backend. The next useful work is
to measure where that backend differs from SonicMoE's paper implementation before
starting a new CK, Opus, FlyDSL, or ASM kernel.

## Phase 1: Stage Breakdown

Use `op_tests/op_benchmarks/bench_sonic_moe_stage_breakdown.py`.

It times:

- router linear
- top-k softmax
- route preparation wall time
- `moe_sorting`
- selected stage1 kernel
- selected stage2 kernel
- direct `aiter.fused_moe`
- A2 inter-stage read/write size
- bottleneck name and share

Example:

```bash
python op_tests/op_benchmarks/bench_sonic_moe_stage_breakdown.py \
  --shape 32768,4096,1024,128,8 \
  --routing topk \
  --warmup 3 \
  --iters 10
```

## Phase 2: Sweep and Tuned Config Candidates

Use `op_tests/op_benchmarks/bench_sonic_moe_stage_sweep.py`.

It runs the stage breakdown in fresh child processes across routing and AITER
knobs, then writes:

- normalized benchmark CSV
- JSONL with full `result_json`
- optional `tuned_fmoe.csv`-shaped candidate rows

Example:

```bash
python op_tests/op_benchmarks/bench_sonic_moe_stage_sweep.py \
  --shape 32768,4096,1024,128,8 \
  --routing topk,rounded,balanced \
  --block-m auto,32,64,128 \
  --dispatch-policy 0 \
  --warmup 3 \
  --iters 10 \
  --csv /tmp/sonic_moe_stage_sweep.csv \
  --jsonl /tmp/sonic_moe_stage_sweep.jsonl \
  --tuned-csv /tmp/sonic_moe_tuned_candidates.csv
```

The generated tuned rows are tagged as `sonic_moe_stage_sweep` by default, so
AITER ignores them if they are appended to `tuned_fmoe.csv`. Pass
`--active-tuned-rows` only after validating correctness and stability.

## Phase 3: Paper Alignment

Use `op_tests/op_benchmarks/bench_sonic_moe_compare_paper.py`.

It compares AITER sweep outputs with headline SonicMoE paper targets:

- 45% activation memory reduction
- 1.86x H100 compute throughput improvement vs ScatterMoE BF16
- 1.16x token-rounding main-kernel speedup
- 213B tokens/day on 64 H100 vs 225B tokens/day on 96 H100
- 25% forward and 15% backward speedups on B300 vs DeepGEMM

Example:

```bash
python op_tests/op_benchmarks/bench_sonic_moe_compare_paper.py \
  /tmp/sonic_moe_stage_sweep.csv \
  --output /tmp/sonic_moe_paper_gap.md
```

For exact shape-by-shape MI355X vs H100 comparison, supply a CSV with columns
`shape_str,h100_sonic_ms,h100_sonic_tflops` via `--paper-h100-csv`.

## Phase 4: Kernel Decision

Do not start by rewriting a full MoE kernel. Use the breakdown result first:

- If `moe_sorting_ms` dominates, tune sorting dispatch, Opus/CK sorting choice,
  routing layout, and token rounding before changing GEMM.
- If rounded routing gives little gain and tile efficiency is already high,
  AITER is not primarily padding-bound on that shape.
- If stage1/stage2 dominate and `a2_read_write_mib` is large, the real SonicMoE
  gap is the dataflow: gather and inter-stage traffic should be fused or avoided.
- If existing CK/FlyDSL/ASM stage kernels are close and the remaining gap is
  sorting or launch overhead, new Opus grouped GEMM is not justified yet.
- If A2 traffic plus stage kernels remain the bottleneck after sweep/tuning, then
  a new kernel family is justified. The best target is a fused grouped MoE dataflow
  that loads from original token positions and avoids separately materializing
  gather/A2 traffic where possible.

Current practical priority:

1. Run the sweep on MI355X for the target shapes.
2. Check whether bottleneck is sorting, stage1, stage2, or A2 traffic.
3. Promote only verified tuned rows.
4. Decide between CK/FlyDSL/ASM extension and a new Opus grouped MoE only after
   the bottleneck is stable across shapes.

## Phase 5: ROCprof and Dataflow Prototype

Use `op_tests/op_benchmarks/run_sonic_moe_rocprof.py` to profile the current
best rows:

```bash
python op_tests/op_benchmarks/run_sonic_moe_rocprof.py \
  --output-root /app/yifehuan_temp/data/sonic_moe_mi355_latest/rocprof \
  --sweep-csv /app/yifehuan_temp/data/sonic_moe_mi355_latest/sonic_moe_mi355_stage_sweep.csv
```

The runner stores raw rocprofv3 CSVs, command logs, status, and generated
`rocprof_stage_summary.csv/md` under the selected output root. The current
MI355X pass is summarized in `docs/sonic_moe_rocprof_dataflow_plan.md`.

## Phase 6: A2 Layout Prototype

Use `--a2-layout current|sorted` in the stage breakdown or sweep scripts:

```bash
python op_tests/op_benchmarks/bench_sonic_moe_stage_sweep.py \
  --shape 32768,4096,512,128,8 \
  --shape 32768,4096,1024,128,8 \
  --shape 32768,4096,1024,256,8 \
  --a2-layout current,sorted \
  --routing balanced \
  --block-m 128 \
  --dispatch-policy 0 \
  --check-correctness
```

The sorted path is a correctness/dataflow probe: current CK stage1 writes
normal A2, the benchmark packs it into sorted row order, then a standalone
Triton fp32-atomic stage2 consumes sorted A2. It is intentionally not the final
kernel. The current MI355X result is summarized in
`docs/sonic_moe_a2_layout_prototype.md`.
