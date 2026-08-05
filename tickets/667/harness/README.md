# SILOTIGER-667 Benchmark Harness

Harness for developing and comparing FlyDSL warp-decode MoE gate_up / down kernels
against the CK C++ reference (commit `62e30c9098`, branch `users/samremes/ck/warp-decode`).

## Directory layout

```
harness/
├── build_ck_bench.sh      — build the CK C++ benchmark binary (one-time setup)
├── bench_flydsl_wd.py     — time + verify FlyDSL kernels
├── compare.py             — join FlyDSL and CK stdout into a Markdown table
├── profile.sh             — rocprof-compute wrapper for either kernel
└── kernels/
    └── wd_gate_up_bf16.py — FlyDSL warp-decode gate_up kernel (Task 0)
```

## Prerequisites

- FlyDSL installed (editable): `pip install -e .` from `FlyDSL/`
- PyTorch with ROCm support: `import torch; torch.cuda.is_available()` → True
- ROCm tools: `rocminfo`, `/opt/rocm/bin/rocprof-compute` (v3.6.0 present)

aiter is **not required** for Task 0 (bf16×bf16 path).

## Quick start

### Step 1 — FlyDSL correctness check (gfx942-safe)

```bash
cd tickets/667/harness
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
  python bench_flydsl_wd.py --shapes deepseek-v3 --batches 1 --verify
```

Expected output:
```
GPU arch: gfx942
  [PASS] deepseek-v3 B=1: max_delta=..., 95.0%+ close (atol=0.5, rtol=0.05)
  deepseek-v3            1  gate_up_bf16x2        ...ms  ...TFLOP/s  ...GB/s
```

### Step 2 — FlyDSL benchmark (all shapes, multiple batch sizes)

```bash
python bench_flydsl_wd.py --shapes deepseek-v3 qwen3next --batches 1 2 4 8 \
    --iters 30 --warmup 5 | tee flydsl.txt
```

### Step 3 — Build CK benchmark (one-time, ~10-20 min)

```bash
bash build_ck_bench.sh
# Binary: /home/AMD/bartgips/code/rocm-libraries-wdec/projects/composablekernel/build-wdec/bin/bench_ck_tile_warp_decode
```

> On gfx942, the CK bench will compile for both gfx942 and gfx950.  The BF16-act×FP8-weight
> kernel (`gate_up_bf16`) will run on gfx942 but uses fp8 weights (different from
> FlyDSL's bf16×bf16 Task 0). A direct apples-to-apples comparison is available on gfx950.

### Step 4 — Run CK benchmark

```bash
CK_BENCH=/home/AMD/bartgips/code/rocm-libraries-wdec/projects/composablekernel/build-wdec/bin/bench_ck_tile_warp_decode
CK_WARP_DECODE_BENCH_SHAPES=deepseek-v3,qwen3next \
CK_WARP_DECODE_BENCH_BATCHES=1,2,4,8 \
CK_WARP_DECODE_BENCH_ITERS=30 \
  $CK_BENCH | tee ck.txt
```

### Step 5 — Compare

```bash
python compare.py --flydsl flydsl.txt --ck ck.txt
```

Prints a Markdown table with FlyDSL ms/GB/s vs CK ms/GB/s and ratio per (shape, B).

### Step 6 — Profile

```bash
# Profile FlyDSL kernel (roofline only by default):
bash profile.sh flydsl --shapes deepseek-v3 --batches 1

# Profile CK kernel:
bash profile.sh ck --shape deepseek-v3 --B 1

# Full counters (no --roof-only):
bash profile.sh flydsl --shapes deepseek-v3 --batches 1 --no-roof
```

After profiling, analyze with:
```bash
rocprof-compute analyze --path profile_out/flydsl_<timestamp>/
```

## Kernel development workflow

1. Edit `kernels/wd_gate_up_bf16.py`
2. Re-run `bench_flydsl_wd.py --verify` to check correctness
3. Re-run `bench_flydsl_wd.py` to measure performance
4. If the JIT cache gets stale: `FLYDSL_RUNTIME_ENABLE_CACHE=0 python bench_flydsl_wd.py ...`

## Model shapes

| name | HIDDEN | INTER | TOPK | E |
|------|--------|-------|------|---|
| deepseek-v3 | 7168 | 2048 | 8 | 256 |
| minimax | 3072 | 1536 | 8 | 256 |
| qwen3next | 2048 | 512 | 10 | 512 |

## Task 0 scope (bf16×bf16, gfx942)

- One wave (64 lanes) per output neuron
- `v_dot2_f32_bf16` via inline asm + `s_nop 2` for dependent chain
- 6-step butterfly XOR shuffle
- No MoE sorting (router_ids = torch.randint)
- No FP8/FP4 conversion (gfx950 only)

## Task 1 scope (bf16×fp8, gfx950)

- Same grid/reduce structure
- Replace bf16 weight load with fp8 weight load + `v_cvt_scalef32_pk_bf16_fp8`
- Switch to independent accumulators + single drain (`s_nop 2`)
- Wire up CK comparison: `gate_up_bf16` / `gate_bf16_d2` kernels

## Task 2 scope (split-K on down, gfx942/950)

- `down_reduce` kernel (not yet implemented)
- `atomicAdd` epilogue with `k_batch` grid axis
- Compare against CK `down_h2_d2` / `down_fp4_h2`
