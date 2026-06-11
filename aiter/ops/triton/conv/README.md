# conv2d (Triton, AMD ROCm)

> **`Conv2d` for AMD ROCm — a drop-in replacement for `torch.nn.Conv2d`,
> optimized for AMD RDNA GPUs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)](https://pytorch.org)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-ED1C24.svg)](https://www.amd.com/en/developer/resources/rocm-hub.html)
[![Triton](https://img.shields.io/badge/Triton-3.7-orange.svg)](https://github.com/triton-lang/triton)

A hand-written Triton 2-D convolution library optimized for AMD RDNA
GPUs. Five kernel families (1×1, 3×3 cblocked, 3×3 NHWC, Winograd
F(4×4, 3×3), general) behind one shape-driven router and one entry
point. Drop-in for the forward path of `nn.Conv2d`.

---

## Why this op exists

PyTorch on AMD goes through MIOpen, which ships hand-tuned solvers per
architecture, per dtype, per layout. That works well on the combinations
the solvers were specifically tuned for, but every new dtype × layout ×
architecture combination needs its own tuning pass — so coverage is
uneven across the matrix (e.g. on RDNA4 the fp16 path is well-served,
while bf16 falls back to direct/GEMM solvers that are noticeably slower
at large channel counts; most modern checkpoints — LLMs, diffusion VAEs
— ship in bf16).

This op takes the opposite approach: a single set of Triton kernels
that runs **fp16 and bf16 through the same code path**, supports
**both NCHW and NHWC end-to-end** (NHWC inputs run on an NHWC kernel —
no NHWC↔NCHW conversion), and gets reasonable performance across the
full matrix **without per-architecture kernel implementations** (one
set of Triton kernels for every arch, with a thin per-arch JSON config
layer — see Tuning). A shape-driven
router picks between five kernel families (1×1, 3×3 cblocked, 3×3 NHWC,
Winograd F(4×4, 3×3), general) so the right kernel runs per layer
automatically. Some kernels do repack inputs/weights into kernel-local
formats (channel-blocked tiles for cblocked, G/Bᵀ transforms for
Winograd) — these packs are LRU-cached so steady-state cost is
negligible.

---

## Performance

Designed to deliver strong throughput on AMD RDNA4 across both fp16 and
bf16. The bench harness logs which MIOpen solver was selected per layer
and reports aggregate TFLOPS for both backends, so you can verify the
behavior on your stack.

To measure on your stack:

```bash
python -m op_tests.op_benchmarks.triton.bench_conv2d \
    --model <resnet50|"stable-diffusion-3.5-medium"|"FLUX.2-klein-9B"> \
    --dtype <fp16|bf16> \
    [--miopen-solvers]   # opt-in; ~60-120s upfront subprocess
```

The bench harness produces three box-drawn tables: LAYER-BY-LAYER (per-layer
Triton vs MIOpen TFLOPS, Triton kernel name, optionally MIOpen solver,
kernel+repack column for shapes that prepack), MIOpen SOLVER SUMMARY (only
with `--miopen-solvers`), and OVERALL PERFORMANCE (mean/median/aggregate
TFLOPS, total time, layer wins, correctness).

> **Note on TFLOPS**: numbers are *direct-convolution-equivalent* throughput
> (the standard convention used by cuDNN, MIOpen, and the Winograd
> literature), applied identically to both backends. Winograd kernels —
> Triton's F(4×4, 3×3) and MIOpen's F(2×2, 3×3) / Fury alike — execute
> fewer literal hardware MACs than this denominator counts (≈4× fewer for
> F(4,3), ≈2.25× for F(2,3)). The comparison is apples-to-apples.

---

## Quick start

### Use the function directly

```python
import torch
from aiter.ops.triton.conv.conv2d import conv2d

x = torch.randn(4, 256, 56, 56, device="cuda", dtype=torch.float16)
w = torch.randn(512, 256, 3, 3, device="cuda", dtype=torch.float16)

y = conv2d(
    x, w, bias=None,
    stride=(1, 1), padding=(1, 1), dilation=(1, 1),
    activation="relu",          # "none" | "relu" | "relu6" | "gelu"
    layout="nchw",              # "nchw" or "nhwc"
)
```

A shape-driven router picks one of five kernel families:

| Family | When it runs |
|---|---|
| 1×1 GEMM | `R==1, S==1` |
| 3×3 cblocked (NCHW) | 3×3, channel-blocked input for coalesced loads |
| 3×3 NHWC | 3×3 with channels-last input — no input repack |
| Winograd F(4×4, 3×3) | 3×3, stride=1, dilation=1, `C ≥ 512`, `K ≥ 512`, enough output tiles |
| General | anything not 1×1 or 3×3 (5×5, 7×7, dilated, strided) |

### Use as `nn.Conv2d` drop-in

The kernel families above are functional; wrapping them in an `nn.Module`
(walk a model, swap each `nn.Conv2d` for a Triton-backed module that
calls `conv2d(...)` in its `forward`) works as expected and produces
images visually indistinguishable from the PyTorch / MIOpen reference.

Pixel-level agreement on FLUX.2-klein-9B (50 diffusion steps, same prompt
and seed under both backends, only VAE convs swapped to Triton): max diff
**6 / 255**, mean diff **0.17 / 255**.

---

## Constraints

- `groups` must equal 1 (depthwise / grouped not yet implemented).
- `padding_mode` must be `"zeros"`. The pad *amount* (`padding=`, e.g.
  `(1, 1)` or asymmetric `(0, 2)`) is unrestricted; only the pad *value*
  is — `"reflect"`, `"replicate"`, and `"circular"` fall back to PyTorch /
  MIOpen.
- Inputs must be `fp16` or `bf16`.
- Forward only (no backward / training).

---

## Reproducing the tests and benchmarks

Run from the AITER repo root (`/app/aiter` in this tree, or `PYTHONPATH=/app/aiter`).

### Correctness (CI-collected; skipped on unsupported archs)

```bash
pytest op_tests/triton_tests/conv/                                # full matrix, 74 tests
pytest op_tests/triton_tests/conv/ -k "no_bias and fp16_nchw"     # subset
pytest op_tests/triton_tests/conv/ -k "test_edge"                 # one test family
```

Tests are parametrized over `(dtype, layout, method)`. Every kernel in
`_helpers.ORDERED_METHODS` is exercised against fp16 and bf16 on NCHW.
NHWC is single-dispatch (only `conv2d_nhwc`), so each NHWC test runs once
per dtype.

### Benchmark

Three modes, all in `bench_conv2d.py`.

**Single shape** (one parseable result line — for ad-hoc measurements):

```bash
python -m op_tests.op_benchmarks.triton.bench_conv2d \
    --N 1 --C 64 --H 56 --W 56 --K 64 --R 3 --S 3 --pad-h 1 --pad-w 1
```

**Real-model sweep** (default — uses ResNet50 if no `--model` given):

```bash
python -m op_tests.op_benchmarks.triton.bench_conv2d --dtype fp16              # default = resnet50
python -m op_tests.op_benchmarks.triton.bench_conv2d --model resnet50
python -m op_tests.op_benchmarks.triton.bench_conv2d --model "FLUX.2-klein-9B" --miopen-solvers
```

**Edge-case smoke sweep** (degenerate paths: `C=1`, dilation>1, asymmetric dims —
NOT representative of production):

```bash
python -m op_tests.op_benchmarks.triton.bench_conv2d --dtype fp16 --smoke
```

Cross-axis flags:

```
--dtype {fp16,bf16}                           # default fp16
--layout {nchw,nhwc}                          # default nchw
--method {auto,default,cblocked,nhwc,winograd_f4x3,winograd_f4x3_fused,winograd_f4x3_cblocked}
--metric {time,throughput}                    # default throughput
--no-bias                                     # bench the bias=None code path
--miopen-solvers                              # detect MIOpen solver names (sweep mode; ~60-120s subprocess)
--show-kernel-name                            # include routed kernel name in single-shape output
```

Real-model shapes are pre-extracted (no torchvision/diffusers needed at
bench time). To add a new model, run `extract_conv_shapes.py` once and
merge its JSON output into `model_shapes.json`:

```bash
python -m op_tests.op_benchmarks.triton.model_benchmarking_tool.extract_conv_shapes \
    --model resnet50    # or sd35_vae / flux2_vae with --model-path
```

Tested on ROCm 7.2 / PyTorch `2.9.1+gitff65f5b` / Triton 3.7 (commit `23f4e522d`).

### Tuning

Per-kernel configs ship as JSON under `aiter/ops/triton/configs/conv/`, one
file per `(arch, kernel)` — e.g. `gfx1201-CONV-3X3-NHWC.json`. The loader walks
three tiers: literal shape pin → `M_LEQ_x` bucket → `"any"` fallback. No
runtime autotune in the hot path, so CI compile time stays predictable and
the first call hits no tuning tax.

Tuned for RDNA4 today (configs ship as `gfx1201-*.json` and `gfx1200-*.json`).

If you need to retune at runtime (e.g. while developing a new kernel), set
`AITER_TRITON_CONV_AUTOTUNE=1` — this restores the original `@triton.autotune`
behaviour across the `AUTOTUNE_*_CONFIGS` lists defined alongside each kernel
in `_triton_kernels/conv/conv_*.py` for the current process.

---

## Documentation

- **[`DESIGN.md`](DESIGN.md)** — architecture, per-kernel deep-dive, full
  Winograd F(4,3) derivation (G/Bᵀ/Aᵀ matrices, 361× amplification
  analysis, why Winograd is disabled for `C < 4`), the
  `_select_3x3_method` heuristic, memory layouts and repacking,
  numerical model, extension guide.

---

## Repository layout

```
aiter/ops/triton/conv/                Kernel library
  conv2d.py                           Public API + smart routing
  _launch.py                          Grid setup + _select_3x3_method
  _prepack.py                         Weight/input repack caches (LRU)
  _utils.py                           Shape math, tolerance model
  README.md, DESIGN.md

aiter/ops/triton/_triton_kernels/conv/   @triton.jit kernels
  (1x1, 3x3 cblocked, 3x3 NHWC, general, 5 Winograd kernels)

op_tests/triton_tests/conv/           Pytest unit tests (CI-collected; skipped on unsupported archs)
  test_conv2d.py                      The only collected test file
  _helpers.py                         TestSuite, registry, shape generators

op_tests/op_benchmarks/triton/
  bench_conv2d.py                     Self-contained bench tool (single + sweep)
  model_benchmarking_tool/
    extract_conv_shapes.py            One-time offline shape extraction
    model_shapes.json                 Pre-extracted conv shapes (resnet50, SD3.5, FLUX2)
```
