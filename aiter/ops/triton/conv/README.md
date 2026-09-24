# Conv2D and Conv3D (Triton, AMD ROCm)

> **`Conv2d` for AMD ROCm — a drop-in replacement for `torch.nn.Conv2d`,
> optimized for AMD RDNA GPUs.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-ee4c2c.svg)](https://pytorch.org)
[![ROCm](https://img.shields.io/badge/ROCm-7.2-ED1C24.svg)](https://www.amd.com/en/developer/resources/rocm-hub.html)
[![Triton](https://img.shields.io/badge/Triton-3.7-orange.svg)](https://github.com/triton-lang/triton)

A hand-written Triton 2-D convolution library optimized for AMD RDNA
GPUs. Six kernel families (1×1, direct NCHW 3×3, 3×3 cblocked, 3×3
NHWC, Winograd F(4×4, 3×3), general) behind one shape-driven router and one entry
point. Drop-in for the forward path of `nn.Conv2d`.

The package also provides Conv3D for video-VAE workloads, organized into public
routing, launch, prepack, utility, and kernel layers. Dimension-specific
functions retain explicit names, and Conv3D uses independent packing caches and
routing logic.

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
**both NCHW and NHWC end-to-end** (channels-last inputs stay channels-last;
an ordinary contiguous input requested with `layout="nhwc"` is converted once),
and gets reasonable performance across the
full matrix **without per-architecture kernel implementations** (one
set of Triton kernels for every arch, with a thin per-arch JSON config
layer). A shape-driven
router picks between six kernel families so the right kernel runs per layer
automatically. Weight transforms are LRU-cached. Tuned NCHW 3×3 activations
can be read directly; other eligible shapes use one fused Triton NCHWc pack.

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

### Conv3D

```python
import torch
from aiter.ops.triton.conv.conv3d import conv3d

x = torch.randn(1, 384, 3, 46, 51, device="cuda", dtype=torch.float16)
w = torch.randn(384, 384, 3, 3, 3, device="cuda", dtype=torch.float16)

y = conv3d(
    x, w, bias=None,
    stride=1, padding=1, dilation=1,
    activation="none",
    layout="ncdhw",           # "ncdhw" or "ndhwc"
)
```

The Conv3D router selects among specialized 1×1×1, NCDHWc 3×3×3, NDHWC
3×3×3, general im2col-free, and two 2.5-D Winograd input paths. The Winograd
paths transform only H/W with F(4×4,3×3) and perform the three depth taps
directly; one reads NCDHW and the other first packs input to NCDHWc. This avoids
the numerical amplification and temporary size of full 3-D Winograd. Weight
packs and filter transforms are version-aware LRU cached.

### Conv2D routing

A shape-driven router picks one of six kernel families:

| Family | When it runs |
|---|---|
| 1×1 GEMM | `R==1, S==1` with unit dilation |
| Direct NCHW 3×3 | Tuned NCHW shapes selected by a crossover or exact-route table; no input repack |
| 3×3 cblocked (NCHW) | Eligible NCHW shapes not routed direct; uses a fused NCHWc pack |
| 3×3 NHWC | 3×3 with channels-last input — no input repack |
| Winograd F(4×4, 3×3) | Eligible non-wave32 targets where transforms amortize |
| General | All other kernels, plus narrow-channel NCHW 3×3 |

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

- Only `groups=1` is implemented; callers must send depthwise or grouped
  convolutions to another backend.
- Only zero padding is implemented. Per-axis pad amounts may differ, for
  example `(0, 2)` for Conv2D or `(0, 1, 2)` for Conv3D, but callers must send
  `"reflect"`, `"replicate"`, and `"circular"` padding modes to PyTorch/MIOpen.
- Inputs must be `fp16` or `bf16`.
- Forward only (no backward / training).

---

## Reproducing the tests and benchmarks

Run these commands from the AITER repository root.

### Tests

Numerical Conv2D and Conv3D tests are CI-collected and skip at module level on
unsupported GPU architectures. The Conv3D configuration tests do not launch
kernels.

```bash
python -m pytest op_tests/triton_tests/conv/                       # all Conv1D/2D/3D tests
python -m pytest op_tests/triton_tests/conv/test_conv2d.py         # Conv2D tests
python -m pytest op_tests/triton_tests/conv/test_conv2d.py \
  -k "no_bias and fp16_nchw"                                      # no-bias FP16 NCHW tests
python -m pytest op_tests/triton_tests/conv/test_conv2d.py \
  -k "test_edge"                                                  # edge-case tests
```

The Conv2D tests are parametrized over `(dtype, layout, method)`. Every kernel in
`_helpers.ORDERED_METHODS` is exercised against fp16 and bf16 on NCHW.
NHWC is single-dispatch (only `conv2d_nhwc`), so each NHWC test runs once
per dtype.

The complete Conv3D suite contains numerical, routing, prepack, cache,
validation, and installed-configuration tests. Run it with:

```bash
python -m pytest op_tests/triton_tests/conv/test_conv3d.py
```

Benchmark execution and its command-line interface are provided separately by
`op_tests/op_benchmarks/triton/bench_conv3d.py`.

### Benchmark

The Conv2D benchmark has three modes, all in `bench_conv2d.py`.

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

The Conv3D benchmark uses the same frozen-shape-database approach as Conv2D.
It defaults to the Wan2.2 A14B-style VAE workload and accepts a
case-insensitive model-name substring with `--model`:

```bash
python -m op_tests.op_benchmarks.triton.bench_conv3d                         # Wan2.2 A14B-style
python -m op_tests.op_benchmarks.triton.bench_conv3d --model ltx25
python -m op_tests.op_benchmarks.triton.bench_conv3d --model vision-patch
python -m op_tests.op_benchmarks.triton.bench_conv3d --model ti2v
```

The available Conv3D databases are `wan22-a14b-vae`, `ltx25-conv-vae`,
`vision-patch-embed`, and `wan22-ti2v-5b-vae`. VAE sweeps report per-shape
results, mean/median/aggregate throughput, layer winners, and call-count-weighted
encoder/decoder totals when the trace provides those counts. Pass
`--miopen-solvers` to add MIOpen solver names and a solver summary. The benchmark
always calls the production `conv3d` router unless `--method` explicitly forces
another supported path. `--batch-size N` overrides the batch dimension across a
model sweep.

The LTX trace uses the documented 121-frame 544×960 workload (latent
`[1,128,16,17,30]`). The Wan TI2V trace uses 121 frames at 704×1280 (latent
`[1,48,31,44,80]`) and includes both first-chunk and steady-state causal-cache
calls. `vision-patch-embed` covers InternVideo2 and Qwen-family tubelet/patch
projection shapes.

As with Conv2D, `--smoke` selects a compact edge-case sweep instead of a model:

```bash
python -m op_tests.op_benchmarks.triton.bench_conv3d --smoke
python -m op_tests.op_benchmarks.triton.bench_conv3d --model ltx25 --miopen-solvers
python -m op_tests.op_benchmarks.triton.bench_conv3d --model ti2v --batch-size 2
```

For a single shape, provide all nine dimension flags. Depth, height, and width
parameters are independent, matching the Conv2D single-shape interface with the
additional depth axis:

```bash
python -m op_tests.op_benchmarks.triton.bench_conv3d \
  --N 1 --C 64 --D 4 --H 16 --W 16 --K 128 --T 1 --R 1 --S 1 \
  --stride-d 1 --stride-h 1 --stride-w 1 \
  --pad-d 0 --pad-h 0 --pad-w 0 \
  --dilation-d 1 --dilation-h 1 --dilation-w 1
```

Conv2D cross-axis flags:

```
--dtype {fp16,bf16}                           # default fp16
--layout {nchw,nhwc}                          # default nchw
--method {auto,default,cblocked,nhwc,winograd_f4x3,winograd_f4x3_cblocked}
--metric {time,throughput}                    # default throughput
--no-bias                                     # bench the bias=None code path
--miopen-solvers                              # detect MIOpen solver names (sweep mode; ~60-120s subprocess)
--show-kernel-name                            # include routed kernel name in single-shape output
--model MODEL                                 # select a model workload by name substring
--smoke                                       # use the edge-case sweep
--batch-size N                                # override N for every swept shape
```

Conv3D cross-axis flags:

```
--dtype {fp16,bf16}                           # default fp16
--layout {ncdhw,ndhwc}                        # default ncdhw
--method {auto,general,1x1x1,cblocked,ndhwc_3x3x3,winograd,winograd_cblocked}
--metric {time,throughput}                    # default throughput
--activation {none,relu,relu6,gelu}           # default none
--no-bias                                     # bench the bias=None code path
--miopen-solvers                              # detect MIOpen solver names (sweep mode)
--show-kernel-name                            # include routed kernel name in single-shape output
--model MODEL                                 # select a model workload by name substring
--smoke                                       # use the edge-case sweep
--batch-size N                                # override N for every swept shape
```

---

## Repository layout

Files relevant to the Conv2D and Conv3D implementation are organized as
follows:

```
aiter/ops/triton/conv/                Kernel library
  conv2d.py                           Public API + smart routing
  _launch.py                          Shared 2-D/3-D launches, grids + configs
  _prepack.py                         Shared pack transforms + separate caches
  _utils.py                           Shared constants and dimension-named helpers
  conv3d.py                           Conv3D public API + router
  README.md

aiter/ops/triton/_triton_kernels/conv/   @triton.jit kernels
  conv_1x1.py                            Conv2D 1x1 kernel
  conv_3x3.py                            Conv2D direct/cblocked/NHWC 3x3 kernels
  conv_general.py                        General Conv2D kernel
  conv_3x3_winograd_f4x3.py              Conv2D Winograd transforms and GEMM
  ncx_to_cblocked.py                     Shared NCHW/NCDHW blocked-layout kernel
  conv3d_1x1x1.py                        Conv3D 1x1x1 kernel
  conv3d_3x3x3.py                        Conv3D cblocked/NDHWC 3x3x3 kernels
  conv3d_general.py                      General Conv3D kernel
  conv3d_winograd_hw_f4x3.py             Conv3D H/W Winograd transforms and GEMM

aiter/ops/triton/configs/<arch>/triton/conv/
  <kernel>/DEFAULT.json                  Per-architecture launch parameters

op_tests/triton_tests/conv/           Pytest correctness and integration tests
  test_conv2d.py                      Conv2D correctness and routing
  test_conv3d.py                      Conv3D correctness, routing, and config
  _helpers.py                         TestSuite, registry, shape generators

op_tests/op_benchmarks/triton/
  bench_conv2d.py                     Self-contained bench tool (single + sweep)
  bench_conv3d.py                     Self-contained Conv3D bench tool (single + sweep)
  conv_shapes.json                    Pre-extracted Conv2D and Conv3D model shapes
```
