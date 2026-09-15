# Opt-in gfx950 split kernels

`build_softmax_split` builds a two-pass row Softmax launcher for contiguous,
finite BF16/FP16/FP32 tensors with 1–4 rows and 32768–262144 columns.
Each 4096-element chunk produces FP32 max/sum statistics, then independently
normalizes using the combined statistics. This exposes more workgroups and
reduces the register pressure of a single-workgroup long row.

`build_gemm_bf16_split_fp32` builds a BF16 `A[M,K] @ B[N,K].T` launcher using
AITER's existing FlyDSL FP32-output split-K GEMM plus a FlyDSL conversion kernel.
The default 16x32x128 tile, 4 stages, split-K=4 targets small M. This avoids
repeated BF16 atomic-add rounding, although the underlying CShuffle still uses
the input dtype. It is not a general replacement for the tuned GEMM dispatcher.

```python
from aiter.ops.flydsl import build_softmax_split, build_gemm_bf16_split_fp32
softmax = build_softmax_split(1, 131072, "bf16")
softmax(x, out)
gemm = build_gemm_bf16_split_fp32(32, 384)
gemm(a, b, out_gemm)  # a=[32,7168], b=[384,7168]
```

Allocate launchers before capture/timing. Each owns a reusable workspace; use
separate instances for concurrent streams and keep launchers alive as long as
captured graphs exist. Input/output aliasing, arbitrary strides, and NaN/Inf
semantics are outside the supported contract. Both APIs validate shape, dtype,
device and contiguous layout. Neither changes the existing global dispatch.

## Validation

```bash
pytest -q op_tests/test_flydsl_split_kernels.py
python op_tests/bench_flydsl_split_kernels.py \
  --flydsl-root /path/to/ROCm/FlyDSL --output split-results.json
```

The implementation and correctness tests require only AITER's FlyDSL wheel.
The optional benchmark requires the FlyDSL source checkout for its original
Softmax baseline. Tested with FlyDSL kernel source commit
`728f6b220518af94a8d641a0512ef87167c90125`, compiler wheel 0.3.2,
AITER base `4dad644a9c839c02f8ecdb012eb541de9b80a24b`, and the AITER CI image
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0`.

110 GPU tests pass on MI355X/gfx950, covering long and ragged rows, all three
Softmax dtypes, random/uniform/constant/wide-range inputs, BF16 split-K GEMM
with independent seeds, invalid API contracts, and separate stream workspaces.
Tests poison outputs before graph replay to catch empty captures.

A 15-round randomized paired run (64 complete calls per graph) measured:

| BF16 operation | Stock FlyDSL, us | New implementation, us | Speedup |
|---|---:|---:|---:|
| Softmax [1,131072] | 26.577 | 6.263 | 4.244x |
| Softmax [1,89999] | 15.135 | 6.325 | 2.393x |
| GEMM [M,N,K]=[32,384,7168] | 9.351 | 7.878 | 1.187x |

Both Softmax launches, the GEMM split-K initialization protocol, and output
conversion are included; allocation/compilation/reference are excluded. These
are resident-working-set measurements, not cold-HBM or model-level speedups.
Only one idle GPU was benchmarked; other GPUs had unrelated existing workloads.
Raw samples are in `split_kernels_mi355x.json`. These measurements compare the
original single-workgroup Softmax and HGEMM unit-test configuration; they do
not claim to beat an exhaustive vendor autotuning run or all GEMM shapes.
