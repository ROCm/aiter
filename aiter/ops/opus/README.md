# OPUS GEMM Python interface

OPUS exposes one public GEMM function. The caller resolves the final kernel id
before the call; OPUS does not select a kernel from the shape.

```python
from aiter.ops.opus import opus_gemm

opus_gemm(
    XQ,
    WQ,
    Y,
    *,
    kid,
    layout="plain",
    x_scale=None,
    w_scale=None,
    bias=None,
    split_k=0,
    workspace=None,
)
```

`kid` is mandatory. `Y` is also caller-owned and is returned unchanged after
the launch. The disjoint architecture id bands and the merged `kernels_list`
already existed in the pre-Task1 baseline. The public entry directly calls
`kernels_list.get(kid)`; this refactor does not introduce or renumber those
ids. The returned instance tag plus the dtype/layout arguments determine the
private family adapter, rather than a second selector or a numeric-range guess.

## Dispatch model

```text
caller-resolved final kid
  -> existing kernels_list.get(kid)
  -> instance arch/tag metadata
  -> family-local dtype/layout/scale checks
  -> private family adapter
  -> exact-kid C++ family table
```

There is no tuned-CSV lookup, architecture heuristic, redirect, or framework
fallback inside this exact public path.  The existing high-level A16 caller
resolves those policies before entering it:

```text
explicit kid -> exact public launch
otherwise: tuned CSV -> per-arch private heuristic -> PyTorch fallback
                              |
                              +-> final kid -> exact public launch
```

The three A16 heuristics are private functions in the existing
`gemm_op_a16w16.py`; no selector or per-architecture Python modules are added.
Invalid tuned `(kid, split-K)` pairs are discarded together.  Legacy gfx942
requested-to-actual resolution also happens in the caller, so only the final
integer id reaches `opus_gemm`.

## Current families

| Registry family | Current route | Python input rules |
|---|---|---|
| `a16w16` | gfx942, gfx950, gfx1250 | BF16 `XQ/WQ`, BF16 or FP32 `Y`, plain WQ, optional bias/split-K/Torch workspace |
| `a8w8` | gfx950 kid 2 | FP8 `XQ/WQ`, FP32 `Y`, plain WQ, no scales |
| `a8w8_blockscale` | gfx950 kid 1 | FP8 `XQ/WQ`, FP32 `Y`, plain WQ, two FP32 scales |
| `a8w8_blockscale_bpreshuffle` | gfx942 kid 11000 | FP8 `XQ/WQ`, BF16 `Y`, pre-shuffled WQ, two FP32 scales |
| `a8w8_mxscale_bmm` | gfx950 global kids 8000--8653 (45 registered ids) | FP8 `[M,G,K]` / `[G,N,K]`, E8M0 scales, BF16 or FP32 `[M,G,N]`, optional split-K Torch workspace |

Empty family tables on another architecture are valid capability states. A
kid registered for another architecture is rejected before family launch.

## Examples

### A16W16

```python
import torch
from aiter.ops.opus import opus_gemm

XQ = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
WQ = torch.randn((1, 64, 512), device="cuda", dtype=torch.bfloat16)
Y = torch.empty((1, 64, 64), device="cuda", dtype=torch.bfloat16)

# The caller/tuner has already chosen gfx950 kid 200 and split_k 2.
opus_gemm(XQ, WQ, Y, kid=200, split_k=2)
```

The A16 interface takes 3D `[batch,M,K]`, `[batch,N,K]`, and
`[batch,M,N]` tensors. Inputs are K-contiguous with a dense batch stride; `Y`
is contiguous. Current gfx1250 kernels require batch one. Exact instances can
impose additional tile, output dtype, bias, or K-loop constraints.

### gfx950 A8W8 without scales

```python
Y = torch.empty((batch, M, N), device=XQ.device, dtype=torch.float32)
opus_gemm(XQ, WQ, Y, kid=2)
```

### gfx950 A8W8 blockscale

```python
opus_gemm(
    XQ,
    WQ,
    Y,
    kid=1,
    x_scale=x_scale,
    w_scale=w_scale,
)
```

The group contract is 1x128x128. Scales are contiguous FP32 tensors on the
input device. Their 3D shapes are `[batch,M,K/128]` and
`[batch,N/128,K/128]`; batch-one 2D forms are also accepted by the generated
launcher.

### gfx950 MXFP8 BMM

```python
Y = torch.empty((M, G, N), device=XQ.device, dtype=torch.bfloat16)
opus_gemm(
    XQ,                    # [M,G,K], K-contiguous
    WQ,                    # [G,N,K], K-contiguous
    Y,
    kid=8311,              # exact global id; upstream PR #4320 id 311
    layout="mxscale_bmm",
    x_scale=x_scale,       # [M,G,K/128], one-byte E8M0
    w_scale=w_scale,       # [G,N/128,K/128], one-byte E8M0
    split_k=1,
)
```

The 45 PR #4320 kernels use global ids `8000 + upstream_kid`; the low digits
remain recognizable while the ids share the canonical `kernels_list` without
colliding with existing GEMM ids. `mxfp8_bmm` and `bmm_mxscale` are accepted
layout aliases. The high-level tuned caller remains in the existing A8W8
module as `aiter.batched_gemm_a8w8_mxscale`; it resolves the final id and then
calls this same public entry.

### gfx942 blockscale bpreshuffle

```python
from aiter.ops.shuffle import shuffle_weight

WQ_shuffled = shuffle_weight(WQ, layout=(16, 16))
opus_gemm(
    XQ,
    WQ_shuffled,
    Y,
    kid=11000,
    layout="bpreshuffle",
    x_scale=x_scale,
    w_scale=w_scale,
)
```

`layout="bpreshuffle"` is a declaration of WQ content, not something Tensor
shape or strides can prove. Kid 11000 requires batch one, exact 128-wide N/K
tiles, BF16 output, and its registered scale storage contracts.

## A16 Torch workspace

Workspace ownership is call-scoped and remains in Torch:

```text
validate exact kid and split_k
  -> derive immutable workspace plan from the exact instance
  -> reuse caller workspace or torch.empty for this call
  -> existing pybind wrapper owns normal lazy JIT build
  -> OPUS-local adapter reuses that module's C ABI on the live stream
```

There is no process-global Tensor, pointer registry, HIP allocator, or prewarm
API. The bounded public-contract and A16 launch-plan caches store only registry
metadata, integers, dtypes, option-presence flags and shapes; they never retain
Tensor objects, data pointers, devices, streams or workspaces.

Let `padded_M=ceil_div(M,B_M)*B_M` and
`padded_N=ceil_div(N,B_N)*B_N`:

| Architecture/family | Workspace shape | Instance storage |
|---|---|---|
| gfx950 two-stage | `[allocation_split_k,batch,padded_M,padded_N]` | FP32 |
| gfx942 two-stage | `[allocation_split_k,batch,padded_M,padded_N]` | exact BF16/FP32 dtype |
| gfx1250 two-stage | `[allocation_split_k,padded_M,padded_N]` | BF16 |
| gfx1250 fused | `[tiles_m,tiles_n,fuse_split_k-1,B_M,B_N]` | exact BF16/FP32 dtype |

An explicit workspace must be on the XQ device, contiguous, 16-byte aligned,
of the exact instance dtype, and large enough for the final split. A direct
kid requires `workspace=None`.

gfx942 BF16-workspace kids `10210`, `10213`, and `10216` are exact ids. Their
registered exact-N contract is `{64,128,256,384,512,1024,2048}`. A different N
is rejected; the call is never redirected to an FP32-workspace kid.

## MXFP8 BMM Torch workspace

MXFP8 BMM follows the same ownership rule: Python either uses the caller's
contiguous FP32 Tensor or creates a call-scoped `torch.empty`; C++ receives a
direct pointer and never retains it. Two-stage split-K uses
`split_k * G * padded_M * padded_N` FP32 elements. The fused family stores its
partials and aligned tile counters in one FP32 Tensor. `split_k == 1` and
families that do not consume workspace reject a supplied Tensor.

## Graphs and streams

Automatic `torch.empty` during graph capture uses the graph-private pool.
Concurrent eager calls own independent workspace Tensors. The private C ABI
switches to the XQ device and live PyTorch stream for the call, restores the
previous state, and carries C++ errors through a thread-local status bridge.
The pybind raw remains available privately for the normal lazy JIT build and
A/B measurement.  The mixed-module ctypes adapter lives entirely in
`gemm_op_a16w16.py`; generic `aiter/jit/core.py` is unchanged from the
Task1 baseline.

## Build-time subset compile

Tuned CSV and the compiled-kids sidecar are build inputs only. Their valid
non-BMM OPUS ids are unioned with:

- `DEFAULT_COMPILED_KIDS_BY_ARCH`, the exact-id compile floor containing every
  A16 caller-side heuristic result;
- mandatory A8 ids (`gfx950: {1,2}`, `gfx942: {11000}`).

This controls which launchers enter a subset `.so`.  The high-level A16 caller
may read a tuned row at runtime, but the public/C++ path receives only its
resolved id. Calling a known non-BMM registry kid that was omitted from a
subset build produces an uncompiled-id error. A gfx950 build emits all 45
MXFP8 BMM routes as one deduplicated family so every registered BMM id remains
exact-routable.

## Migration

The former shape-driven and family-specific Python APIs were removed. Migrate
by allocating `Y`, resolving the final id in the caller, and calling the one
entry above. The private family modules intentionally export an empty
`__all__`.

## Source map

| Path | Role |
|---|---|
| `__init__.py` | one public exact-kid router and logical-contract checks |
| `_arch.py` | per-explicit-device architecture/CU scalar cache |
| `gemm_op_a16w16.py` | private exact A16 resolver, scalar plan cache, Torch workspace and OPUS-local C ABI adapter |
| `gemm_op_a8w8.py` | four private exact A8 family adapters, including MXFP8 BMM workspace planning |
| `../batched_gemm_op_a8w8.py` | MXFP8 BMM tuned-row lookup and high-level caller |
| `../../../csrc/opus_gemm/` | canonical registry, C++ family launchers, codegen, traits and pipelines |
