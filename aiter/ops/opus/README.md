# OPUS GEMM and BMM Python interfaces

OPUS exposes separate public functions for logical 2D GEMM and batch-first 3D
BMM. The caller resolves the final kernel id before the call; OPUS does not
select a kernel from the shape.

```python
from aiter.ops.opus import opus_bmm, opus_gemm

opus_gemm(  # XQ [M,K], WQ [N,K], Y [M,N]
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

opus_bmm(  # XQ [B,M,K], WQ [B,N,K], Y [B,M,N]
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

`kid` is mandatory. `Y` is caller-owned and is returned unchanged after the
launch. GEMM versus BMM is determined only by the selected public function;
dtype does not determine the operation. The disjoint architecture id bands
and the merged `kernels_list` form the canonical registry. Both entries call
`kernels_list.get(kid)`; they do not introduce or renumber ids. The returned
instance tag plus the dtype/layout arguments determine the private family
adapter, rather than a second selector or a numeric-range guess.

## Dispatch model

```text
caller-resolved final kid
  -> opus_gemm (strict 2D) or opus_bmm (strict batch-first 3D)
  -> existing kernels_list.get(kid)
  -> instance arch/tag metadata
  -> family-local dtype/layout/scale checks
  -> A16W16 or A8W8 family adapter
  -> shared exact launcher/workspace planner
  -> unchanged exact-kid C++ family table
```

The public operation split does not duplicate kernels, workspace allocation,
or raw bindings. A logical GEMM becomes a batch-one view at the family
boundary. A logical BMM keeps its batch-first public layout; the MXScale
adapter alone converts activation/output tensors to the raw kernel's existing
M-major views with `transpose(0, 1)`, which does not copy storage.

There is no tuned-CSV lookup, architecture heuristic, redirect, or framework
fallback inside either exact public path. The existing high-level A16 caller
resolves those policies before entering it:

```text
explicit kid -> exact public launch
otherwise: tuned CSV -> validated final kid -> exact public launch
                  |
                  +-> no valid row -> skinny (if eligible) -> PyTorch fallback
```

The A16 policy helpers are isolated in `a16w16_policy.py`, while
`tuned_gemm.py` imports only the tuned-candidate validator.  A missing tuned
row does not invoke an OPUS heuristic.  Invalid tuned `(kid, split-K)` pairs
are discarded together before the caller continues to its normal skinny or
PyTorch fallback. Legacy gfx942 requested-to-actual resolution also happens
in the caller, so only the final integer id reaches `opus_gemm` or `opus_bmm`.

## Current families

| Registry family | Current route | Public operation and dtype rules |
|---|---|---|
| `a16w16` | gfx942, gfx950, gfx1250 | GEMM or BMM; BF16 `XQ/WQ`, BF16 or FP32 `Y`, plain WQ, optional bias/split-K/Torch workspace |
| `a8w8` | gfx950 kid 2 | GEMM or BMM; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, no scales |
| `a8w8_blockscale` | gfx950 kid 1 | GEMM or BMM; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, two FP32 scales |
| `a8w8_blockscale_bpreshuffle` | gfx942 kid 11000 | GEMM or batch-one BMM; FP8 `XQ/WQ`, BF16 `Y`, pre-shuffled WQ, two FP32 scales |
| `a8w8_mxscale_bmm` | gfx950 global kids 8000--8653 (45 registered ids) | BMM only; batch-first FP8 inputs, E8M0 scales, BF16 or FP32 output, optional split-K Torch workspace |

Empty family tables on another architecture are valid capability states. A
kid registered for another architecture is rejected before family launch.

## Examples

### A16W16 GEMM and BMM

```python
import torch
from aiter.ops.opus import opus_bmm, opus_gemm

XQ = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
WQ = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
Y = torch.empty((64, 64), device="cuda", dtype=torch.bfloat16)

# The caller/tuner has already chosen gfx950 kid 200 and split_k 2.
opus_gemm(XQ, WQ, Y, kid=200, split_k=2)

XQ_b = torch.randn((8, 64, 512), device="cuda", dtype=torch.bfloat16)
WQ_b = torch.randn((8, 64, 512), device="cuda", dtype=torch.bfloat16)
Y_b = torch.empty((8, 64, 64), device="cuda", dtype=torch.bfloat16)
opus_bmm(XQ_b, WQ_b, Y_b, kid=200, split_k=2)
```

`opus_gemm` requires 2D tensors; `opus_bmm` requires batch-first 3D tensors.
Inputs are K-contiguous and `Y` is N-contiguous. Current gfx1250 kernels still
require BMM batch one. Exact instances can impose additional tile, output
dtype, bias, or K-loop constraints.

### gfx950 A8W8 without scales

```python
Y = torch.empty((M, N), device=XQ.device, dtype=torch.float32)
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

The group contract is 1x128x128. GEMM scales are contiguous FP32
`[M,K/128]` and `[N/128,K/128]` tensors. BMM uses their batch-first 3D forms
`[B,M,K/128]` and `[B,N/128,K/128]` with `opus_bmm`.

### gfx950 MXFP8 BMM

```python
Y = torch.empty((G, M, N), device=XQ.device, dtype=torch.bfloat16)
opus_bmm(
    XQ,                    # [G,M,K], batch-first and K-contiguous
    WQ,                    # [G,N,K], batch-first and K-contiguous
    Y,
    kid=8311,              # exact global id; family-local id 311
    layout="mxscale_bmm",
    x_scale=x_scale,       # [G,M,K/128], one-byte E8M0
    w_scale=w_scale,       # [G,N/128,K/128], one-byte E8M0
    split_k=1,
)
```

The 45 MXFP8 BMM kernels use global ids `8000 + family_local_kid`; the
family-local ids remain recognizable while sharing the canonical
`kernels_list` without colliding with existing GEMM ids. `mxfp8_bmm` and
`bmm_mxscale` are accepted layout aliases. Internally, the family adapter
passes zero-copy `[M,G,*]` transpose views to the unchanged raw kernel ABI.
The high-level tuned caller remains in the existing A8W8 module as
`aiter.batched_gemm_a8w8_mxscale`; it resolves the final id and then calls
`opus_bmm`.

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
| gfx1250 fused | not publicly registered | factory/emitter/source retained for repair |

An explicit workspace must be on the XQ device, contiguous, 16-byte aligned,
of the exact instance dtype, and large enough for the final split. A direct
kid requires `workspace=None`.

gfx1250 clusterlaunch exact kids round the physical launch grid up to complete
clusters; tile-less workgroups exit inside the pipeline. This does not change
the logical workspace shape above. The experimental fused family is disabled,
so no kid in `[21000,30000)` can be resolved through the public registry.

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
A/B measurement. The mixed-module ctypes adapter lives entirely in
`gemm_op_a16w16.py`; generic JIT machinery owns module discovery and build,
but not per-call workspace state.

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

The former shape-driven and family-specific Python APIs were removed. Allocate
`Y`, resolve the final id in the caller, and choose `opus_gemm` for logical 2D
tensors or `opus_bmm` for batch-first 3D tensors. Do not infer the operation
from dtype. The private family modules intentionally export an empty
`__all__`.

## Validation

Run the focused OPUS suite on each target GPU rather than treating
architecture skips as coverage:

```bash
pytest -q \
  op_tests/test_opus_interfaces.py \
  op_tests/test_opus_dispatch.py \
  op_tests/test_opus_ctypes.py \
  op_tests/test_opus_graph.py \
  op_tests/test_opus_workspace.py \
  op_tests/test_opus_gfx950_exhaustive.py \
  op_tests/test_opus_a8w8_bmm.py
```

In particular, gfx942 and gfx1250 validation must run on matching hardware; a
skip on another architecture is not a pass for that target.

## Source map

| Path | Role |
|---|---|
| `__init__.py` | strict public `opus_gemm`/`opus_bmm` contracts and exact-kid routing |
| `_arch.py` | per-explicit-device architecture/CU scalar cache |
| `gemm_op_a16w16.py` | A16 GEMM/BMM adapters over one exact resolver, scalar plan cache, Torch workspace and OPUS-local C ABI adapter |
| `gemm_op_a8w8.py` | A8 GEMM/BMM adapters over shared raw launchers, including MXFP8 BMM layout conversion and workspace planning |
| `../batched_gemm_op_a8w8.py` | MXFP8 BMM tuned-row lookup and high-level caller |
| `../../../csrc/opus_gemm/` | canonical registry, C++ family launchers, codegen, traits and pipelines |
