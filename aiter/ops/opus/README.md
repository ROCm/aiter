# OPUS GEMM and BMM Python interfaces

OPUS exposes strict exact-kid functions for logical 2D GEMM and batch-first 3D
BMM, plus the retained shape-driven `gemm_a16w16_opus` compatibility entry.
The exact functions never select a kernel from the shape; the compatibility
entry resolves an A16W16 kid before entering the same exact path.

## Public API

```python
import torch

from aiter.ops.opus import gemm_a16w16_opus, opus_bmm, opus_gemm

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

# Retained A16W16 shape-driven API. An explicit kernelId wins; otherwise this
# performs OPUS-only tuned lookup followed by the per-architecture heuristic.
result = gemm_a16w16_opus(A, B, bias=None, dtype=torch.bfloat16)
```

For `opus_gemm` and `opus_bmm`, `kid` is mandatory. `Y` is caller-owned and is
returned unchanged after the launch. The selected exact function determines
the logical rank, while the resolved family must support that operation;
dtype does not determine it.
Among A8 families, no-scale, blockscale and blockscale-bpreshuffle are
GEMM-only, while MXFP8 is BMM-only. The disjoint architecture id bands and the
merged `kernels_list` form the canonical registry. Both entries call
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
  -> shared immutable A16 launch plan or A8 family planner
  -> family executor
  -> unchanged exact-kid C++ family table
```

The public operation split does not duplicate kernels, workspace allocation,
or raw bindings. A logical GEMM becomes a batch-one view at the family
boundary. A logical BMM keeps its batch-first public layout; the MXScale
adapter alone converts activation/output tensors to the raw kernel's existing
M-major views with `transpose(0, 1)`, which does not copy storage. The physical
3D raw ABI used by a non-MX A8 GEMM is not exposed as public BMM.

There is no tuned-CSV lookup, architecture heuristic, redirect, or framework
fallback inside either exact public path. The two shape-driven A16 callers
remain intentionally different:

```text
aiter.gemm_a16w16
  -> global multi-backend tuned row -> selected backend
  -> no valid row -> skinny (eligible gfx90a/gfx942/gfx950)
                  -> gfx1250 Triton
                  -> otherwise PyTorch

gemm_a16w16_opus
  -> explicit kernelId -> local exact A16 GEMM/BMM launcher
  -> otherwise OPUS-only tuned row -> attempt that exact kid as-is
  -> missing OPUS row -> per-arch OPUS heuristic
                       -> validate -> local exact A16 GEMM/BMM launcher
```

The shared OPUS candidate helpers are isolated in `policy.py`.
`tuned_gemm.py` uses only the tuned-candidate validator; a missing or invalid
OPUS row continues through its normal framework fallback and never invokes an
OPUS heuristic. `gemm_op_a16w16.py` owns the compatibility wrapper and applies
the migrated gfx942/gfx950/gfx1250 heuristic only after an OPUS-only tuned
miss. A present but invalid OPUS row is not replaced by a heuristic kid: the
exact launcher reports the invalid row. If the heuristic-selected kid is
invalid, the OPUS-only wrapper raises instead of trying secondary kids or
silently switching backend. Legacy gfx942 requested-to-actual resolution
happens before heuristic launch, so only a final integer id reaches
the local exact A16 family launcher. The compatibility entry calls that
launcher directly rather than re-entering the package-level family router.

There is currently no high-level A16W16 BF16 BMM wrapper. In particular,
`aiter/ops/batched_gemm_op_bf16.py` contains the existing CK entry points but
does not define `batched_gemm_bf16_OPUS` or a tuned CK/OPUS dispatcher. A16W16
BMM therefore starts at exact-kid `opus_bmm`: its caller owns `Y`, resolves the
final `kid`/`split_k`, and may provide a Torch workspace. The public router
calls `_launch_a16w16_bmm`, which preserves the batch dimension and forwards
to the same `_execute_a16w16` planner/executor used by A16W16 GEMM.

## Current families

| Registry family | Current route | Public operation and dtype rules |
|---|---|---|
| `a16w16` | gfx942, gfx950, gfx1250 | GEMM or BMM; BF16 `XQ/WQ`, normally BF16 or FP32 `Y`, plain WQ, optional bias/split-K/Torch workspace; gfx942 BF16-workspace exact kids require BF16 `Y` |
| `a8w8` | gfx950 kid 2 | GEMM only; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, no scales |
| `a8w8_blockscale` | gfx950 kid 1 | GEMM only; FP8 `XQ/WQ`, FP32 `Y`, plain WQ, two FP32 scales |
| `a8w8_blockscale_bpreshuffle` | gfx942 kid 11000 | GEMM only; FP8 `XQ/WQ`, BF16 `Y`, pre-shuffled WQ, two FP32 scales |
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
dtype, bias, or K-loop constraints. The BMM example is a direct exact-API call;
there is no current `batched_gemm_bf16_OPUS` high-level wrapper.

A16 bias follows the `F.linear` output-feature convention: `[N]` broadcasts
across batch and `[batch,N]` supplies a separate bias for each batch.

### gfx950 A8W8 without scales

```python
from aiter.ops.opus import opus_gemm

Y = torch.empty((M, N), device=XQ.device, dtype=torch.float32)
opus_gemm(XQ, WQ, Y, kid=2)
```

The general `aiter.gemm_a8w8` API remains the scaled CK/Triton operation and
requires both `x_scale` and `w_scale`; omitting scales does not select OPUS.

### gfx950 A8W8 blockscale

```python
from aiter.ops.opus import opus_gemm

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
`[M,K/128]` and `[N/128,K/128]` tensors. This family does not accept
`opus_bmm`. The general `aiter.gemm_a8w8_blockscale` dispatcher remains a
BF16/FP16 CK/CKTile/ASM/Triton API; FP32 output is available only through the
explicit OPUS exact-kid call above.

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
`kernels_list` without colliding with existing GEMM ids. The public layout
name is strictly `mxscale_bmm`. Internally, the family adapter passes zero-copy
`[M,G,*]` transpose views to the unchanged raw kernel ABI.
The high-level tuned caller remains
`aiter.batched_gemm_a8w8_mxscale`. Its cold-path tuned-row, padded-M,
local-to-global-id and heuristic selection live in `policy.py` beside the A16
caller policy.
The caller caches the final id/split pair per shape: split-one enters the
checked raw launcher directly, while workspace launches call `opus_bmm`.

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
tiles, BF16 output, and its registered scale storage contracts. The high-level
`gemm_a8w8_blockscale_bpreshuffle` dispatcher enters this OPUS route only when
the tuned row has `libtype=opus`; CK, CKTile, ASM and Triton rows remain on
their respective backends. gfx950 currently registers zero OPUS kids for this
family, so a gfx950 OPUS validation must report it unavailable rather than run
a non-OPUS fallback as coverage.

## Tuning compatibility

The exact public APIs execute a caller-selected kid. A16 production tuning
continues through `csrc/gemm_a16w16/gemm_a16w16_tune.py`; MXFP8 BMM uses
`csrc/opus_gemm/opus_bmm_mxscale_tune.py`.

The CK-owned blockscale tuner remains unchanged. Its legacy
`opus_gemm_a8w8_blockscale_bpreshuffle_tune(...)` import is retained in
`gemm_op_a8w8.py` and calls the bpreshuffle family launcher directly. New
OPUS-specific tuning code should live under `csrc/opus_gemm/`.

## A16 Torch workspace

Workspace ownership is call-scoped and remains in Torch:

```text
validate exact kid and split_k
  -> derive immutable workspace plan from the exact instance
  -> reuse caller workspace or torch.empty for this call
  -> _launch_a16w16_backend
```

There is no process-global Tensor, pointer registry, HIP allocator, or prewarm
API. The bounded public-contract and A16 launch-plan caches store only registry
metadata, integers, dtypes, option-presence flags and shapes; they never retain
Tensor objects, data pointers, devices, streams or workspaces.

Let `padded_M=ceil_div(M,B_M)*B_M` and
`padded_N=ceil_div(N,B_N)*B_N`:

| Architecture/family | Workspace shape | Instance storage |
|---|---|---|
| gfx950 two-stage | `[workspace_capacity_split_k,batch,padded_M,padded_N]` | FP32 |
| gfx942 two-stage | `[workspace_capacity_split_k,batch,padded_M,padded_N]` | exact BF16/FP32 dtype |
| gfx1250 two-stage | `[workspace_capacity_split_k,padded_M,padded_N]` | BF16 |
| gfx1250 pre-built CO direct | none | none |
| gfx1250 fused | not publicly registered | factory/emitter/source retained for repair |

For gfx942, `workspace_capacity_split_k` records the allocated capacity while
`abi_split_k` records the possibly down-clamped value passed to the launcher.
Both are fields of the same immutable launch plan.

An explicit workspace must be on the XQ device, contiguous, 16-byte aligned,
of the exact instance dtype, and large enough for the final split. A direct
kid requires `workspace=None`.

gfx1250 clusterlaunch exact kids round the physical launch grid up to complete
clusters; tile-less workgroups exit inside the pipeline. This does not change
the logical workspace shape above. The experimental fused family is disabled,
so no fused kid in `[27000,30000)` can be resolved through the public registry.
The `[21000,27000)` band belongs to pre-built CO kids, which are direct and
therefore never request a workspace.

gfx942 BF16-workspace kids `10210`, `10213`, and `10216` are exact ids. Their
registered exact-N contract is `{64,128,256,384,512,1024,2048}`. A different N
or an FP32 `Y` is rejected; the call is never redirected to an FP32-workspace
kid.

## MXFP8 BMM Torch workspace

MXFP8 BMM follows the same ownership rule: Python either uses the caller's
contiguous FP32 Tensor or creates a call-scoped `torch.empty`; C++ receives a
direct pointer and never retains it. Two-stage split-K uses
`split_k * G * padded_M * padded_N` FP32 elements. The fused family stores its
partials and aligned tile counters in one FP32 Tensor. `split_k == 1` and
families that do not consume workspace reject a supplied Tensor.

`launch_plan.py` owns the shared immutable workspace specification plus the
family-specific A16W16 and A8W8 plans. Its `A8W8MxscaleBMMPlan` records the
resolved exact kid, the split-K value passed to the ABI and an optional
`WorkspaceSpec`. `gemm_op_a8w8.py` only adapts logical layouts, materializes
that workspace and invokes `_launch_a8w8_backend`.

For gfx950 kid 8326, only the `split_k > 1`, `D_OUT=void` workspace
specialization sets `PRELOAD_SF_LDS=false` to avoid the ROCm 7.2.4 compiler
failure. Its direct BF16/FP32 `split_k == 1` specializations keep
`PRELOAD_SF_LDS=true`.

## A8 pybind backend

All three non-MX A8 GEMM adapters and the MXFP8 BMM executor enter one
low-level facade:

```text
validated family + resolved kid + physical Tensor views
  -> _launch_a8w8_backend
       -> no-scale pybind raw launcher
       -> plain blockscale pybind raw launcher
       -> blockscale-bpreshuffle pybind raw launcher
       -> MXFP8 BMM pybind raw launcher
```

## Graphs and streams

Automatic `torch.empty` during graph capture uses the graph-private pool.

A CO image is opened and registered on the first call to its launcher. That
first load must happen before graph capture; warm-up followed by capture/replay
is supported, while first-ever loading inside capture is not. `import aiter`
sets `OPUS_GEN_CO_DIR` to the packaged `csrc/opus_gemm/gen_co` directory, and an
explicit environment value overrides it for testing locally rebuilt images.

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
exact-routable. A gfx1250 build keeps all 219 available CO host launchers in its
default compile floor; their device code remains in the packaged `.co` files.

## Migration

For new exact-id integrations, allocate `Y`, resolve the final id in the
caller, use `opus_gemm` for logical 2D calls, and use `opus_bmm` for
batch-first 3D calls. The retained `gemm_a16w16_opus` entry preserves the
former A16W16 shape-driven behavior: explicit id, then OPUS-only tuned lookup,
then the migrated per-architecture heuristic. Do not infer the operation from
dtype or expose the physical 3D raw ABI of a GEMM-only A8 family as public
BMM. The A8 family module exports only its legacy tuner compatibility name.

## Validation

Run the retained OPUS numerical tests on matching target GPUs rather than
treating architecture skips as coverage:

```bash
pytest -q op_tests/test_opus_a16w16_gemm.py
PYTHONPATH=. python3 op_tests/test_opus_a8w8_bmm.py \
  -g 2 -s 16,1024,4096 -d bf16
```

In particular, gfx942 and gfx1250 validation must run on matching hardware; a
skip on another architecture is not a pass for that target.

## Source map

| Path | Role |
|---|---|
| `__init__.py` | thin public `opus_gemm`/`opus_bmm` delegates and lazy `gemm_a16w16_opus` compatibility entry |
| `dispatch.py` | public contract validation and strict exact-kid family routing |
| `_arch.py` | per-explicit-device architecture/CU scalar cache |
| `policy.py` | A16 tuned/heuristic candidate selection plus MXFP8 tuned CSV discovery, padded-M lookup, local-to-global kid normalization and heuristic fallback |
| `launch_plan.py` | shared `WorkspaceSpec`, A16 exact-kid/split-K planning, and A8 family contract/MXFP8 BMM planning |
| `gemm_op_a8w8.py` | three non-MX A8 GEMM adapters, the legacy bpreshuffle tuner compatibility entry, MXFP8 BMM workspace materialization, and the unified `_launch_a8w8_backend` over four pybind raw bindings |
| `moe_stage1_a8w4.py` | A8W4 MoE stage-1 runtime binding and launcher |
| `moe_stage2_a8w4.py` | A8W4 MoE stage-2 runtime bindings and launchers |
| `../gemm_op_a8w8.py` | general scaled CK/CKTile/ASM/Triton A8 dispatchers plus the tuned-row OPUS bpreshuffle route |
| `../batched_gemm_op_bf16.py` | existing high-level CK BF16 BMM path; it is not an OPUS A16W16 BMM wrapper |
| `../batched_gemm_op_a8w8.py` | MXFP8 high-level caller, scalar launch cache, output allocation and split-one/workspace execution choice |
| `../../../csrc/opus_gemm/` | canonical registry, C++ family launchers, codegen, traits and pipelines |
