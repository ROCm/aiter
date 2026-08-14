# OPUS GEMM C++ and code generation

The public Python contract is documented in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). C++ keeps five
family launch ABIs. They are private implementation boundaries for the one
Python `opus_gemm(..., kid=...)` entry.

## Exact-id architecture

Kernel identity is `(arch, logical family, kid, Y dtype)`. Python resolves a
bare final id through the merged `kernels_list`; C++ receives an already
resolved family call and performs strict lookup in the current architecture's
typed table.

```text
caller final kid
  -> Python canonical registry route
  -> family C++ entry
  -> runtime architecture + output-dtype table
  -> exact kid lookup
  -> generated launcher checks
```

C++ does not choose a default kid, read a CSV, run a shape heuristic, redirect
an id, allocate a workspace, or fall back to another backend.

For A16 calls without an explicit id, the existing Python high-level caller
performs `tuned CSV -> per-arch heuristic -> PyTorch fallback`.  A successful
tuned or heuristic result is reduced to one final integer kid before the
public Python router and this C++ layer are entered.  The heuristic functions
live privately in the existing `aiter/ops/opus/gemm_op_a16w16.py` file.

## Family entries

```cpp
void opus_gemm_a16w16_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);

void opus_gemm_a8w8_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    int kid);

void opus_gemm_a8w8_blockscale_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    int kid);

void opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    aiter_tensor_t& Y,
    int kid);

void opus_gemm_a8w8_mxscale_bmm_launch(
    aiter_tensor_t& XQ,
    aiter_tensor_t& WQ,
    aiter_tensor_t& Y,
    aiter_tensor_t& x_scale,
    aiter_tensor_t& w_scale,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);
```

The A16 production path uses the exported status-returning
`opus_gemm_a16w16_launch_cabi`. Nullable tensors represent optional bias and
workspace. The caller supplies the live HIP stream. The bridge validates
integer conversions, switches/restores device and thread-local stream state,
and reports exceptions through thread-local error text before returning to
Python. Its small ctypes adapter is local to
`aiter/ops/opus/gemm_op_a16w16.py`; generic `aiter/jit/core.py` is not
modified. The pybind A16 entry owns normal lazy JIT build and remains private
for parity and performance A/B tests.

## Registry and capability

| Family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a16w16` | direct + two-stage | direct + two-stage | two-stage + fused |
| `a8w8` | empty | kid 2, FP32 Y | empty |
| `a8w8_blockscale` | empty | kid 1, FP32 Y | empty |
| `a8w8_blockscale_bpreshuffle` | kid 11000, BF16 Y | empty | empty |
| `a8w8_mxscale_bmm` | empty | 45 exact ids in 8000--8653, BF16/FP32 Y | empty |

Empty tables are explicit capability states. The merged registry currently
contains 2084 final ids. The PR #4320 BMM ids are `8000 + upstream_kid`, which
places them in an unused global band while preserving upstream tuning/debug
correlation. Historical child-dictionary collisions are resolved
by the final merge; runtime routing always follows the resulting
`kernels_list[kid]` instance and never a numeric interval.

## Generated tables

Generated roots are:

```text
opus_gemm_a16w16_kid_dispatch.h
opus_gemm_a8w8_kid_dispatch.h
opus_bmm_mxscale_kid_dispatch.h
opus_gemm_manifest.h
opus_build_archs.h
```

A16 tables separate direct BF16/FP32 launchers from workspace launchers. A8
tables are family and output-dtype scoped. Every macro has a `_SIZE`; an empty
capability produces `std::array<Entry,0>` without referencing a missing
launcher.

Full canonical A16 counts are:

| Architecture | Direct BF16 | Direct FP32 | Workspace |
|---|---:|---:|---:|
| gfx942 | 14 | 1 | 8 |
| gfx950 | 92 | 92 | 48 |
| gfx1250 | 0 | 0 | 1874 |

`gen_instances.py` treats tuned CSV ids, the sidecar, the per-architecture
default compile floor, and mandatory A8 ids as build availability. It emits no
runtime shape table. All 45 gfx950 MXFP8 BMM ids are emitted as one family and
deduplicated by generated symbol name rather than entering the ordinary
per-kid subset.

## A16 workspace checks

Torch owns every workspace Tensor. Generated launchers validate the final
launch inputs after architecture-specific split resolution:

- XQ/WQ/Y shape, dtype, stride and batch rules;
- exact instance workspace dtype;
- same device, contiguous storage and 16-byte alignment;
- overflow-checked extent and byte-span arithmetic;
- sufficient capacity for the final effective split;
- exact-kid bias support.

Two-stage layouts are split-major. gfx1250 fused instances use tile-major
`[tiles_m,tiles_n,fuse_split_k-1,B_M,B_N]` storage and their compile-time split.
C++ never owns or retains a Tensor or pointer.

gfx942 continues to wave-uniformize both halves of the direct 64-bit workspace
pointer with `__builtin_amdgcn_readfirstlane` in main and reduce kernels.

## A8 input checks

The family router owns common device/dtype checks. Generated exact-instance
launchers own tile and storage details:

- gfx950 no-scale kid 2: matching 3D FP8 inputs, contiguous FP32 output and
  valid K-loop depth/parity;
- gfx950 blockscale kid 1: the same tensors plus contiguous FP32 1x128x128
  scales and exact scale shapes;
- gfx942 bpreshuffle kid 11000: batch one, BF16 output, exact 128-wide N/K
  tiles, registered scale layouts and truly pre-shuffled WQ content.

MXFP8 BMM is gfx950-only. `opus_bmm.cu` first applies the shared FP8/E8M0
shape, stride, device and output checks, then performs an exact lookup in
`opus_bmm_mxscale_kid_dispatch.h`. Unknown ids fail immediately. Generated
launchers enforce their own M/tile/K restrictions; they never redirect to kid
8000 or another family.

For two-stage BMM split-K, the caller supplies a direct FP32 partial-buffer
pointer. Fused split-K stores partials and aligned tile counters in the same
caller Tensor. The reduce kernel also receives the direct pointer. No BMM
launcher allocates, frees, registers or retains workspace memory.

## Source layout

| Path | Role |
|---|---|
| `opus_gemm.cu` | family routers, C ABI bridge and strict current-arch dispatch |
| `opus_bmm.cu` / `include/opus_bmm.h` | MXFP8 BMM exact-kid family entry and Torch-workspace forwarding |
| `opus_gemm_common.py` | canonical registry, unique route map and compile-floor constants |
| `gen_instances.py` | subset selection, manifests and typed dispatch generation |
| `codegen/gen_instances_gfx*.py` | exact-instance host launchers and generated input checks |
| `include/gfx950/opus_bmm_*` | PR #4320 BMM traits, launchers and pipelines |
| `include/gfx*/opus_gemm_arch_*.cuh` | sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | kernel arguments and traits |
