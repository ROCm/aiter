# OPUS GEMM C++ and code generation

The public Python contract is documented in
[`aiter/ops/opus/README.md`](../../aiter/ops/opus/README.md). C++ keeps five
family launch ABIs. They are shared private implementation boundaries for the
Python `opus_gemm(..., kid=...)` and `opus_bmm(..., kid=...)` entries; the
public operation split does not duplicate C++ launchers or kernels.

## Exact-id architecture

Kernel identity is `(arch, logical family, kid, Y dtype)`. Python resolves a
bare final id through the merged `kernels_list`; C++ receives an already
resolved family call and performs strict lookup in the current architecture's
typed table.

```text
caller final kid
  -> strict 2D opus_gemm or batch-first 3D opus_bmm
  -> Python canonical registry route and family adapter
  -> family C++ entry
  -> runtime architecture + output-dtype table
  -> exact kid lookup
  -> generated launcher checks
```

C++ does not choose a default kid, read a CSV, run a shape heuristic, redirect
an id, allocate a workspace, or fall back to another backend.

The Python layer retains two distinct A16 shape-driven flows. The generic
`aiter.gemm_a16w16` dispatcher uses the global multi-backend tuned result and,
on a miss or invalid OPUS row, keeps its original skinny, gfx1250 Triton, or
PyTorch fallback. It does not run an OPUS heuristic. The OPUS-only
`gemm_a16w16_opus` compatibility entry instead uses an explicit id when
provided, otherwise attempts a present OPUS tuned row as-is and runs the
migrated per-architecture OPUS heuristic only when no tuned row exists. Every
successful OPUS selection is reduced to one final integer kid before the local
A16 exact launcher and this C++ layer are entered. The compatibility entry does
not re-enter the package-level `opus_gemm`/`opus_bmm` family router. Reusable
candidate/heuristic policy helpers live in `aiter/ops/opus/policy.py`.

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
`aiter/ops/opus/gemm/gemm_op_a16w16.py`. The pybind A16 entry owns normal lazy JIT
build and remains private for parity and performance A/B tests.

## Registry and capability

| Family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a16w16` | direct + two-stage | direct + two-stage | two-stage + pre-built BF16 direct; fused source retained but unregistered |
| `a8w8` | empty | kid 2, FP32 Y | empty |
| `a8w8_blockscale` | empty | kid 1, FP32 Y | empty |
| `a8w8_blockscale_bpreshuffle` | kid 11000, BF16 Y | empty | empty |
| `a8w8_mxscale_bmm` | empty | 45 exact ids in 8000--8653, BF16/FP32 Y | empty |

Empty tables are explicit capability states. The merged registry currently
contains 925 final ids, including 219 pre-built gfx1250 A16W16 CO ids. Those CO
ids currently occupy 21016--21315 inside the reserved `[21000,27000)` band.
The MXFP8 BMM ids are
`8000 + family_local_kid`, which places them in an unused global band while
preserving family-local tuning/debug correlation. Historical child-dictionary
collisions are resolved by the final merge; runtime routing always follows the
resulting `kernels_list[kid]` instance and never a numeric interval.

The gfx942 BF16-workspace A16 exact kids (`10210`, `10213`, `10216`) are the
one workspace-output exception: their current exact-N reducer requires BF16
`Y`. The canonical Python registry rejects FP32 `Y` before launch, matching the
generated host guard.

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
| gfx1250 | 219 | 0 | 496 |

`gen_instances.py` treats tuned CSV ids, the sidecar, the per-architecture
default compile floor, and mandatory A8 ids as build availability. It emits no
runtime shape table. All 45 gfx950 MXFP8 BMM ids are emitted as one family and
deduplicated by generated symbol name rather than entering the ordinary
per-kid subset. All available gfx1250 CO ids are in the gfx1250 compile floor;
codegen emits their five-argument host launchers but no device translation
units. The device bodies come from `gen_co/gfx1250/<symbol>.co`.

The two CO tags (`a16w16_4wave_co` and `a16w16_4wave_wl_co`) use the existing
BF16 direct exact-kid table. Their contract is BF16 XQ/WQ/Y, no bias, no
workspace, and `split_k` 0 or 1. The current gfx1250 public shape policy also
keeps batch at one. Python validates these capabilities before the unchanged
unified A16 C ABI is called; there is no C++ `(M,N,K)` CO lookup table.

## A16 workspace checks

Torch owns every workspace Tensor. Generated launchers validate the final
launch inputs after architecture-specific split resolution:

- XQ/WQ/Y shape, dtype, stride and batch rules;
- exact instance workspace dtype;
- same device, contiguous storage and 16-byte alignment;
- overflow-checked extent and byte-span arithmetic;
- sufficient capacity for the final effective split;
- exact-kid bias support.

Two-stage layouts are split-major. gfx1250 exact kids currently use BF16
workspace storage; the generated launcher/reducer ABI remains typed for either
BF16 or FP32. C++ never owns or retains a Tensor or pointer.

The gfx1250 TDM pipelines use the policy-tag, element-unit API. Clusterlaunch
rounds only the physical grid to `(cluster_wg_m, cluster_wg_n)` multiples;
surplus workgroups arrive at the required cluster barrier and leave through the
uniform `tile_oob` path. Logical tile counts and workspace strides remain
unrounded. The separate reducer dispatches runtime split-K to compile-time
specializations `SPLIT_K_=1..16`, with `SPLIT_K_=0` as the runtime fallback,
using the VEC=8/BLOCK=128 geometry.

The fused gfx1250 factory, emitter and device pipeline remain in-tree for repair,
but `GFX1250_SPLITK_FUSE_ENABLED` is `False`. No fused kid is registered, the
unified capability tables cannot return one, and its `[27000,30000)` band is
unclaimed. The preceding `[21000,27000)` band is reserved for CO ids.

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

For global kid 8326 (family-local kid 326), codegen sets
`PRELOAD_SF_LDS=false` only on the `split_k > 1`, `D_OUT=void` workspace
specialization that writes partial sums. Its direct BF16/FP32
`split_k == 1` specializations keep `PRELOAD_SF_LDS=true`.

## Source layout

| Path | Role |
|---|---|
| `opus_gemm.cu` | family routers, C ABI bridge and strict current-arch dispatch |
| `opus_bmm.cu` / `include/opus_bmm.h` | MXFP8 BMM exact-kid family entry and Torch-workspace forwarding |
| `opus_gemm_common.py` | canonical registry, unique route map and compile-floor constants |
| `gen_instances.py` | subset selection, manifests and typed dispatch generation |
| `codegen/gen_instances_gfx*.py` | exact-instance host launchers and generated input checks |
| `gen_co/` | offline CO manifest/builder, build metadata and packaged gfx1250 ELF images |
| `include/gfx950/opus_bmm_*` | MXFP8 BMM traits, launchers and pipelines |
| `include/gfx1250/opus_co_launch_gfx1250.cuh` | first-use CO loader and cluster launcher |
| `include/gfx*/opus_gemm_arch_*.cuh` | sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | kernel arguments and traits |
