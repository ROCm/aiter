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
provided, otherwise tries a tuned row and falls back to its heuristic for a
missing or invalid row. All selections pass through legacy compatibility
resolution before the local exact launcher; the strict `opus_gemm`/`opus_bmm`
APIs never redirect. Reusable policy helpers live in
`aiter/ops/opus/policy.py`.

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

## Registry and capability

| Family | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `a16w16` | direct + two-stage | direct + two-stage | two-stage + pre-built BF16 direct; fused source retained but unregistered |
| `a8w8` | empty | kid 2, FP32 Y | empty |
| `a8w8_blockscale` | empty | kid 1, FP32 Y | empty |
| `a8w8_blockscale_bpreshuffle` | kid 11000, BF16 Y | empty | empty |
| `a8w8_mxscale_bmm` | empty | registered BMM ids, including compact ids 8470--8484; BF16/FP32 Y | empty |

Empty tables are explicit capability states. `kernels_list` is the source of
truth for registered ids. The pre-built gfx1250 A16W16 CO ids occupy the
reserved `[21000,27000)` band.
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
runtime shape table. All gfx950 MXFP8 BMM ids in `BMM_MXSCALE_KIDS` are emitted
as one family and deduplicated by generated symbol name rather than entering the ordinary
per-kid subset. All available gfx1250 CO ids are in the gfx1250 compile floor;
codegen emits their five-argument host launchers but no device translation
units. The device bodies come from `gen_co/gfx1250/<symbol>.co`.

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
| `opus_bmm.cu` / `include/opus_bmm.h` | MXFP8 BMM exact-kid family entry and Torch-workspace forwarding |
| `opus_gemm_common.py` | canonical registry, unique route map and compile-floor constants |
| `gen_instances.py` | subset selection, manifests and typed dispatch generation |
| `codegen/gen_instances_gfx*.py` | exact-instance host launchers and generated input checks |
| `gen_co/` | offline CO manifest/builder, build metadata and packaged gfx1250 ELF images |
| `include/gfx950/opus_bmm_*` | MXFP8 BMM traits, launchers and pipelines |
| `include/gfx1250/opus_co_launch_gfx1250.cuh` | first-use CO loader and cluster launcher |
| `include/gfx*/opus_gemm_arch_*.cuh` | sorted exact-kid tables |
| `include/gfx*/**/opus_gemm_traits*.cuh` | kernel arguments and traits |


## Compact MXFP8 BMM on gfx950

Kids 8470--8476 retain their original `N=1024, K=4096` device specialization.
Other supported N/K shapes launch the same kid's general compact kernel. It
stages at most 4096 K elements of E8M0 scales at a time and retains FP32
accumulators across segments, so LDS usage does not grow with total K.
Short and final segments reread their last valid A/B tile to maintain the
prefetch depth, then drain outstanding copies before reusing LDS.

The compact contract is:

- `1 <= M < 2048`, `1 <= batch <= 16`;
- positive N divisible by both 128 and the kid's `B_N`;
- positive K divisible by the kid's `B_K` (128, 256, or 512);
- ordinary 128x128 weight scales and 1x128 activation scales;
- contiguous token-major A/scales/Y storage and `(16,16)` preshuffled B;
- BF16 or FP32 output, split-K 0/1, no workspace;
- 16-byte A/B/Y pointer alignment and 4-byte scale pointer alignment;
- A's total bytes, each B batch's bytes, and Y's total bytes fit signed int32.

The public `opus_bmm` interface still uses batch-first transpose views and
`layout="mxscale_bmm"`. The C++ launcher validates physical strides. Shape
support is not a performance selection: existing general pipelines remain
available, and the tuner compares eligible compact kids without changing
existing tuned CSV rows. Regression coverage is in
`op_tests/test_opus_a8w8_bmm_mxscale_bpreshuffle.py`.

Kid 8477 adds a `16x32x512` compact tile for small M: 128 threads (two
compute waves in a 1x2 grid), three A/B LDS slots, and 73 KiB of LDS.
It supports both the fixed N=1024/K=4096 and general paths, and requires K
divisible by 512. All waves stage A/B and compute; this differs
from kid 8179's same-size `bdirect` tile, where two producer waves stage A
and two consumer waves load B directly into registers and perform MFMA.

Additional compact candidates use four compute waves and the same group128
scale contract. The LDS column includes A/B ring and scale panels; it alone
does not determine actual occupancy, which also depends on registers.

| Kid | Tile MxNxK | Wave grid MxN | A/B slots | LDS KiB |
|---|---|---|---|---|
| 8478 | 32x32x512 | 2x2 | 2 | 66 |
| 8479 | 16x64x512 | 1x4 | 2 | 82 |
| 8480 | 16x64x256 | 1x4 | 3 | 62 |
| 8481 | 32x64x256 | 2x2 | 3 | 74 |
| 8482 | 64x32x256 | 2x2 | 3 | 75 |
| 8483 | 64x64x128 | 2x2 | 3 | 51 |
| 8484 | 128x128x128 | 2x2 | 2 | 69 |

These candidates cover intermediate tile sizes and shallower rings. They are
additional tuner options, with no changes to existing kids or tuned CSV rows.

## Tuning MXFP8 BMM

See [MXFP8 BMM tuning](../bmm_a8w8_mxscale/README.md) for the header-only
input/output CSV templates, input schema, OPUS/FlyDSL tuning commands,
configuration verification and performance comparison workflow.

The joint tuner is `csrc/bmm_a8w8_mxscale/bmm_a8w8_mxscale_bpreshuffle_tune.py`.
The OPUS-only entry is `csrc/opus_gemm/opus_bmm_mxscale_tune.py`; it uses the
same CSV schema and shared tuner options. Pass `--bpreshuffle` when tuning
for the preshuffled-weight interface.
