# Opus MoE backward

This directory contains the native gfx950 BF16 MoE backward implementation.
The production path is fixed top-k, concat gate/up SwiGLU, no bias, and no
expert parallelism. Optional bias and compact variable-routing entry points
are kept out of the fixed K1--K5 launch sequence.

## Fixed backward pipeline

The native entry launches five dependent families on the caller's current
stream:

| Stage | Operation | Main output |
| --- | --- | --- |
| K1 `down_bwd` | `dO @ W2`, `dS`, SwiGLU backward, `S * A` | `dZ`, `a_scaled`, `dS` |
| K2 `route_dx` | expert-grouped varlen-M `dZ @ W1` | route `dX` |
| K3 `route_reduce` | reduce fixed top-k routes by token | `dX` |
| K4 `dw1` | expert-grouped varlen-K `dZ^T @ X` | `dW1` |
| K5 `dw2` | expert-grouped varlen-K `dO^T @ (S * A)` | `dW2` |

The host keeps both legal dependency orders flat.  When `1536 <= D <= 2048`
and the runtime `dZ` working set is at least 512 MiB it launches
`K1 -> K4 -> K2 -> K3 -> K5`: K4 consumes K1's large `dZ` stream immediately,
then K2 refreshes the same stream before route reduction.  Smaller working
sets and narrower or wider model dimensions retain
`K1 -> K2 -> K3 -> K4 -> K5`.  This policy is derived from runtime dimensions
and sorted capacity, not an exact model shape.

Backward consumes the sorting metadata saved by forward. The low 24 bits of
`sorted_token_ids` hold the token and the high 8 bits hold the top-k slot.
Padding rows use an out-of-range token. No backward stage reruns TopK.

All GEMM accumulation and route reduction use FP32. Public expert and input
gradients are BF16; router-score gradients remain FP32.

## Code structure

```text
moe_backward/
├── README.md
├── opus_moe_backward_common.py       # production instance table
├── gen_instances.py                  # generated kid/traits/launcher manifest
├── opus_moe_backward.cu              # JIT translation unit
└── include/
    ├── opus_moe_backward.h            # tensor and internal launch API
    ├── opus_moe_backward_common.cuh   # route ABI and family kargs
    ├── opus_moe_backward_host_impl.cuh
    └── gfx950/
        ├── opus_moe_backward_arch_gfx950.cuh
        ├── opus_moe_backward_dispatch_gfx950.cuh
        └── bf16/
            ├── opus_moe_backward_traits_gfx950.cuh
            ├── opus_moe_down_bwd_pipeline_gfx950.cuh
            ├── opus_moe_route_dx_pipeline_gfx950.cuh
            ├── opus_moe_route_reduce_pipeline_gfx950.cuh
            ├── opus_moe_weight_bwd_pipeline_gfx950.cuh
            ├── opus_moe_router_bwd_pipeline_gfx950.cuh
            └── opus_moe_bias_bwd_pipeline_gfx950.cuh
```

The layout follows forward `opus_moe_stage1`:

1. `*_traits_gfx950.cuh` owns compile-time tile and layout choices.
2. Each pipeline file owns device helpers and a flat `process_tile` routine.
3. The `__global__` entry is a thin wrapper around `process_tile`.
4. The family launcher owns only grid selection and whole-kernel dispatch.
5. The gfx950 dispatch table maps a generated kernel id to one launcher.
6. `launch_fixed_pipeline` exposes K1--K5 as one explicit sequential host
   pipeline; tensor validation and kargs construction stay outside it.

Failed tuning candidates do not belong in the instance table. Keep only
configurations that are valid production dispatch targets, including a
fallback when a genuinely different working-set regime needs it.

## Production instances

| Family | Kid | Tile |
| --- | ---: | --- |
| K1 `down_bwd` | 2 | `BM32 x BN128 x BK64` |
| K1 `down_bwd` medium grid | 9 | `BM32 x BN128 x BK32`, M5/twenty-route cohort + CTA-local route metadata |
| K1 `down_bwd` large grid | 10 | `BM32 x BN128 x BK32`, M6/twenty-four-route cohort + CTA-local route metadata |
| K1 `down_bwd` wide-N grid | 11 | `BM32 x BN256 x BK32`, M6 cohort + immediate Z wait |
| K1 `down_bwd` long wide-N grid | 12 | kid 11 geometry + deferred Z wait |
| K1 `down_bwd` forward-cache grid | 13 | kid 11 without the `a_scaled` scale/pack/store epilogue |
| K1 `down_bwd` long forward-cache grid | 14 | kid 12 without the `a_scaled` scale/pack/store epilogue |
| K2 `route_dx` legacy | 5 | `BM32 x BN128 x BK64`, route-major grid |
| K2 `route_dx` cohort baseline | 6 | `BM32 x BN128 x BK64`, four-route cohort |
| K2 `route_dx` small working set | 7 | `BM32 x BN128 x BK32`, M2/four-route cohort |
| K2 `route_dx` large working set | 9 | `BM32 x BN128 x BK32`, M5/ten-route cohort + dZ LDS padding |
| K2 `route_dx` large sorted output | 11 | kid 9 geometry, sorted route workspace for K3 reverse gather |
| K2 `route_dx` wide-N logical output | 13 | `BM32 x BN256 x BK32`, M3/six-route cohort + 96-byte A-slab rotation |
| K2 `route_dx` wide-N sorted output | 14 | kid 13 geometry, sorted route workspace for K3 reverse gather |
| K2 `route_dx` wide-N sorted B-first | 15 | kid 14 geometry, issue the wider W1 transfer before dZ |
| K2 `route_dx` wide-N M5 sorted B-first | 16 | `BM32 x BN256 x BK32`, M5/ten-route compact cohort |
| K2 `route_dx` wide-N M5 binary compact | 17 | kid 16 geometry with binary expert-boundary decode |
| K2 `route_dx` BN512 binary compact | 18 | `BM32 x BN512 x BK32`, M3/six-route compact cohort |
| K2 `route_dx` BN512 N-fast | 19 | kid 18 compute; contiguous output-N traversal, selected for four/eight-tile grids |
| K3 `route_reduce` | 0 | `BM16 x BN128` |
| K4 `dw1` small working set | 5 | `BM64 x BN128 x BK32`, expert-fastest |
| K4 `dw1` direct-LDS baseline | 8 | `BM64 x BN128 x BK32`, two-expert cohort |
| K4 `dw1` production | 9 | `BM128 x BN128 x BK32`, double LDS + two-expert cohort |
| K4 `dw1` wide/long wave8 baseline | 10 | `BM256 x BN128 x BK32`, 512 threads + shared gathered-X tile |
| K4 `dw1` wide/long production | 11 | `BM256 x BN128 x BK32`, 256 threads + eight native C tiles per wave |
| K4 `dw1` long-reduction production | 13 | kid 11 geometry + reverse cohort4 + A-fragment prefetch |
| K4 `dw1` forward-saved sorted-X | 14 | kid 13 geometry + direct padded sorted-row X reads |
| K4 `dw1` sorted-X pipelined | 15 | kid 14 geometry + both K16 fragments prefetched, sorted-X issued first |
| K4 `dw1` sorted-X three-stage | 17 | kid 15 geometry + two future BK32 tiles in flight |
| K4 `dw1` sorted-X eager-AB | 18 | kid 17 geometry + both K16 LDS fragments issued before one wait |
| K5 `dw2` small/degenerate fallback | 3 | `BM64 x BN64 x BK64`, single 8 KiB LDS |
| K5 `dw2` medium-grid production | 10 | `BM128 x BN128 x BK64`, four waves + dual operand LDS |
| K5 `dw2` wide-grid production | 11 | `BM256 x BN128 x BK64`, four waves + K16 reduction fragments, single direct kernel |
| `router_bwd` | 0 | `BM32 x BE8` |
| `bias_bwd` | 0 | `BM32 x BN16` |

Compact variable-routing instances use kid `100`. `RouteLayout` is a
compile-time layout gate, so compact route ids cannot be decoded as fixed
packed token/slot ids.

K2 auto-dispatch uses the combined runtime `dZ + W1` working set. Fixed routing
with expert offsets selects M2 kid 7 at or below 128 MiB. Above that threshold,
`D % 256 == 0` selects the BN256-M3 kid 13; other widths retain BN128-M5 kid 9.
The full K>=4 pipeline pairs their sorted-output variants (kids 14 and 11)
with K3 reverse gather. The BN256 tile rotates successive 16-row dZ slabs by
96 bytes in LDS, which preserves the generic runtime geometry while lowering
LDS wait pressure. Degenerate grids and callers without expert offsets retain
kid 5.

K1 auto-dispatch uses useful launch geometry instead of an exact model tuple
or a tensor-byte proxy. For `I >= 512` divisible by 256, it estimates the
four-group-cohort BN256 grid as
`round_up(ceil(sorted_block_capacity / 6) + E, 4) * (I / 256)` CTAs. At 384
or more CTAs, kid 11 computes two adjacent 128-column N repeats per wave. The
six-route M group still reuses W2 while the wider N tile halves gathered-dO
rereads and dScore workspace/finalize traffic. Smaller or incompatible grids
retain the BN128 policy: it estimates
`(ceil(sorted_block_capacity / 6) + E) * ceil(I / 128)` CTAs, selects kid 2
below 256 CTAs, predecoded M5 kid 9 from 256 through 639, and predecoded M6
kid 10 at 640 or more. The 384-CTA BN256 threshold is a family-level gfx950
occupancy boundary validated across D/E/K and expert-skew cases; it does not
special-case a model shape.
The wide-N epilogue issues all four BF16x8 Z loads before activation math but
keeps them packed as `u32x4`; each group is expanded to FP32 only when it is
consumed. This preserves global-load overlap while removing the previous
56-byte private segment per thread. The production code object remains at
128 VGPRs, 128 accumulator VGPRs, and 60,928 bytes of LDS with zero scratch.
Within each eight-column group, the eight independent `exp2`/reciprocal
sigmoid chains are issued as a batch before the SwiGLU arithmetic. This gives
the transcendental pipeline independent work without extending Z across an
MFMA-result permutation or changing the activation's numerical order.

Forward may optionally save the BF16 tensor
`a_scaled = route_weight * SwiGLU(z_sorted)` in expert-sorted route-major
layout.  The fixed Python wrappers then map the same generic BN256 geometry to
kid 13 for fewer than 65,536 sorted-capacity rows and kid 16 otherwise.  These
instances retain K1's dO@W2, dZ and dScore work, but compile out the scale,
BF16 pack and global `a_scaled` store; K5 consumes the supplied cache directly.
Kid 16 retains kid 15's split-N64 W2 and exact MFMA/activation order, but uses
two 12-KiB Z slabs.  While the current route group's SwiGLU epilogue runs, its
global-to-LDS transfer for the following group is already outstanding.  The
gfx950 trace reports 72,192 bytes of LDS, 128 VGPRs and zero scratch, retaining
two CTAs per CU.  Balanced, active-eight, skewed, K=4, E=8, T=8K--65K and
smaller D/I family screens were bit-exact and favored the pipelined instance;
the selector remains a sorted-capacity/BN256 geometry policy rather than an
exact model-shape check.  Kids 14 and 15 remain explicit comparison overrides.
The cache has shape `[sorted_capacity,I]`, must have been produced with the
same sorting metadata, and is non-differentiable.  Sorter padding rows are
ignored.  Shapes that do not select the general BN256 geometry reject this
fast path instead of silently falling back to a kernel that overwrites the
cache.  The default path remains kids 2/9/10/11/12 and has no API or numerical
change.

K4 auto-dispatch uses runtime launch geometry rather than an exact model
tuple. Kid 11 is selected once the average padded expert interval reaches
2048 routes and its BM256 output grid either has at least 1536 tiles or forms
the smaller 1024-tile full-residency grid. Four waves then share one gathered-X
tile; each wave owns eight independent native C tiles, using 194 VGPRs and
48 KiB LDS with no scratch or register spills. Kid 9 retains short reductions
and smaller output grids; this avoids the old expert-count proxy selecting
BM256 for narrow `2I` grids. Kid 8 covers large BM64 problems, and kid 5
remains the small/degenerate fallback. Kid 10 remains available as the
512-thread wave8 comparison instance but is not selected by production auto.
For average padded intervals of at least 3072 routes with `I <= 1024`, kid 13
retains the reverse-cohort4 schedule and prefetches the second K16 dZ operand
fragment before the first fragment's MFMA.  The narrower A-only prefetch hides
part of the LDS latency without carrying the second X fragment early.
Forward may optionally preserve `x[token]` in the same padded expert-sorted
row domain as `dZ`.  Kid 14 then removes K4's repeated route decode and
token-major X gather while retaining kid 13's generic geometry policy.  The
cache is `[sorted_capacity,D]` BF16 and every sorter-padding row must be exact
zero.  It is forward-owned: constructing it as a standalone backward gather
costs more than the K4 saving.  On the target shape it occupies about 1.08 GB.
Kid 15 additionally places both second-K16 operand fragments before the first
MFMA and issues the smaller sorted-X transfer before dZ.  Kid 17 extends that
path from two 24-KiB LDS stages to three.  After the first stage is ready it
keeps two future BK32 tiles outstanding, uses `vmcnt(6)` to retire only the
older tile, and overlaps the younger tile with the next 16 MFMAs.  Its 72-KiB
LDS and 116 VGPR footprint retains two CTAs per gfx950 CU.  Auto selects kid 17
when kid 13's long-reduction, `I <= 1024`, BM256 output-grid conditions hold,
`I >= 768`, `D >= 1536`, and average padded routes are at most 6400.  Longer
reductions retain two-stage kid 15.  These are runtime grouped-GEMM geometry
boundaries; explicit kids 14/15/17 remain tuning overrides.

K5 auto-dispatch is also geometry based.  Kid 11 requires `D % 256 == 0`,
`I % 128 == 0`, at least 4096 BM256 output tiles, no more than 64 output
tiles per expert, and at least 128 average padded routes.  Its four waves share
one 128-column `a_scaled` slab across 256 `dO` rows; each wave owns eight native
C tiles and consumes the K64 reduction as four K16 transpose-load/MFMA
fragments.  The production kid launches BM256 directly for every expert
interval without a host readback.
The retired BM128 mid-route fallback was slower not only for balanced and
skewed routing but also for its intended 20993--30720 interval after the K16
pipeline landed.  The target code object uses 52 arch VGPRs, 132 accum VGPRs,
112 SGPRs, and 48 KiB LDS with no scratch or spills.  Shapes outside that
launch regime retain kid 10 or the existing smaller fallbacks.

## Shape contract

The current no-tail fixed kernels require:

- gfx950 and BF16 expert tensors;
- contiguous tensors;
- sorting `block_m=32`;
- concat `W1=[E,2I,D]` and `W2=[E,D,I]`;
- `D % 128 == 0` and `I % 128 == 0`;
- fixed top-k in `{1,2,4,8}` for route reduction and router backward.

K4/K5 consume padded expert offsets directly. Empty experts must be written
as exact zero gradients. K1 owns its internal multipart `dS` finalize; this is
not a separate public family.

## Python entry

`aiter/ops/opus/moe_backward.py` contains the JIT bindings, allocation,
validation, fixed/compact metadata objects, public wrappers, and the two
autograd attachment Functions. It has no Torch reference fallback.

## Adding or tuning a kernel

1. Add or change the tile implementation in the matching pipeline file.
2. Add a named trait only when a new production shape is required.
3. Register the production instance in `opus_moe_backward_common.py`.
4. Regenerate the manifest with `gen_instances.py`.
5. Run downstream correctness validation from a clean JIT directory.
6. Compare paired Opus/Triton latency on the same idle gfx950 node before
   changing the auto-dispatch id.
