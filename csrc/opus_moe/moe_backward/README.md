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

The host keeps both legal dependency orders flat.  For power-of-two model
dimensions in `[1024, 2048]` it launches `K1 -> K5 -> K2 -> K3 -> K4`: K5
consumes K1's `a_scaled` while it is cache-young, and K2 refreshes `dZ` before
K4.  Other dimensions retain `K1 -> K2 -> K3 -> K4 -> K5`; wider and
non-power-of-two grids did not benefit from the shorter K5 reuse distance.
This policy uses runtime geometry, not an exact model shape.

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
| K2 `route_dx` legacy | 5 | `BM32 x BN128 x BK64`, route-major grid |
| K2 `route_dx` cohort baseline | 6 | `BM32 x BN128 x BK64`, four-route cohort |
| K2 `route_dx` small working set | 7 | `BM32 x BN128 x BK32`, M2/four-route cohort |
| K2 `route_dx` large working set | 9 | `BM32 x BN128 x BK32`, M5/ten-route cohort + dZ LDS padding |
| K3 `route_reduce` | 0 | `BM16 x BN128` |
| K4 `dw1` small working set | 5 | `BM64 x BN128 x BK32`, expert-fastest |
| K4 `dw1` direct-LDS baseline | 8 | `BM64 x BN128 x BK32`, two-expert cohort |
| K4 `dw1` production | 9 | `BM128 x BN128 x BK32`, double LDS + two-expert cohort |
| K4 `dw1` wide/long wave8 baseline | 10 | `BM256 x BN128 x BK32`, 512 threads + shared gathered-X tile |
| K4 `dw1` wide/long production | 11 | `BM256 x BN128 x BK32`, 256 threads + eight native C tiles per wave |
| K5 `dw2` small/degenerate fallback | 3 | `BM64 x BN64 x BK64`, single 8 KiB LDS |
| K5 `dw2` medium-grid production | 10 | `BM128 x BN128 x BK64`, four waves + dual operand LDS |
| K5 `dw2` wide-grid production | 11 | `BM256 x BN128 x BK64`, four waves + K16 reduction fragments + route-length hybrid |
| `router_bwd` | 0 | `BM32 x BE8` |
| `bias_bwd` | 0 | `BM32 x BN16` |

Compact variable-routing instances use kid `100`. `RouteLayout` is a
compile-time layout gate, so compact route ids cannot be decoded as fixed
packed token/slot ids.

K2 auto-dispatch uses the combined runtime `dZ + W1` working set. Fixed routing
with expert offsets selects M2 kid 7 at or below 128 MiB and M5 kid 9 above
that threshold. Both BK32 tiles retain two-stage async loads while reducing
their LDS allocation and VGPR count relative to kid 6. Degenerate grids and
callers without expert offsets retain kid 5.

K1 auto-dispatch uses the useful M6 launch geometry instead of an exact model
tuple or a tensor-byte proxy. It estimates
`(ceil(sorted_block_capacity / 6) + E) * ceil(I / 128)` CTAs. Fewer than 256
CTAs retain kid 2 for load balance, 256--639 CTAs use the predecoded M5 kid 9,
and 640 or more use the predecoded M6 kid 10. Both grouped kernels keep token,
logical-route, and score metadata CTA-local and preserve the BK32 MFMA and
128-column dScore reduction order. M6 covers 192 route rows per CTA, making
its gathered-dO vector work exactly three loads per thread; the CTA threshold
prevents that wider reuse window from starving gfx950 on smaller grids.

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

K5 auto-dispatch is also geometry based.  Kid 11 requires `D % 256 == 0`,
`I % 128 == 0`, at least 4096 BM256 output tiles, no more than 64 output
tiles per expert, and at least 128 average padded routes.  Its four waves share
one 128-column `a_scaled` slab across 256 `dO` rows; each wave owns eight native
C tiles and consumes the K64 reduction as four K16 transpose-load/MFMA
fragments.  Two device kernels partition experts without a host readback:
route intervals from 20993 through 30720 use the BM128 kid-10 kernel, while
all other intervals use BM256.  The target wide-grid kernel uses 178 VGPRs and
48 KiB LDS with no scratch or spills; the BM128 fallback uses 146 VGPRs and
32 KiB LDS.  Shapes outside that launch regime retain kid 10 or the existing
smaller fallbacks.

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
