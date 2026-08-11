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
| K2 `route_dx` legacy | 5 | `BM32 x BN128 x BK64`, route-major grid |
| K2 `route_dx` cohort baseline | 6 | `BM32 x BN128 x BK64`, four-route cohort |
| K2 `route_dx` production | 7 | `BM32 x BN128 x BK32`, four-route cohort |
| K3 `route_reduce` | 0 | `BM16 x BN128` |
| K4 `dw1` small working set | 5 | `BM64 x BN128 x BK32`, expert-fastest |
| K4 `dw1` cohort baseline | 6 | `BM64 x BN128 x BK32`, four-expert cohort |
| K4 `dw1` large working set | 7 | `BM64 x BN128 x BK32`, cohort + direct GMEM-to-LDS |
| K5 `dw2` | 3 | `BM64 x BN64 x BK64` |
| `router_bwd` | 0 | `BM32 x BE8` |
| `bias_bwd` | 0 | `BM32 x BN16` |

Compact variable-routing instances use kid `100`. `RouteLayout` is a
compile-time layout gate, so compact route ids cannot be decoded as fixed
packed token/slot ids.

K2 auto-dispatch selects kid 7 whenever fixed routing provides expert offsets
and the launch has at least four route tiles and more than one N tile. The
BK32 production tile retains two-stage async loads while reducing its LDS
allocation and VGPR count relative to kid 6. Degenerate grids and callers
without expert offsets retain kid 5.

K4 auto-dispatch uses runtime source working-set size rather than an exact
model tuple. Kid 5 remains selected when the combined padded `dZ` and `X`
sources are at most 128 MiB (or fewer than four experts are present). Larger
working sets select kid 7. Kid 6 is retained as the register-load/LDS-store
cohort baseline used to isolate and validate the direct-LDS optimization; it
is not selected by auto-dispatch.

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
