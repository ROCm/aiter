<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE

This directory contains Opus MoE stage2 kernels and their Python bindings. The
current code is gfx950-only and intentionally narrow: fused MoE enablement is
case-gated through tuned A8W4 stage2 configs.

There is one active fused MoE path:

- A8W4 decode stage2 kernels for the `a8w4_decode_k3` family:
  `logical_inter_dim=512`, `inter_dim_pad=128`, and runtime `topk`,
  `hidden`, and `experts`.

The `a8w4_decode_k5` metadata family is retained only as a small bring-up
coverage set for the generalized codegen path; it is not part of the tuned DSV4
production target set.

Private BF16 route-reduce kernel source is retained for future bring-up, but it
is not exposed through `fused_moe` or a Python user API in this PR.

## Kernel Surfaces

### Private BF16 Stage2 Source

The private BF16 source takes:

- `inter_states [token, topk, inter_dim]`, BF16.
- `w2 [expert, hidden, inter_dim]`, BF16.
- CK/Opus MoE sorting metadata: `sorted_token_ids`, `sorted_expert_ids`,
  `num_valid_ids`, and optional `sorted_weights`.
- `route_out [token * topk, hidden]`, BF16 scratch.
- `out [token, hidden]`, BF16 final output.

It writes token-slot route output first, then runs a separate token/topk reduce.

Private BF16 kernel id:

| kid | name | contract |
|---:|---|---|
| `-1` | auto | Select current gfx950 BF16 stage2 kid. |
| `1` | `bf16_gemmstyle256x256x64_token_slot_route_out_no_oob_nfast` | `256 routes x 256 hidden x 64 K`, route-output then reduce, no padded/OOB route rows. |

### A8W4 Decode Stage2

The A8W4 path takes:

- `inter_states [token, topk, 512]`, FP8.
- `w2 [expert, hidden, 256]`, FP4x2 packed.
- `a2_scale [route, scale_cols]`, FP8 E8M0.
- `w2_scale [expert * hidden, scale_cols]`, FP8 E8M0.
- CK/Opus MoE sorting metadata and optional `sorted_weights`.
- `out [token, hidden]`, BF16 direct output, BF16 per-slot route output, or
  MXFP8 per-slot route output.

Direct-output kids atomically accumulate into `[token, hidden]`. The BF16
route-out kid writes `[token * topk, hidden]`; the MXFP8 route-out kid writes
`[token * topk, hidden + hidden / 8]` as payload plus scale. Both use the
shared route-output reduce.

Supported A8W4 kernel ids:

Numbering convention:

- `2000-2099`: K3 decode candidates, assigned contiguously.
- `2100-2109`: K5 direct atomic bring-up candidates.
- `2110-2119`: K5 route-output bring-up candidates.

The current K3 active ids are contiguous. Synthetic balanced / round-robin and
full-tile experiments were retired because they are not valid for general MoE
routing.

| kid | name | block shape | output contract |
|---:|---|---|---|
| `-1` | auto | selected by host shape/block rules | direct-atomic only |
| `2000` | `opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_cache_b3_ws2` | `BM16 x BN64`, `sort_block_m=16` | direct atomic; tuner candidate only for `token <= 1024` |
| `2001` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn2560` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2002` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3008` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2003` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3072` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2004` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3136` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2005` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3200` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2006` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3264` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2007` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3328` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2008` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3456` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2009` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3584` | `BM64 x BN256`, `sort_block_m=64` | MXFP8 route output, reduce width sweep |
| `2010` | `opus_moe2_afp8_wfp4_bf16_t64x256x256_sbm64` | `BM64 x BN256`, `sort_block_m=64` | BF16 route output, then reduce; source retained for explicit use, disabled from tuner because it is not competitive for K3 targets |
| `2100` | `opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_occ6_cache_b2_ws2_k5` | `BM16 x BN64`, `sort_block_m=16` | K5 direct atomic bring-up |
| `2101` | `opus_moe2_afp8_wfp4_atomic_t16x64x256_sbm16_cache_b3_ws2_k5` | `BM16 x BN64`, `sort_block_m=16` | K5 direct atomic bring-up |
| `2110` | `opus_moe2_afp8_wfp4_bf16_t64x256x256_sbm64_k5` | `BM64 x BN256`, `sort_block_m=64` | K5 BF16 route output, then reduce |
| `2111` | `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_k5_rbn2816` | `BM64 x BN256`, `sort_block_m=64` | K5 MXFP8 route output, then reduce |

In fused MoE tuned configs, the preferred A8W4 stage2 selection is a per-kid
`kernelName2` value from the table above. The generic wrapper name
`opus_moe_stage2_a8w4_decode` is still accepted for rows that carry explicit
numeric columns:

- `stage2_kernel_id`: `-1` for direct-atomic auto, or one of the A8W4 kids
  above.
- `stage2_block_m`: the kernel tile M passed to Opus stage2.
- `stage2_route_out`: `1` when stage2 returns per-slot route output that needs
  route-output reduce, otherwise `0`.
- `stage2_reduce_block_n`: optional route-output reduce tile width. Per-kid
  kernel names can also carry this as an `_rbn<N>` suffix, for example
  `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn2816`.

Optional tuned CSV metadata columns `route_bucket`, `expected_sorted_blocks`,
`min_sorted_blocks`, and `max_sorted_blocks` are carried to runtime and checked
after sorting.

## File Layout

Host and shared code:

- `include/opus_moe.h`: C++ entry points exposed to pybind/JIT.
- `include/opus_moe_common.cuh`: shared kernel ids, constants, kargs, and
  metadata helpers.
- `include/opus_moe_arch.cuh`: runtime architecture probe wrapper.
- `include/opus_moe_host_impl.cuh`: host validation and launch selection.
- `opus_moe.cu`: pybind-facing translation unit.

gfx950 code:

- `include/gfx950/opus_moe_arch_gfx950.cuh`: gfx950 launch wrappers and BF16
  generated manifest dispatch.
- `include/gfx950/opus_moe_stage2_route_output_reduce_gfx950.cuh`: shared
  token/topk route-output reduction.
- `include/gfx950/opus_moe_stage2_utils_gfx950.cuh`: small gfx950 device
  helpers, including BF16 packing/conversion helpers.
- `include/gfx950/a16w16/`: BF16/A16W16 stage2 traits and pipeline.
- `include/gfx950/a8w4/opus_moe_traits_stage2_a8w4_decode_gfx950.cuh`:
  A8W4 decode shape traits.
- `include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh`:
  A8W4 decode schedule policy and layout helpers.
- `include/gfx950/a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh`:
  A8W4 decode prologue, mainloop, epilogue, and kernel entry.
- `include/gfx950/a8w4/opus_moe_stage2_a8w4_decode_dispatch_gfx950.cuh`:
  A8W4 kid-to-trait dispatch. The switch cases are generated into
  `opus_moe_stage2_a8w4_manifest.h`.

Python/JIT code:

- `aiter/ops/opus/moe_stage2_a8w4_meta.py`: torch-free A8W4 stage2 kid
  metadata shared by runtime wrapper and csrc tuner/codegen helpers.
- `aiter/ops/opus/moe_stage2_a8w4.py`: A8W4 Python wrapper and route-output
  reduce wrapper.
- `aiter/ops/opus/moe_stage2_a8w4_fused_adapter.py`: fused MoE CSV parsing and
  stage2 wrapper glue for A8W4.
- `gen_instances.py`: JIT-time private BF16 and A8W4 manifest generator.
- `opus_moe_common.py`: private BF16 metadata plus the A8W4 metadata bridge for
  manifest codegen.

## Tuning and Dispatch

`gen_instances.py` emits `opus_moe_stage2_manifest.h` and
`opus_moe_stage2_a8w4_manifest.h` into the JIT build blob. The first generated
header is consumed by private BF16 dispatch source; the second is consumed by
the A8W4 dispatch wrapper for
`kid -> OpusMoeStage2A8W4DecodeShape -> launcher` cases.

A8W4 production selection should be done through the fused MoE tuned
configuration by adding `opus_...` stage2 kernel names only for measured cases
where Opus is correct and faster than the baseline.

## Validation

A8W4 production selection should be validated with model-level traces before
adding or changing tuned CSV entries; local replay manifests and captured routing
dumps should stay outside the repository.

## Current Limits

- gfx950 only.
- Private BF16 source assumes no padded/OOB route rows if it is re-enabled in a
  future change.
- A8W4 production tuning currently targets the `a8w4_decode_k3` family:
  `logical_inter_dim=512`, `inter_dim_pad=128`, effective inter dim `384`,
  and runtime `topk`, `hidden`, and `experts`.
- A small `a8w4_decode_k5` bring-up family is generated to keep the generalized
  metadata/codegen path exercised, but it is not production tuned.
- A8W4 direct-output kernels do not support EP `expert_mask/topk_ids` or
  `bias2`.
- A8W4 route-out kids include a BF16 precision fallback and the MXFP8 path used
  for the fastest large-token route-output candidate.
- Final fused MoE enablement should be case-gated through tuned CSV entries, not
  globally enabled for every compatible-looking shape.
