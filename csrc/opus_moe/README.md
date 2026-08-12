<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE

This directory contains Opus MoE stage2 kernels and their Python bindings. The
current code is gfx950-only and intentionally narrow: fused MoE enablement is
case-gated through tuned A8W4 stage2 configs.

There is one active fused MoE path:

- A8W4 decode stage2 kernels selected by public algorithm ids plus generated
  effective inter-dim specializations. Runtime `logical_inter_dim` and
  `inter_dim_pad` select the effective-K specialization; `topk`, `hidden`, and
  `experts` remain runtime values.

The same public A8W4 algorithm ids dispatch the matching specialization by
effective inter dim; K is not encoded into separate public kid ranges.

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

- `2000-2099`: A8W4 decode algorithm candidates, assigned contiguously. K is
  selected by generated shape contract dispatch for supported kids, not encoded
  into the kid.

Synthetic balanced / round-robin and full-tile experiments were retired because
they are not valid for general MoE routing.

The source of truth for public ids, names, block shapes, output mode, and
route-reduce tile width is
`aiter/ops/opus/moe_stage2_a8w4_meta.py`. `-1` remains the direct-atomic auto
selector.

In fused MoE tuned configs, the preferred A8W4 stage2 selection is a per-kid
`kernelName2` value from metadata. The generic wrapper name
`opus_moe_stage2_a8w4_decode` is still accepted for rows that carry explicit
numeric columns:

- `stage2_kernel_id`: `-1` for direct-atomic auto, or one of the A8W4 kids
  above.
- `stage2_block_m`: the kernel tile M passed to Opus stage2.
- `stage2_route_out`: `1` when stage2 returns per-slot route output that needs
  route-output reduce, otherwise `0`.
- `stage2_reduce_block_n`: optional route-output reduce tile width. Per-kid
  kernel names can also carry this as an `_rbn<N>` suffix, for example
  `opus_moe2_afp8_wfp4_fp8_t64x256x256_sbm64_rbn3072`.

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
- A8W4 production tuning should use tuned CSV entries for the runtime logical
  inter dim; runtime `inter_dim_pad` selects the compiled effective-K
  specialization.
- A8W4 codegen derives compiled logical/effective inter dims from tuned CSV
  Opus rows. Because the public CSV schema does not encode `inter_dim_pad`,
  the current DSV4 `512-128 -> 384` path is kept by a small codegen seed.
- A8W4 direct-output kernels do not support EP `expert_mask/topk_ids` or
  `bias2`.
- A8W4 route-out kids include a BF16 precision fallback and the MXFP8 path used
  for the fastest large-token route-output candidate.
- Final fused MoE enablement should be case-gated through tuned CSV entries, not
  globally enabled for every compatible-looking shape.
