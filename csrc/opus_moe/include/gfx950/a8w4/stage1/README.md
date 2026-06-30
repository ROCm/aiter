<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE A8W4 Stage1

This directory is the isolated workspace for experimental gfx950 A8W4 stage1
kernels. It is wired only through the explicit stage1 Python/API entry point
and is not selected by fused MoE serving dispatch.

Initial target:

```text
large routed A8W4 stage1 bucket
kernel family: t128x256x256
model_dim: 7168
logical_inter_dim: 512
effective_inter_dim: 384
inter_dim_pad: 128
experts: 384
topk: 6
```

File roles:

```text
opus_moe_stage1_a8w4_traits_gfx950.cuh
  Kernel argument contract plus compile-time shape/layout constants for the
  active stage1 candidates.

opus_moe_stage1_a8w4_policy_gfx950.cuh
  Side-effect-free operand layouts, scale/output offsets, CTA tile mapping, and
  route validity helpers.

opus_moe_stage1_a8w4_mainloop_gfx950.cuh
  A global-to-LDS staging, W1/scale register loads, and scaled MFMA
  accumulation for the active P0 path.

opus_moe_stage1_a8w4_epilogue_gfx950.cuh
  `rc -> shared gate/up -> SiLU -> MXFP8 quant -> output/out_scale`.

opus_moe_stage1_a8w4_pipeline_gfx950.cuh
  Small orchestration wrapper: allocate shared scratch, create layouts, call
  mainloop, call epilogue, and expose the kernel symbol.

opus_moe_stage1_a8w4_dispatch_gfx950.cuh
  Stage1 `kid -> traits -> launch` dispatch. The switch cases are generated
  from the shared stage1 metadata manifest, not hand-maintained locally.
```

Keep stage2 files in `../` unchanged while stage1 is experimental. Move shared
A8W4 helpers into a common folder only after both stage1 and stage2 need the same
code.

Active candidates:

```text
Stage1 ids are registered through `aiter/ops/opus_moe_stage1_a8w4_meta.py` and
the generated `opus_moe_stage1_a8w4_meta.h` / `opus_moe_stage1_a8w4_manifest.h`
headers. They use the 10xx range; A8W4 stage2 ids are generated from the
stage2 meta manifest outside this experimental range.

1010: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, K_WAVE=4,
      single output-group A-reuse path. Selected for token=1/2.

1020: P0 BM16 x BN64 x BK256, SORT_BLOCK_M=32, K_WAVE=2,
      single output-group A-reuse path. Selected for token=4 only.

1030: P0 BM16 x BN64 x BK256, SORT_BLOCK_M=32,
      single output-group A-reuse path. Selected for token=8/16.

1040: P0 BM32 x BN384 x BK256, SORT_BLOCK_M=32,
      full six-output-group A-reuse path. Selected for token=64/256/512/1024.

1050: P0 BM64 x BN384 x BK256, SORT_BLOCK_M=64,
      gate/up group-split path. Selected for token=128.

1060: P0 BM128 x BN256 x BK256, SORT_BLOCK_M=128,
      EPILOGUE_ROW_SPLIT=2, gate/up group-split ownership. Selected for
      token=32 and large token buckets.
```

Current code structure:

```text
traits
  Owns kargs, compile-time shape, row-split constants, and smem sizing.

policy
  Owns `make_layout_ga`, `make_layout_gb`, scale/output/smem coordinate helpers,
  CTA tile mapping, and route validity.

mainloop
  Owns explicit `ga/gb/ra/rb/rc`, A LDS staging, W1/scale loads, and scaled
  MFMA accumulation.

epilogue
  Owns `rc -> shared gate/up -> SiLU -> MXFP8 quant -> output/out_scale`.

pipeline
  Composes policy/mainloop/epilogue and exposes the kernel symbol.

dispatch
  Maps `kid -> traits -> launch`.
```

Current limitation:

```text
The current tuner policy is intentionally pruned while stage1 remains
experimental. Additional candidates should be scheduled only after they have a
measured replay need and a matching correctness/timing artifact.
```
