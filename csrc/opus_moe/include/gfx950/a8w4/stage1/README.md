<!--
SPDX-License-Identifier: MIT
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
-->

# Opus MoE A8W4 Stage1

This directory is the isolated workspace for experimental gfx950 A8W4 stage1
kernels. It is wired only through the explicit stage1 Python/API entry point
and opt-in tuner flags. It is not selected by fused MoE serving dispatch.

Current replay target:

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

Entry points and gating:

```text
Runtime/JIT API:
  C++/pybind symbol: opus_moe_stage1_a8w4_fwd
  Python wrapper:    aiter/ops/opus/moe_stage1_a8w4.py

Tuner opt-in:
  csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
    --stage1-a8w4-flydsl-opus-only
    --include-opus-stage1-a8w4

Serving:
  Fused MoE default dispatch does not select these kernels, and the DSV4 tuned
  CSV currently has no `opus_moe1_a8w4_*` entries.
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
  Thin free-function orchestration: allocate shared scratch, map the CTA tile,
  create layouts/MFMA, call mainloop, call epilogue, and expose the kernel
  symbol without a wrapper struct.

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

1006: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=2,
      cap-routes, split-selector-B, min2 pair-gate/up A-reuse path. Historical
      tuner token hint: 8.

1009: P0 BM128 x BN384 x BK256, SORT_BLOCK_M=128, gate/up group-split,
      noclamp min2, no-scale-guard full-next-A path. Historical tuner token
      hints: 8192/16384.

1011: P0 BM128 x BN384 x BK256, SORT_BLOCK_M=128, gate/up group-split,
      noclamp min3, no-scale-guard full-next-A path. Historical tuner token
      hints: 8192/16384.

1013: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=4,
      cap-routes, split-selector-B, min1 pair-gate/up A-reuse path. Historical
      tuner token hints: 1/4/16/32/64/128/256/512.

1014: P0 BM32 x BN384 x BK256, SORT_BLOCK_M=32, gate/up group-split noclamp
      path. Historical tuner token hints: 512/1024/2048.

1015: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=4,
      cap-routes, split-selector-B, min2 pair-gate/up A-reuse path. Historical
      tuner token hints: 4/16/32/64/128/256/512.

1017: P0 BM128 x BN384 x BK256, SORT_BLOCK_M=128, gate/up group-split,
      noclamp min4, no-scale-guard full-next-A path. Historical tuner token
      hints: 8192/16384.

1019: P0 BM64 x BN384 x BK256, SORT_BLOCK_M=64, t4096 gate/up group-split,
      noclamp min1 path without async-A/cap-routes/assume-route split-B.
      Historical tuner token hint: 8192.

1021: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=4,
      cap-routes, split-selector-B, min3 pair-gate/up A-reuse path. Historical
      tuner token hint: 256.

1022: P0 BM64 x BN384 x BK256, SORT_BLOCK_M=64, t4096 gate/up group-split,
      noclamp min1, async-A, cap-routes, assume-route split-B path. Historical
      tuner token hints: 4096/32768.

1025: P0 BM128 x BN384 x BK256, SORT_BLOCK_M=128, gate/up group-split,
      noclamp min2, no-scale-guard full-next-A split-B path. Historical tuner
      token hints: 8192/16384.

1026/1027/1028/1029: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=4,
      cap-routes noclamp min1/min2/min3/min4 pair-gate/up A-reuse variants.
      Historical tuner token hint: 256.

1031: P0 BM128 x BN384 x BK256, SORT_BLOCK_M=128, gate/up group-split,
      noclamp min3, no-scale-guard split-B path. Historical tuner token hint:
      4096.

1041: P0 BM32 x BN384 x BK256, SORT_BLOCK_M=32, gate/up group-split,
      cap-routes, assume-route split-B, noclamp min1 path. Historical tuner
      token hints: 512/1024/2048.

1049: P0 BM64 x BN384 x BK256, SORT_BLOCK_M=64, t4096 gate/up group-split,
      noclamp min1, async-A, cap-routes, assume-route split-B, no-scale-guard
      path. Historical tuner token hint: 32768.

1074: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G1 K_WAVE=4,
      cap-routes, split-selector-B A-reuse path. Historical tuner token hints:
      2/64/128.

1099: P0 BM16 x BN384 x BK256, SORT_BLOCK_M=16, G6 K_WAVE=1,
      noclamp group-split min1 A-reuse path. Historical tuner token hints:
      256/1024.
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
  Uses free-function helpers to preserve codegen while keeping the body in Opus
  GEMM order: tile/layout creation, MFMA creation, mainloop, then epilogue.

dispatch
  Maps `kid -> traits -> launch`.
```

Current limitation:

```text
The current tuner policy is intentionally pruned while stage1 remains
experimental. Additional candidates should be scheduled only after they have a
measured replay need and a matching correctness/timing artifact.

Clear the Opus MoE JIT cache before validating source or metadata changes:
  rm -rf aiter/jit/build/module_moe_opus aiter/jit/module_moe_opus.so
```
