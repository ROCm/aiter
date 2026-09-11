# FlyDSL eight-wave MXFP8 core

Vendored from [ROCm/FlyDSL at 98c47b58](https://github.com/ROCm/FlyDSL/tree/98c47b58):

- `gemm.py`: `kernels/gemm/mxfp8_gemm_8wave.py`
- `gemm_utils.py`: `kernels/gemm/fp8_gemm_utils.py`
- `moe.py`: `kernels/moe/mxfp8_moe_8wave.py`

The original Apache-2.0 copyright headers and license are retained. Local changes
are package-relative imports, the small integer `ceildiv` helper, formatting,
and a lint annotation preserving distinct compile-time/per-wave DSL conditions.
The adapter in `aiter/ops/flydsl/mxfp8_moe_8wave.py` owns the AITER ABI and allocation.

The grouped core uses the validated final-B-half wait before LDS rotation and
rejoins the staggered M wave groups before the epilogue. Keep both when syncing.
Stage 2 supports separate physical B stride 384 and padded A/scale stride 512.
