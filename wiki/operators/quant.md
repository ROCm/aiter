---
title: "Quantization Operators"
last_verified: 2026-04-06
source_files:
  - aiter/ops/quant.py
tags: [quantization, fp8, fp4, int8, smoothquant]
---

# Quantization Operators

## Overview
aiter provides quantization operators for converting between floating-point formats (BF16/FP16) and quantized formats (FP8, INT4/FP4, INT8). These are critical for feeding quantized GEMM operators and for MoE pipelines.

## Key Functions

### SmoothQuant
- `smoothquant_fwd(out, input, x_scale, y_scale)` -- applies smooth quantization
- `moe_smoothquant_fwd(out, input, x_scale, topk_ids, y_scale)` -- MoE-specific variant

### Per-token Quantization
- `pertoken_quant(x, scale, x_scale, scale_dtype, quant_dtype, dtypeMax)` -- per-token dynamic quantization
- Pure PyTorch implementation with optional smooth-quant pre-scaling
- Computes per-token amax, scales, and quantizes to target dtype

### FP4 Quantization
- `per_1x32_f4_quant(x, scale, quant_dtype, shuffle)` -- per-1x32 block FP4 quantization
- Block size of 32, uses E8M0 scale format with E2M1 mantissa
- Optional scale shuffling for optimized memory access

### Dynamic FP8 Quantization
- Triton-based kernels for per-token and per-group FP8 quantization
- Fused with activation in MoE context (act + quant in single kernel)

### MXFP4 Quantization
- Triton-based microscaling FP4 quantization
- Block-level quantization with shared exponents

## Supported Formats
| Format | Quantized Type | Scale Type | Block Size |
|--------|---------------|------------|------------|
| Per-token FP8 | FP8 (e4m3/e5m2) | FP32 | per-token |
| Per-token INT8 | INT8 | FP32 | per-token |
| Block-scale FP8 | FP8 | FP32 | per-block |
| Per-1x32 FP4 | FP4 (e2m1) | E8M0 | 32 |
| MXFP4 | FP4 | shared exponent | block |

## Related Pages
- [[operators/gemm]] -- quantized GEMM consumers
- [[operators/moe]] -- MoE pipeline with fused quant+activation
- [[operators/norm]] -- fused norm+quant variants

## Source Files
- `aiter/ops/quant.py` -- quantization functions (smoothquant, pertoken, FP4)
