---
title: "Rotary Position Embedding (RoPE)"
last_verified: 2026-04-06
source_files:
  - aiter/ops/rope.py
  - aiter/ops/pos_encoding.py
  - aiter/ops/fused_qk_norm_rope_cache_quant.py
  - aiter/ops/fused_qk_norm_mrope_cache_quant.py
tags: [rope, position-encoding, ck]
---

# Rotary Position Embedding (RoPE)

## Overview
aiter provides RoPE operators for both forward and backward passes, supporting NEOX and GPT-J rotation styles. Multiple fused variants combine RoPE with normalization, KV cache updates, and quantization.

## Key Functions

### Basic RoPE
- `rope_fwd_impl(output, input, freqs, rotate_style, reuse_freqs_front_part, nope_first)` -- forward RoPE
- `rope_bwd_impl(input_grads, output_grads, freqs, rotate_style, ...)` -- backward RoPE
- Input format: "sbhd" (sequence, batch, head, dim)
- Frequency format: [s, 1, 1, d//2] if reuse_freqs_front_part else [s, 1, 1, d]

### Two-Component RoPE
- `rope_2c_fwd_impl(output_x, output_y, input_x, input_y, freqs, ...)` -- separate x/y components
- `rope_2c_bwd_impl(...)` -- backward pass for two-component

### Rotation Styles
- `rotate_style=0`: NEOX style (rotates 2nd half of elements)
- `rotate_style=1`: GPT-J style (rotates odd elements)

### Fused Variants
- `fused_qk_norm_rope_cache_quant` -- fused QK normalization + RoPE + KV cache update + quantization
- `fused_qk_norm_mrope_cache_quant` -- same but with multi-head RoPE (mRoPE)

## Related Pages
- [[operators/cache]] -- KV cache operations (fused with RoPE)
- [[operators/norm]] -- normalization (fused with RoPE in some variants)
- [[operators/attention]] -- attention operators consume RoPE-encoded Q/K

## Source Files
- `aiter/ops/rope.py` -- RoPE forward/backward implementations (~1306 lines)
- `aiter/ops/pos_encoding.py` -- position encoding utilities
- `aiter/ops/fused_qk_norm_rope_cache_quant.py` -- fused RoPE+norm+cache+quant
- `aiter/ops/fused_qk_norm_mrope_cache_quant.py` -- multi-head RoPE variant
