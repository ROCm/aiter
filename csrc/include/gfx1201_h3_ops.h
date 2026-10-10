// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>

// MiniMax-H3 gfx1201 row/element-wise ops: BF16 DiT block pieces and the FP16/FP32 video VAE decoder block pieces.
// Pointers are device addresses (0 = absent where documented); `stream` is a hipStream_t passed as an integer.
void launch_gfx1201_rms_modulate(int64_t x, int64_t weight, int64_t scale, int64_t shift, int64_t indices,
                                 bool indices_int64, int64_t out, int64_t rows, int64_t scale_stride,
                                 int64_t shift_stride, double eps, int64_t stream);
void launch_gfx1201_swiglu(int64_t in, int64_t out, int64_t rows, int64_t features, int64_t stream);
void launch_gfx1201_gated_residual(int64_t residual, int64_t projected, int64_t gate, int64_t indices,
                                   bool indices_int64, int64_t out, int64_t rows, int64_t dim, int64_t gate_stride,
                                   bool vectorized, int64_t stream);
// addend/scale may be 0 (no residual update); weight/out may be 0 (residual update only).
void launch_gfx1201_vae_residual_rms(int64_t hidden, int64_t addend, int64_t scale, int64_t weight, int64_t out,
                                     int64_t rows, int64_t dim, double eps, int64_t stream);
void launch_gfx1201_vae_qk_norm_rope(int64_t query, int64_t key, int64_t cos16, int64_t sin16, int64_t rows,
                                     int64_t heads, int64_t query_stride, int64_t key_stride, double eps,
                                     int64_t stream);
void launch_gfx1201_vae_swiglu(int64_t in, int64_t out, int64_t rows, int64_t features, int64_t in_stride,
                               int64_t stream);
