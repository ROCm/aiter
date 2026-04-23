# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import triton
import triton.language as tl
from .helpers import (
    _tanh,
    AUTOTUNE_WINO4_INPUT_CONFIGS,
    AUTOTUNE_WINO_GEMM_CONFIGS,
    AUTOTUNE_WINO4_OUTPUT_CONFIGS,
    AUTOTUNE_FUSED_F4X3_CONFIGS,
)


@triton.autotune(
    configs=AUTOTUNE_WINO4_INPUT_CONFIGS,
    key=["T", "C_pad"],
    cache_results=True,
)
@triton.jit
def _winograd_f4x3_input_transform_kernel(
    X,
    V,
    N: tl.constexpr,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
    LAYOUT: tl.constexpr = 0,
):
    # X layout: LAYOUT=0 NCHW, LAYOUT=1 NHWC
    if LAYOUT == 0:
        stride_x_w: tl.constexpr = 1
        stride_x_h: tl.constexpr = W_in
        stride_x_c: tl.constexpr = H * W_in
        stride_x_n: tl.constexpr = C * H * W_in
    else:
        stride_x_w: tl.constexpr = C
        stride_x_h: tl.constexpr = W_in * C
        stride_x_c: tl.constexpr = 1
        stride_x_n: tl.constexpr = H * W_in * C
    # V layout: [36, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    h_start = th * 4 - pad_h
    w_start = tw * 4 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    base = X + n * stride_x_n + offs_c * stride_x_c
    n_valid = n < N

    # Load 6x6 patch
    d00 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d01 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d02 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d03 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d04 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d05 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d10 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d11 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d12 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d13 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d14 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d15 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d20 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d21 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d22 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d23 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d24 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d25 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d30 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d31 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d32 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d33 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d34 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d35 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d40 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d41 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d42 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d43 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d44 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d45 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d50 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d51 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d52 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d53 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d54 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d55 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)

    for r in tl.static_range(6):
        h = h_start + r
        h_valid = n_valid & (h >= 0) & (h < H)
        for s in tl.static_range(6):
            w = w_start + s
            valid = h_valid & (w >= 0) & (w < W_in)
            ptr = base + h * stride_x_h + w * stride_x_w
            val = tl.load(ptr, mask=valid & c_mask, other=0.0)
            if r == 0:
                if s == 0:
                    d00 = val
                elif s == 1:
                    d01 = val
                elif s == 2:
                    d02 = val
                elif s == 3:
                    d03 = val
                elif s == 4:
                    d04 = val
                else:
                    d05 = val
            elif r == 1:
                if s == 0:
                    d10 = val
                elif s == 1:
                    d11 = val
                elif s == 2:
                    d12 = val
                elif s == 3:
                    d13 = val
                elif s == 4:
                    d14 = val
                else:
                    d15 = val
            elif r == 2:
                if s == 0:
                    d20 = val
                elif s == 1:
                    d21 = val
                elif s == 2:
                    d22 = val
                elif s == 3:
                    d23 = val
                elif s == 4:
                    d24 = val
                else:
                    d25 = val
            elif r == 3:
                if s == 0:
                    d30 = val
                elif s == 1:
                    d31 = val
                elif s == 2:
                    d32 = val
                elif s == 3:
                    d33 = val
                elif s == 4:
                    d34 = val
                else:
                    d35 = val
            elif r == 4:
                if s == 0:
                    d40 = val
                elif s == 1:
                    d41 = val
                elif s == 2:
                    d42 = val
                elif s == 3:
                    d43 = val
                elif s == 4:
                    d44 = val
                else:
                    d45 = val
            else:
                if s == 0:
                    d50 = val
                elif s == 1:
                    d51 = val
                elif s == 2:
                    d52 = val
                elif s == 3:
                    d53 = val
                elif s == 4:
                    d54 = val
                else:
                    d55 = val

    # B^T column transform (6x6):
    # B^T = [[ 4,  0, -5,  0,  1,  0],
    #        [ 0, -4, -4,  1,  1,  0],
    #        [ 0,  4, -4, -1,  1,  0],
    #        [ 0, -2, -1,  2,  1,  0],
    #        [ 0,  2, -1, -2,  1,  0],
    #        [ 0,  4,  0, -5,  0,  1]]
    # Apply to each column (compute t[row][col] from d[row][col])
    # Float32 for transform arithmetic to avoid fp16 overflow with multipliers like 4, 5
    d00f = d00.to(tl.float32)
    d01f = d01.to(tl.float32)
    d02f = d02.to(tl.float32)
    d03f = d03.to(tl.float32)
    d04f = d04.to(tl.float32)
    d05f = d05.to(tl.float32)
    d10f = d10.to(tl.float32)
    d11f = d11.to(tl.float32)
    d12f = d12.to(tl.float32)
    d13f = d13.to(tl.float32)
    d14f = d14.to(tl.float32)
    d15f = d15.to(tl.float32)
    d20f = d20.to(tl.float32)
    d21f = d21.to(tl.float32)
    d22f = d22.to(tl.float32)
    d23f = d23.to(tl.float32)
    d24f = d24.to(tl.float32)
    d25f = d25.to(tl.float32)
    d30f = d30.to(tl.float32)
    d31f = d31.to(tl.float32)
    d32f = d32.to(tl.float32)
    d33f = d33.to(tl.float32)
    d34f = d34.to(tl.float32)
    d35f = d35.to(tl.float32)
    d40f = d40.to(tl.float32)
    d41f = d41.to(tl.float32)
    d42f = d42.to(tl.float32)
    d43f = d43.to(tl.float32)
    d44f = d44.to(tl.float32)
    d45f = d45.to(tl.float32)
    d50f = d50.to(tl.float32)
    d51f = d51.to(tl.float32)
    d52f = d52.to(tl.float32)
    d53f = d53.to(tl.float32)
    d54f = d54.to(tl.float32)
    d55f = d55.to(tl.float32)

    # Column transform: for each column s, t[row][s] = B^T @ d[:,s]
    t00 = 4 * d00f - 5 * d20f + d40f
    t01 = 4 * d01f - 5 * d21f + d41f
    t02 = 4 * d02f - 5 * d22f + d42f
    t03 = 4 * d03f - 5 * d23f + d43f
    t04 = 4 * d04f - 5 * d24f + d44f
    t05 = 4 * d05f - 5 * d25f + d45f

    t10 = -4 * d10f - 4 * d20f + d30f + d40f
    t11 = -4 * d11f - 4 * d21f + d31f + d41f
    t12 = -4 * d12f - 4 * d22f + d32f + d42f
    t13 = -4 * d13f - 4 * d23f + d33f + d43f
    t14 = -4 * d14f - 4 * d24f + d34f + d44f
    t15 = -4 * d15f - 4 * d25f + d35f + d45f

    t20 = 4 * d10f - 4 * d20f - d30f + d40f
    t21 = 4 * d11f - 4 * d21f - d31f + d41f
    t22 = 4 * d12f - 4 * d22f - d32f + d42f
    t23 = 4 * d13f - 4 * d23f - d33f + d43f
    t24 = 4 * d14f - 4 * d24f - d34f + d44f
    t25 = 4 * d15f - 4 * d25f - d35f + d45f

    t30 = -2 * d10f - d20f + 2 * d30f + d40f
    t31 = -2 * d11f - d21f + 2 * d31f + d41f
    t32 = -2 * d12f - d22f + 2 * d32f + d42f
    t33 = -2 * d13f - d23f + 2 * d33f + d43f
    t34 = -2 * d14f - d24f + 2 * d34f + d44f
    t35 = -2 * d15f - d25f + 2 * d35f + d45f

    t40 = 2 * d10f - d20f - 2 * d30f + d40f
    t41 = 2 * d11f - d21f - 2 * d31f + d41f
    t42 = 2 * d12f - d22f - 2 * d32f + d42f
    t43 = 2 * d13f - d23f - 2 * d33f + d43f
    t44 = 2 * d14f - d24f - 2 * d34f + d44f
    t45 = 2 * d15f - d25f - 2 * d35f + d45f

    t50 = 4 * d10f - 5 * d30f + d50f
    t51 = 4 * d11f - 5 * d31f + d51f
    t52 = 4 * d12f - 5 * d32f + d52f
    t53 = 4 * d13f - 5 * d33f + d53f
    t54 = 4 * d14f - 5 * d34f + d54f
    t55 = 4 * d15f - 5 * d35f + d55f

    # Row transform: v[r][col] = B^T applied to row t[r][:]
    v00 = 4 * t00 - 5 * t02 + t04
    v01 = -4 * t01 - 4 * t02 + t03 + t04
    v02 = 4 * t01 - 4 * t02 - t03 + t04
    v03 = -2 * t01 - t02 + 2 * t03 + t04
    v04 = 2 * t01 - t02 - 2 * t03 + t04
    v05 = 4 * t01 - 5 * t03 + t05

    v10 = 4 * t10 - 5 * t12 + t14
    v11 = -4 * t11 - 4 * t12 + t13 + t14
    v12 = 4 * t11 - 4 * t12 - t13 + t14
    v13 = -2 * t11 - t12 + 2 * t13 + t14
    v14 = 2 * t11 - t12 - 2 * t13 + t14
    v15 = 4 * t11 - 5 * t13 + t15

    v20 = 4 * t20 - 5 * t22 + t24
    v21 = -4 * t21 - 4 * t22 + t23 + t24
    v22 = 4 * t21 - 4 * t22 - t23 + t24
    v23 = -2 * t21 - t22 + 2 * t23 + t24
    v24 = 2 * t21 - t22 - 2 * t23 + t24
    v25 = 4 * t21 - 5 * t23 + t25

    v30 = 4 * t30 - 5 * t32 + t34
    v31 = -4 * t31 - 4 * t32 + t33 + t34
    v32 = 4 * t31 - 4 * t32 - t33 + t34
    v33 = -2 * t31 - t32 + 2 * t33 + t34
    v34 = 2 * t31 - t32 - 2 * t33 + t34
    v35 = 4 * t31 - 5 * t33 + t35

    v40 = 4 * t40 - 5 * t42 + t44
    v41 = -4 * t41 - 4 * t42 + t43 + t44
    v42 = 4 * t41 - 4 * t42 - t43 + t44
    v43 = -2 * t41 - t42 + 2 * t43 + t44
    v44 = 2 * t41 - t42 - 2 * t43 + t44
    v45 = 4 * t41 - 5 * t43 + t45

    v50 = 4 * t50 - 5 * t52 + t54
    v51 = -4 * t51 - 4 * t52 + t53 + t54
    v52 = 4 * t51 - 4 * t52 - t53 + t54
    v53 = -2 * t51 - t52 + 2 * t53 + t54
    v54 = 2 * t51 - t52 - 2 * t53 + t54
    v55 = 4 * t51 - 5 * t53 + t55

    v_base = V + tile_idx * stride_v_tile + offs_c * stride_v_c
    c_store_mask = offs_c < C_pad

    tl.store(v_base + 0 * stride_v_alpha, v00, mask=c_store_mask)
    tl.store(v_base + 1 * stride_v_alpha, v01, mask=c_store_mask)
    tl.store(v_base + 2 * stride_v_alpha, v02, mask=c_store_mask)
    tl.store(v_base + 3 * stride_v_alpha, v03, mask=c_store_mask)
    tl.store(v_base + 4 * stride_v_alpha, v04, mask=c_store_mask)
    tl.store(v_base + 5 * stride_v_alpha, v05, mask=c_store_mask)
    tl.store(v_base + 6 * stride_v_alpha, v10, mask=c_store_mask)
    tl.store(v_base + 7 * stride_v_alpha, v11, mask=c_store_mask)
    tl.store(v_base + 8 * stride_v_alpha, v12, mask=c_store_mask)
    tl.store(v_base + 9 * stride_v_alpha, v13, mask=c_store_mask)
    tl.store(v_base + 10 * stride_v_alpha, v14, mask=c_store_mask)
    tl.store(v_base + 11 * stride_v_alpha, v15, mask=c_store_mask)
    tl.store(v_base + 12 * stride_v_alpha, v20, mask=c_store_mask)
    tl.store(v_base + 13 * stride_v_alpha, v21, mask=c_store_mask)
    tl.store(v_base + 14 * stride_v_alpha, v22, mask=c_store_mask)
    tl.store(v_base + 15 * stride_v_alpha, v23, mask=c_store_mask)
    tl.store(v_base + 16 * stride_v_alpha, v24, mask=c_store_mask)
    tl.store(v_base + 17 * stride_v_alpha, v25, mask=c_store_mask)
    tl.store(v_base + 18 * stride_v_alpha, v30, mask=c_store_mask)
    tl.store(v_base + 19 * stride_v_alpha, v31, mask=c_store_mask)
    tl.store(v_base + 20 * stride_v_alpha, v32, mask=c_store_mask)
    tl.store(v_base + 21 * stride_v_alpha, v33, mask=c_store_mask)
    tl.store(v_base + 22 * stride_v_alpha, v34, mask=c_store_mask)
    tl.store(v_base + 23 * stride_v_alpha, v35, mask=c_store_mask)
    tl.store(v_base + 24 * stride_v_alpha, v40, mask=c_store_mask)
    tl.store(v_base + 25 * stride_v_alpha, v41, mask=c_store_mask)
    tl.store(v_base + 26 * stride_v_alpha, v42, mask=c_store_mask)
    tl.store(v_base + 27 * stride_v_alpha, v43, mask=c_store_mask)
    tl.store(v_base + 28 * stride_v_alpha, v44, mask=c_store_mask)
    tl.store(v_base + 29 * stride_v_alpha, v45, mask=c_store_mask)
    tl.store(v_base + 30 * stride_v_alpha, v50, mask=c_store_mask)
    tl.store(v_base + 31 * stride_v_alpha, v51, mask=c_store_mask)
    tl.store(v_base + 32 * stride_v_alpha, v52, mask=c_store_mask)
    tl.store(v_base + 33 * stride_v_alpha, v53, mask=c_store_mask)
    tl.store(v_base + 34 * stride_v_alpha, v54, mask=c_store_mask)
    tl.store(v_base + 35 * stride_v_alpha, v55, mask=c_store_mask)


@triton.autotune(
    configs=AUTOTUNE_WINO4_INPUT_CONFIGS,
    key=["T", "C_pad"],
    cache_results=True,
)
@triton.jit
def _winograd_f4x3_cblocked_input_transform_kernel(
    X,
    V,
    N: tl.constexpr,
    C: tl.constexpr,
    C_pad: tl.constexpr,
    H: tl.constexpr,
    W_in: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    Cb: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_DTYPE: tl.constexpr = tl.float16,
):
    # X layout: [N, C_blocks, H, W_in, Cb] where C_blocks = C_pad // Cb
    stride_x_w: tl.constexpr = Cb
    stride_x_h: tl.constexpr = W_in * Cb
    stride_x_cblock: tl.constexpr = H * W_in * Cb
    stride_x_n: tl.constexpr = (C_pad // Cb) * H * W_in * Cb
    # V layout: [36, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad

    tile_idx = tl.program_id(0)
    c_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    h_start = th * 4 - pad_h
    w_start = tw * 4 - pad_w

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    # NCHWc addressing: cblock_idx = offs_c // Cb, c_local = offs_c % Cb
    cblock_idx = offs_c // Cb
    c_local = offs_c % Cb
    base = X + n * stride_x_n + cblock_idx * stride_x_cblock + c_local
    n_valid = n < N

    # Load 6x6 patch — 36 values per channel
    d00 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d01 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d02 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d03 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d04 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d05 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d10 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d11 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d12 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d13 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d14 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d15 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d20 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d21 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d22 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d23 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d24 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d25 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d30 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d31 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d32 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d33 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d34 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d35 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d40 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d41 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d42 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d43 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d44 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d45 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d50 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d51 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d52 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d53 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d54 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)
    d55 = tl.zeros((BLOCK_C,), dtype=INPUT_DTYPE)

    for r in tl.static_range(6):
        h = h_start + r
        h_valid = n_valid & (h >= 0) & (h < H)
        for s in tl.static_range(6):
            w = w_start + s
            valid = h_valid & (w >= 0) & (w < W_in)
            ptr = base + h * stride_x_h + w * stride_x_w
            val = tl.load(ptr, mask=valid & c_mask, other=0.0)
            if r == 0:
                if s == 0:
                    d00 = val
                elif s == 1:
                    d01 = val
                elif s == 2:
                    d02 = val
                elif s == 3:
                    d03 = val
                elif s == 4:
                    d04 = val
                else:
                    d05 = val
            elif r == 1:
                if s == 0:
                    d10 = val
                elif s == 1:
                    d11 = val
                elif s == 2:
                    d12 = val
                elif s == 3:
                    d13 = val
                elif s == 4:
                    d14 = val
                else:
                    d15 = val
            elif r == 2:
                if s == 0:
                    d20 = val
                elif s == 1:
                    d21 = val
                elif s == 2:
                    d22 = val
                elif s == 3:
                    d23 = val
                elif s == 4:
                    d24 = val
                else:
                    d25 = val
            elif r == 3:
                if s == 0:
                    d30 = val
                elif s == 1:
                    d31 = val
                elif s == 2:
                    d32 = val
                elif s == 3:
                    d33 = val
                elif s == 4:
                    d34 = val
                else:
                    d35 = val
            elif r == 4:
                if s == 0:
                    d40 = val
                elif s == 1:
                    d41 = val
                elif s == 2:
                    d42 = val
                elif s == 3:
                    d43 = val
                elif s == 4:
                    d44 = val
                else:
                    d45 = val
            else:
                if s == 0:
                    d50 = val
                elif s == 1:
                    d51 = val
                elif s == 2:
                    d52 = val
                elif s == 3:
                    d53 = val
                elif s == 4:
                    d54 = val
                else:
                    d55 = val

    d00f = d00.to(tl.float32)
    d01f = d01.to(tl.float32)
    d02f = d02.to(tl.float32)
    d03f = d03.to(tl.float32)
    d04f = d04.to(tl.float32)
    d05f = d05.to(tl.float32)
    d10f = d10.to(tl.float32)
    d11f = d11.to(tl.float32)
    d12f = d12.to(tl.float32)
    d13f = d13.to(tl.float32)
    d14f = d14.to(tl.float32)
    d15f = d15.to(tl.float32)
    d20f = d20.to(tl.float32)
    d21f = d21.to(tl.float32)
    d22f = d22.to(tl.float32)
    d23f = d23.to(tl.float32)
    d24f = d24.to(tl.float32)
    d25f = d25.to(tl.float32)
    d30f = d30.to(tl.float32)
    d31f = d31.to(tl.float32)
    d32f = d32.to(tl.float32)
    d33f = d33.to(tl.float32)
    d34f = d34.to(tl.float32)
    d35f = d35.to(tl.float32)
    d40f = d40.to(tl.float32)
    d41f = d41.to(tl.float32)
    d42f = d42.to(tl.float32)
    d43f = d43.to(tl.float32)
    d44f = d44.to(tl.float32)
    d45f = d45.to(tl.float32)
    d50f = d50.to(tl.float32)
    d51f = d51.to(tl.float32)
    d52f = d52.to(tl.float32)
    d53f = d53.to(tl.float32)
    d54f = d54.to(tl.float32)
    d55f = d55.to(tl.float32)

    t00 = 4 * d00f - 5 * d20f + d40f
    t01 = 4 * d01f - 5 * d21f + d41f
    t02 = 4 * d02f - 5 * d22f + d42f
    t03 = 4 * d03f - 5 * d23f + d43f
    t04 = 4 * d04f - 5 * d24f + d44f
    t05 = 4 * d05f - 5 * d25f + d45f

    t10 = -4 * d10f - 4 * d20f + d30f + d40f
    t11 = -4 * d11f - 4 * d21f + d31f + d41f
    t12 = -4 * d12f - 4 * d22f + d32f + d42f
    t13 = -4 * d13f - 4 * d23f + d33f + d43f
    t14 = -4 * d14f - 4 * d24f + d34f + d44f
    t15 = -4 * d15f - 4 * d25f + d35f + d45f

    t20 = 4 * d10f - 4 * d20f - d30f + d40f
    t21 = 4 * d11f - 4 * d21f - d31f + d41f
    t22 = 4 * d12f - 4 * d22f - d32f + d42f
    t23 = 4 * d13f - 4 * d23f - d33f + d43f
    t24 = 4 * d14f - 4 * d24f - d34f + d44f
    t25 = 4 * d15f - 4 * d25f - d35f + d45f

    t30 = -2 * d10f - d20f + 2 * d30f + d40f
    t31 = -2 * d11f - d21f + 2 * d31f + d41f
    t32 = -2 * d12f - d22f + 2 * d32f + d42f
    t33 = -2 * d13f - d23f + 2 * d33f + d43f
    t34 = -2 * d14f - d24f + 2 * d34f + d44f
    t35 = -2 * d15f - d25f + 2 * d35f + d45f

    t40 = 2 * d10f - d20f - 2 * d30f + d40f
    t41 = 2 * d11f - d21f - 2 * d31f + d41f
    t42 = 2 * d12f - d22f - 2 * d32f + d42f
    t43 = 2 * d13f - d23f - 2 * d33f + d43f
    t44 = 2 * d14f - d24f - 2 * d34f + d44f
    t45 = 2 * d15f - d25f - 2 * d35f + d45f

    t50 = 4 * d10f - 5 * d30f + d50f
    t51 = 4 * d11f - 5 * d31f + d51f
    t52 = 4 * d12f - 5 * d32f + d52f
    t53 = 4 * d13f - 5 * d33f + d53f
    t54 = 4 * d14f - 5 * d34f + d54f
    t55 = 4 * d15f - 5 * d35f + d55f

    v00 = 4 * t00 - 5 * t02 + t04
    v01 = -4 * t01 - 4 * t02 + t03 + t04
    v02 = 4 * t01 - 4 * t02 - t03 + t04
    v03 = -2 * t01 - t02 + 2 * t03 + t04
    v04 = 2 * t01 - t02 - 2 * t03 + t04
    v05 = 4 * t01 - 5 * t03 + t05

    v10 = 4 * t10 - 5 * t12 + t14
    v11 = -4 * t11 - 4 * t12 + t13 + t14
    v12 = 4 * t11 - 4 * t12 - t13 + t14
    v13 = -2 * t11 - t12 + 2 * t13 + t14
    v14 = 2 * t11 - t12 - 2 * t13 + t14
    v15 = 4 * t11 - 5 * t13 + t15

    v20 = 4 * t20 - 5 * t22 + t24
    v21 = -4 * t21 - 4 * t22 + t23 + t24
    v22 = 4 * t21 - 4 * t22 - t23 + t24
    v23 = -2 * t21 - t22 + 2 * t23 + t24
    v24 = 2 * t21 - t22 - 2 * t23 + t24
    v25 = 4 * t21 - 5 * t23 + t25

    v30 = 4 * t30 - 5 * t32 + t34
    v31 = -4 * t31 - 4 * t32 + t33 + t34
    v32 = 4 * t31 - 4 * t32 - t33 + t34
    v33 = -2 * t31 - t32 + 2 * t33 + t34
    v34 = 2 * t31 - t32 - 2 * t33 + t34
    v35 = 4 * t31 - 5 * t33 + t35

    v40 = 4 * t40 - 5 * t42 + t44
    v41 = -4 * t41 - 4 * t42 + t43 + t44
    v42 = 4 * t41 - 4 * t42 - t43 + t44
    v43 = -2 * t41 - t42 + 2 * t43 + t44
    v44 = 2 * t41 - t42 - 2 * t43 + t44
    v45 = 4 * t41 - 5 * t43 + t45

    v50 = 4 * t50 - 5 * t52 + t54
    v51 = -4 * t51 - 4 * t52 + t53 + t54
    v52 = 4 * t51 - 4 * t52 - t53 + t54
    v53 = -2 * t51 - t52 + 2 * t53 + t54
    v54 = 2 * t51 - t52 - 2 * t53 + t54
    v55 = 4 * t51 - 5 * t53 + t55

    v_base = V + tile_idx * stride_v_tile + offs_c * stride_v_c
    c_store_mask = offs_c < C_pad

    tl.store(v_base + 0 * stride_v_alpha, v00, mask=c_store_mask)
    tl.store(v_base + 1 * stride_v_alpha, v01, mask=c_store_mask)
    tl.store(v_base + 2 * stride_v_alpha, v02, mask=c_store_mask)
    tl.store(v_base + 3 * stride_v_alpha, v03, mask=c_store_mask)
    tl.store(v_base + 4 * stride_v_alpha, v04, mask=c_store_mask)
    tl.store(v_base + 5 * stride_v_alpha, v05, mask=c_store_mask)
    tl.store(v_base + 6 * stride_v_alpha, v10, mask=c_store_mask)
    tl.store(v_base + 7 * stride_v_alpha, v11, mask=c_store_mask)
    tl.store(v_base + 8 * stride_v_alpha, v12, mask=c_store_mask)
    tl.store(v_base + 9 * stride_v_alpha, v13, mask=c_store_mask)
    tl.store(v_base + 10 * stride_v_alpha, v14, mask=c_store_mask)
    tl.store(v_base + 11 * stride_v_alpha, v15, mask=c_store_mask)
    tl.store(v_base + 12 * stride_v_alpha, v20, mask=c_store_mask)
    tl.store(v_base + 13 * stride_v_alpha, v21, mask=c_store_mask)
    tl.store(v_base + 14 * stride_v_alpha, v22, mask=c_store_mask)
    tl.store(v_base + 15 * stride_v_alpha, v23, mask=c_store_mask)
    tl.store(v_base + 16 * stride_v_alpha, v24, mask=c_store_mask)
    tl.store(v_base + 17 * stride_v_alpha, v25, mask=c_store_mask)
    tl.store(v_base + 18 * stride_v_alpha, v30, mask=c_store_mask)
    tl.store(v_base + 19 * stride_v_alpha, v31, mask=c_store_mask)
    tl.store(v_base + 20 * stride_v_alpha, v32, mask=c_store_mask)
    tl.store(v_base + 21 * stride_v_alpha, v33, mask=c_store_mask)
    tl.store(v_base + 22 * stride_v_alpha, v34, mask=c_store_mask)
    tl.store(v_base + 23 * stride_v_alpha, v35, mask=c_store_mask)
    tl.store(v_base + 24 * stride_v_alpha, v40, mask=c_store_mask)
    tl.store(v_base + 25 * stride_v_alpha, v41, mask=c_store_mask)
    tl.store(v_base + 26 * stride_v_alpha, v42, mask=c_store_mask)
    tl.store(v_base + 27 * stride_v_alpha, v43, mask=c_store_mask)
    tl.store(v_base + 28 * stride_v_alpha, v44, mask=c_store_mask)
    tl.store(v_base + 29 * stride_v_alpha, v45, mask=c_store_mask)
    tl.store(v_base + 30 * stride_v_alpha, v50, mask=c_store_mask)
    tl.store(v_base + 31 * stride_v_alpha, v51, mask=c_store_mask)
    tl.store(v_base + 32 * stride_v_alpha, v52, mask=c_store_mask)
    tl.store(v_base + 33 * stride_v_alpha, v53, mask=c_store_mask)
    tl.store(v_base + 34 * stride_v_alpha, v54, mask=c_store_mask)
    tl.store(v_base + 35 * stride_v_alpha, v55, mask=c_store_mask)


@triton.autotune(
    configs=AUTOTUNE_WINO_GEMM_CONFIGS,
    key=["T", "K_out", "C_pad"],
    cache_results=True,
)
@triton.jit
def _winograd_f4x3_batched_gemm_kernel(
    V,
    U,
    M_out,
    T: tl.constexpr,
    K_out: tl.constexpr,
    C_pad: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Batched GEMM: M[alpha] = V[alpha] @ U[alpha]^T, alpha in [0..36)"""
    # V layout: [36, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad
    # U layout: [36, K_out, C_pad] contiguous
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    # M layout: [36, T, K_out] contiguous
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out

    pid = tl.program_id(0)
    alpha = tl.program_id(1)

    num_pid_m = tl.cdiv(T, BLOCK_M)
    num_pid_n = tl.cdiv(K_out, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    v_base = V + alpha * stride_v_alpha
    u_base = U + alpha * stride_u_alpha

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, C_pad, BLOCK_K):
        k_offs = k0 + offs_k

        v_ptrs = v_base + offs_m[:, None] * stride_v_tile + k_offs[None, :] * stride_v_c
        v_mask = (offs_m[:, None] < T) & (k_offs[None, :] < C_pad)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        u_ptrs = u_base + offs_n[:, None] * stride_u_k + k_offs[None, :] * stride_u_c
        u_mask = (offs_n[:, None] < K_out) & (k_offs[None, :] < C_pad)
        u_tile = tl.load(u_ptrs, mask=u_mask, other=0.0)

        acc += tl.dot(v_tile, tl.trans(u_tile), out_dtype=tl.float32)

    m_ptrs = (
        M_out
        + alpha * stride_m_alpha
        + offs_m[:, None] * stride_m_tile
        + offs_n[None, :] * stride_m_k
    )
    m_mask = (offs_m[:, None] < T) & (offs_n[None, :] < K_out)
    tl.store(m_ptrs, acc, mask=m_mask)


@triton.autotune(
    configs=AUTOTUNE_WINO4_OUTPUT_CONFIGS,
    key=["T", "K_out"],
    cache_results=True,
)
@triton.jit
def _winograd_f4x3_output_transform_kernel(
    M_in,
    BIAS,
    Y,
    N: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    LAYOUT: tl.constexpr = 0,
):
    # M layout: [36, T, K_out] contiguous
    stride_m_k: tl.constexpr = 1
    stride_m_tile: tl.constexpr = K_out
    stride_m_alpha: tl.constexpr = T * K_out
    # Y layout: LAYOUT=0 NCHW, LAYOUT=1 NHWC
    if LAYOUT == 0:
        stride_y_q: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q
        stride_y_k: tl.constexpr = P * Q
        stride_y_n: tl.constexpr = K_out * P * Q
    else:
        stride_y_q: tl.constexpr = K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_n: tl.constexpr = P * Q * K_out

    tile_idx = tl.program_id(0)
    k_block = tl.program_id(1)

    n = tile_idx // (tile_H * tile_W)
    rem = tile_idx % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W

    p_start = th * 4
    q_start = tw * 4

    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K_out

    # Load 36 values from M[alpha, tile_idx, k]
    m_base = M_in + tile_idx * stride_m_tile + offs_k * stride_m_k

    m00 = tl.load(m_base + 0 * stride_m_alpha, mask=k_mask, other=0.0)
    m01 = tl.load(m_base + 1 * stride_m_alpha, mask=k_mask, other=0.0)
    m02 = tl.load(m_base + 2 * stride_m_alpha, mask=k_mask, other=0.0)
    m03 = tl.load(m_base + 3 * stride_m_alpha, mask=k_mask, other=0.0)
    m04 = tl.load(m_base + 4 * stride_m_alpha, mask=k_mask, other=0.0)
    m05 = tl.load(m_base + 5 * stride_m_alpha, mask=k_mask, other=0.0)
    m10 = tl.load(m_base + 6 * stride_m_alpha, mask=k_mask, other=0.0)
    m11 = tl.load(m_base + 7 * stride_m_alpha, mask=k_mask, other=0.0)
    m12 = tl.load(m_base + 8 * stride_m_alpha, mask=k_mask, other=0.0)
    m13 = tl.load(m_base + 9 * stride_m_alpha, mask=k_mask, other=0.0)
    m14 = tl.load(m_base + 10 * stride_m_alpha, mask=k_mask, other=0.0)
    m15 = tl.load(m_base + 11 * stride_m_alpha, mask=k_mask, other=0.0)
    m20 = tl.load(m_base + 12 * stride_m_alpha, mask=k_mask, other=0.0)
    m21 = tl.load(m_base + 13 * stride_m_alpha, mask=k_mask, other=0.0)
    m22 = tl.load(m_base + 14 * stride_m_alpha, mask=k_mask, other=0.0)
    m23 = tl.load(m_base + 15 * stride_m_alpha, mask=k_mask, other=0.0)
    m24 = tl.load(m_base + 16 * stride_m_alpha, mask=k_mask, other=0.0)
    m25 = tl.load(m_base + 17 * stride_m_alpha, mask=k_mask, other=0.0)
    m30 = tl.load(m_base + 18 * stride_m_alpha, mask=k_mask, other=0.0)
    m31 = tl.load(m_base + 19 * stride_m_alpha, mask=k_mask, other=0.0)
    m32 = tl.load(m_base + 20 * stride_m_alpha, mask=k_mask, other=0.0)
    m33 = tl.load(m_base + 21 * stride_m_alpha, mask=k_mask, other=0.0)
    m34 = tl.load(m_base + 22 * stride_m_alpha, mask=k_mask, other=0.0)
    m35 = tl.load(m_base + 23 * stride_m_alpha, mask=k_mask, other=0.0)
    m40 = tl.load(m_base + 24 * stride_m_alpha, mask=k_mask, other=0.0)
    m41 = tl.load(m_base + 25 * stride_m_alpha, mask=k_mask, other=0.0)
    m42 = tl.load(m_base + 26 * stride_m_alpha, mask=k_mask, other=0.0)
    m43 = tl.load(m_base + 27 * stride_m_alpha, mask=k_mask, other=0.0)
    m44 = tl.load(m_base + 28 * stride_m_alpha, mask=k_mask, other=0.0)
    m45 = tl.load(m_base + 29 * stride_m_alpha, mask=k_mask, other=0.0)
    m50 = tl.load(m_base + 30 * stride_m_alpha, mask=k_mask, other=0.0)
    m51 = tl.load(m_base + 31 * stride_m_alpha, mask=k_mask, other=0.0)
    m52 = tl.load(m_base + 32 * stride_m_alpha, mask=k_mask, other=0.0)
    m53 = tl.load(m_base + 33 * stride_m_alpha, mask=k_mask, other=0.0)
    m54 = tl.load(m_base + 34 * stride_m_alpha, mask=k_mask, other=0.0)
    m55 = tl.load(m_base + 35 * stride_m_alpha, mask=k_mask, other=0.0)

    # A^T column transform (4x6):
    # A^T = [[ 1,  1,  1,  1,  1,  0],
    #        [ 0,  1, -1,  2, -2,  0],
    #        [ 0,  1,  1,  4,  4,  0],
    #        [ 0,  1, -1,  8, -8,  1]]
    # Apply to each column s: s_col = A^T @ m_col

    # Column 0
    s00 = m00 + m10 + m20 + m30 + m40
    s10 = m10 - m20 + 2 * m30 - 2 * m40
    s20 = m10 + m20 + 4 * m30 + 4 * m40
    s30 = m10 - m20 + 8 * m30 - 8 * m40 + m50
    # Column 1
    s01 = m01 + m11 + m21 + m31 + m41
    s11 = m11 - m21 + 2 * m31 - 2 * m41
    s21 = m11 + m21 + 4 * m31 + 4 * m41
    s31 = m11 - m21 + 8 * m31 - 8 * m41 + m51
    # Column 2
    s02 = m02 + m12 + m22 + m32 + m42
    s12 = m12 - m22 + 2 * m32 - 2 * m42
    s22 = m12 + m22 + 4 * m32 + 4 * m42
    s32 = m12 - m22 + 8 * m32 - 8 * m42 + m52
    # Column 3
    s03 = m03 + m13 + m23 + m33 + m43
    s13 = m13 - m23 + 2 * m33 - 2 * m43
    s23 = m13 + m23 + 4 * m33 + 4 * m43
    s33 = m13 - m23 + 8 * m33 - 8 * m43 + m53
    # Column 4
    s04 = m04 + m14 + m24 + m34 + m44
    s14 = m14 - m24 + 2 * m34 - 2 * m44
    s24 = m14 + m24 + 4 * m34 + 4 * m44
    s34 = m14 - m24 + 8 * m34 - 8 * m44 + m54
    # Column 5
    s05 = m05 + m15 + m25 + m35 + m45
    s15 = m15 - m25 + 2 * m35 - 2 * m45
    s25 = m15 + m25 + 4 * m35 + 4 * m45
    s35 = m15 - m25 + 8 * m35 - 8 * m45 + m55

    # A^T row transform
    y00 = s00 + s01 + s02 + s03 + s04
    y01 = s01 - s02 + 2 * s03 - 2 * s04
    y02 = s01 + s02 + 4 * s03 + 4 * s04
    y03 = s01 - s02 + 8 * s03 - 8 * s04 + s05

    y10 = s10 + s11 + s12 + s13 + s14
    y11 = s11 - s12 + 2 * s13 - 2 * s14
    y12 = s11 + s12 + 4 * s13 + 4 * s14
    y13 = s11 - s12 + 8 * s13 - 8 * s14 + s15

    y20 = s20 + s21 + s22 + s23 + s24
    y21 = s21 - s22 + 2 * s23 - 2 * s24
    y22 = s21 + s22 + 4 * s23 + 4 * s24
    y23 = s21 - s22 + 8 * s23 - 8 * s24 + s25

    y30 = s30 + s31 + s32 + s33 + s34
    y31 = s31 - s32 + 2 * s33 - 2 * s34
    y32 = s31 + s32 + 4 * s33 + 4 * s34
    y33 = s31 - s32 + 8 * s33 - 8 * s34 + s35

    # Bias
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_k, mask=k_mask, other=0.0)
        y00 += bias
        y01 += bias
        y02 += bias
        y03 += bias
        y10 += bias
        y11 += bias
        y12 += bias
        y13 += bias
        y20 += bias
        y21 += bias
        y22 += bias
        y23 += bias
        y30 += bias
        y31 += bias
        y32 += bias
        y33 += bias

    # Activation
    if ACT_TYPE == 1:
        y00 = tl.maximum(y00, 0)
        y01 = tl.maximum(y01, 0)
        y02 = tl.maximum(y02, 0)
        y03 = tl.maximum(y03, 0)
        y10 = tl.maximum(y10, 0)
        y11 = tl.maximum(y11, 0)
        y12 = tl.maximum(y12, 0)
        y13 = tl.maximum(y13, 0)
        y20 = tl.maximum(y20, 0)
        y21 = tl.maximum(y21, 0)
        y22 = tl.maximum(y22, 0)
        y23 = tl.maximum(y23, 0)
        y30 = tl.maximum(y30, 0)
        y31 = tl.maximum(y31, 0)
        y32 = tl.maximum(y32, 0)
        y33 = tl.maximum(y33, 0)
    elif ACT_TYPE == 2:
        y00 = tl.minimum(tl.maximum(y00, 0), 6)
        y01 = tl.minimum(tl.maximum(y01, 0), 6)
        y02 = tl.minimum(tl.maximum(y02, 0), 6)
        y03 = tl.minimum(tl.maximum(y03, 0), 6)
        y10 = tl.minimum(tl.maximum(y10, 0), 6)
        y11 = tl.minimum(tl.maximum(y11, 0), 6)
        y12 = tl.minimum(tl.maximum(y12, 0), 6)
        y13 = tl.minimum(tl.maximum(y13, 0), 6)
        y20 = tl.minimum(tl.maximum(y20, 0), 6)
        y21 = tl.minimum(tl.maximum(y21, 0), 6)
        y22 = tl.minimum(tl.maximum(y22, 0), 6)
        y23 = tl.minimum(tl.maximum(y23, 0), 6)
        y30 = tl.minimum(tl.maximum(y30, 0), 6)
        y31 = tl.minimum(tl.maximum(y31, 0), 6)
        y32 = tl.minimum(tl.maximum(y32, 0), 6)
        y33 = tl.minimum(tl.maximum(y33, 0), 6)
    elif ACT_TYPE == 3:
        y00 = (
            0.5 * y00 * (1.0 + _tanh(0.7978845608 * (y00 + 0.044715 * y00 * y00 * y00)))
        )
        y01 = (
            0.5 * y01 * (1.0 + _tanh(0.7978845608 * (y01 + 0.044715 * y01 * y01 * y01)))
        )
        y02 = (
            0.5 * y02 * (1.0 + _tanh(0.7978845608 * (y02 + 0.044715 * y02 * y02 * y02)))
        )
        y03 = (
            0.5 * y03 * (1.0 + _tanh(0.7978845608 * (y03 + 0.044715 * y03 * y03 * y03)))
        )
        y10 = (
            0.5 * y10 * (1.0 + _tanh(0.7978845608 * (y10 + 0.044715 * y10 * y10 * y10)))
        )
        y11 = (
            0.5 * y11 * (1.0 + _tanh(0.7978845608 * (y11 + 0.044715 * y11 * y11 * y11)))
        )
        y12 = (
            0.5 * y12 * (1.0 + _tanh(0.7978845608 * (y12 + 0.044715 * y12 * y12 * y12)))
        )
        y13 = (
            0.5 * y13 * (1.0 + _tanh(0.7978845608 * (y13 + 0.044715 * y13 * y13 * y13)))
        )
        y20 = (
            0.5 * y20 * (1.0 + _tanh(0.7978845608 * (y20 + 0.044715 * y20 * y20 * y20)))
        )
        y21 = (
            0.5 * y21 * (1.0 + _tanh(0.7978845608 * (y21 + 0.044715 * y21 * y21 * y21)))
        )
        y22 = (
            0.5 * y22 * (1.0 + _tanh(0.7978845608 * (y22 + 0.044715 * y22 * y22 * y22)))
        )
        y23 = (
            0.5 * y23 * (1.0 + _tanh(0.7978845608 * (y23 + 0.044715 * y23 * y23 * y23)))
        )
        y30 = (
            0.5 * y30 * (1.0 + _tanh(0.7978845608 * (y30 + 0.044715 * y30 * y30 * y30)))
        )
        y31 = (
            0.5 * y31 * (1.0 + _tanh(0.7978845608 * (y31 + 0.044715 * y31 * y31 * y31)))
        )
        y32 = (
            0.5 * y32 * (1.0 + _tanh(0.7978845608 * (y32 + 0.044715 * y32 * y32 * y32)))
        )
        y33 = (
            0.5 * y33 * (1.0 + _tanh(0.7978845608 * (y33 + 0.044715 * y33 * y33 * y33)))
        )

    # Store 4x4 output tile
    n_valid = n < N
    y_base = Y + n * stride_y_n + offs_k * stride_y_k

    if n_valid:
        for r in tl.static_range(4):
            p = p_start + r
            if p < P:
                for s in tl.static_range(4):
                    q = q_start + s
                    if q < Q:
                        if r == 0:
                            if s == 0:
                                val = y00
                            elif s == 1:
                                val = y01
                            elif s == 2:
                                val = y02
                            else:
                                val = y03
                        elif r == 1:
                            if s == 0:
                                val = y10
                            elif s == 1:
                                val = y11
                            elif s == 2:
                                val = y12
                            else:
                                val = y13
                        elif r == 2:
                            if s == 0:
                                val = y20
                            elif s == 1:
                                val = y21
                            elif s == 2:
                                val = y22
                            else:
                                val = y23
                        else:
                            if s == 0:
                                val = y30
                            elif s == 1:
                                val = y31
                            elif s == 2:
                                val = y32
                            else:
                                val = y33
                        tl.store(
                            y_base + p * stride_y_p + q * stride_y_q, val, mask=k_mask
                        )


@triton.autotune(
    configs=AUTOTUNE_FUSED_F4X3_CONFIGS,
    key=["T", "K_out", "C_pad"],
    cache_results=True,
)
@triton.jit
def _winograd_f4x3_fused_gemm_output_kernel(
    V,
    U,
    BIAS,
    Y,
    N: tl.constexpr,
    K_out: tl.constexpr,
    P: tl.constexpr,
    Q: tl.constexpr,
    C_pad: tl.constexpr,
    tile_H: tl.constexpr,
    tile_W: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    ACT_TYPE: tl.constexpr,
    LAYOUT: tl.constexpr = 0,
):
    """Fused GEMM + output transform for Winograd F(4x4,3x3).
    Processes 6 alphas per column (6 columns = 36 total) sequentially,
    accumulating column-transform results to reduce register pressure."""
    # V layout: [36, T, C_pad] contiguous
    stride_v_c: tl.constexpr = 1
    stride_v_tile: tl.constexpr = C_pad
    stride_v_alpha: tl.constexpr = T * C_pad
    # U layout: [36, K_out, C_pad] contiguous
    stride_u_c: tl.constexpr = 1
    stride_u_k: tl.constexpr = C_pad
    stride_u_alpha: tl.constexpr = K_out * C_pad
    # Y layout: LAYOUT=0 NCHW, LAYOUT=1 NHWC
    if LAYOUT == 0:
        stride_y_q: tl.constexpr = 1
        stride_y_p: tl.constexpr = Q
        stride_y_k: tl.constexpr = P * Q
        stride_y_n: tl.constexpr = K_out * P * Q
    else:
        stride_y_q: tl.constexpr = K_out
        stride_y_p: tl.constexpr = Q * K_out
        stride_y_k: tl.constexpr = 1
        stride_y_n: tl.constexpr = P * Q * K_out

    pid_t = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    t_mask = offs_t < T
    k_mask = offs_k < K_out

    offs_c = tl.arange(0, BLOCK_C)

    # Process column by column to reduce register pressure.
    # For each column s (0..5), accumulate 6 alpha GEMMs (rows 0..5),
    # then apply A^T column transform (6→4), storing s[row][col] results.
    # A^T = [[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]
    # We need s[0..3][0..5] = 24 accumulators after column transform.
    # Then row transform: s[row][0..5] → y[row][0..3] = 16 output values.

    # Accumulate all 24 s-values (4 rows × 6 columns)
    s00 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s01 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s02 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s03 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s04 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s05 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s10 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s11 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s12 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s13 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s14 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s15 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s20 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s21 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s22 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s23 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s24 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s25 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s30 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s31 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s32 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s33 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s34 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)
    s35 = tl.zeros((BLOCK_T, BLOCK_K), dtype=tl.float32)

    v_mask_2d = t_mask[:, None] & (offs_c < C_pad)[None, :]
    u_mask_2d = k_mask[:, None] & (offs_c < C_pad)[None, :]

    for c0 in range(0, C_pad, BLOCK_C):
        c_offs = c0 + offs_c
        if c0 > 0:
            v_mask_2d = t_mask[:, None] & (c_offs < C_pad)[None, :]
            u_mask_2d = k_mask[:, None] & (c_offs < C_pad)[None, :]

        # Process all 36 alphas, applying column transform on-the-fly
        for col in tl.static_range(6):
            # Load 6 V tiles (rows 0-5 for this column) and 6 U tiles
            # alpha = row * 6 + col
            v_base_c = (
                V + offs_t[:, None] * stride_v_tile + c_offs[None, :] * stride_v_c
            )
            u_base_c = U + offs_k[:, None] * stride_u_k + c_offs[None, :] * stride_u_c

            # Load V and U for 6 rows, compute GEMM, apply column transform
            v0 = tl.load(
                v_base_c + (0 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u0 = tl.load(
                u_base_c + (0 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m0 = tl.dot(v0, tl.trans(u0), out_dtype=tl.float32)

            v1 = tl.load(
                v_base_c + (1 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u1 = tl.load(
                u_base_c + (1 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m1 = tl.dot(v1, tl.trans(u1), out_dtype=tl.float32)

            v2 = tl.load(
                v_base_c + (2 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u2 = tl.load(
                u_base_c + (2 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m2 = tl.dot(v2, tl.trans(u2), out_dtype=tl.float32)

            v3 = tl.load(
                v_base_c + (3 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u3 = tl.load(
                u_base_c + (3 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m3 = tl.dot(v3, tl.trans(u3), out_dtype=tl.float32)

            v4 = tl.load(
                v_base_c + (4 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u4 = tl.load(
                u_base_c + (4 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m4 = tl.dot(v4, tl.trans(u4), out_dtype=tl.float32)

            v5 = tl.load(
                v_base_c + (5 * 6 + col) * stride_v_alpha, mask=v_mask_2d, other=0.0
            )
            u5 = tl.load(
                u_base_c + (5 * 6 + col) * stride_u_alpha, mask=u_mask_2d, other=0.0
            )
            m5 = tl.dot(v5, tl.trans(u5), out_dtype=tl.float32)

            # Column transform: A^T @ [m0..m5] → [s0, s1, s2, s3]
            sc0 = m0 + m1 + m2 + m3 + m4
            sc1 = m1 - m2 + 2 * m3 - 2 * m4
            sc2 = m1 + m2 + 4 * m3 + 4 * m4
            sc3 = m1 - m2 + 8 * m3 - 8 * m4 + m5

            # Accumulate into the appropriate column
            if col == 0:
                s00 += sc0
                s10 += sc1
                s20 += sc2
                s30 += sc3
            elif col == 1:
                s01 += sc0
                s11 += sc1
                s21 += sc2
                s31 += sc3
            elif col == 2:
                s02 += sc0
                s12 += sc1
                s22 += sc2
                s32 += sc3
            elif col == 3:
                s03 += sc0
                s13 += sc1
                s23 += sc2
                s33 += sc3
            elif col == 4:
                s04 += sc0
                s14 += sc1
                s24 += sc2
                s34 += sc3
            elif col == 5:
                s05 += sc0
                s15 += sc1
                s25 += sc2
                s35 += sc3

    # Row transform: A^T applied to rows
    y00 = s00 + s01 + s02 + s03 + s04
    y01 = s01 - s02 + 2 * s03 - 2 * s04
    y02 = s01 + s02 + 4 * s03 + 4 * s04
    y03 = s01 - s02 + 8 * s03 - 8 * s04 + s05

    y10 = s10 + s11 + s12 + s13 + s14
    y11 = s11 - s12 + 2 * s13 - 2 * s14
    y12 = s11 + s12 + 4 * s13 + 4 * s14
    y13 = s11 - s12 + 8 * s13 - 8 * s14 + s15

    y20 = s20 + s21 + s22 + s23 + s24
    y21 = s21 - s22 + 2 * s23 - 2 * s24
    y22 = s21 + s22 + 4 * s23 + 4 * s24
    y23 = s21 - s22 + 8 * s23 - 8 * s24 + s25

    y30 = s30 + s31 + s32 + s33 + s34
    y31 = s31 - s32 + 2 * s33 - 2 * s34
    y32 = s31 + s32 + 4 * s33 + 4 * s34
    y33 = s31 - s32 + 8 * s33 - 8 * s34 + s35

    # Bias
    if HAS_BIAS:
        bias = tl.load(BIAS + offs_k, mask=k_mask, other=0.0)
        b = bias[None, :]
        y00 += b
        y01 += b
        y02 += b
        y03 += b
        y10 += b
        y11 += b
        y12 += b
        y13 += b
        y20 += b
        y21 += b
        y22 += b
        y23 += b
        y30 += b
        y31 += b
        y32 += b
        y33 += b

    # Activation
    if ACT_TYPE == 1:
        y00 = tl.maximum(y00, 0)
        y01 = tl.maximum(y01, 0)
        y02 = tl.maximum(y02, 0)
        y03 = tl.maximum(y03, 0)
        y10 = tl.maximum(y10, 0)
        y11 = tl.maximum(y11, 0)
        y12 = tl.maximum(y12, 0)
        y13 = tl.maximum(y13, 0)
        y20 = tl.maximum(y20, 0)
        y21 = tl.maximum(y21, 0)
        y22 = tl.maximum(y22, 0)
        y23 = tl.maximum(y23, 0)
        y30 = tl.maximum(y30, 0)
        y31 = tl.maximum(y31, 0)
        y32 = tl.maximum(y32, 0)
        y33 = tl.maximum(y33, 0)
    elif ACT_TYPE == 2:
        y00 = tl.minimum(tl.maximum(y00, 0), 6)
        y01 = tl.minimum(tl.maximum(y01, 0), 6)
        y02 = tl.minimum(tl.maximum(y02, 0), 6)
        y03 = tl.minimum(tl.maximum(y03, 0), 6)
        y10 = tl.minimum(tl.maximum(y10, 0), 6)
        y11 = tl.minimum(tl.maximum(y11, 0), 6)
        y12 = tl.minimum(tl.maximum(y12, 0), 6)
        y13 = tl.minimum(tl.maximum(y13, 0), 6)
        y20 = tl.minimum(tl.maximum(y20, 0), 6)
        y21 = tl.minimum(tl.maximum(y21, 0), 6)
        y22 = tl.minimum(tl.maximum(y22, 0), 6)
        y23 = tl.minimum(tl.maximum(y23, 0), 6)
        y30 = tl.minimum(tl.maximum(y30, 0), 6)
        y31 = tl.minimum(tl.maximum(y31, 0), 6)
        y32 = tl.minimum(tl.maximum(y32, 0), 6)
        y33 = tl.minimum(tl.maximum(y33, 0), 6)
    elif ACT_TYPE == 3:
        y00 = (
            0.5 * y00 * (1.0 + _tanh(0.7978845608 * (y00 + 0.044715 * y00 * y00 * y00)))
        )
        y01 = (
            0.5 * y01 * (1.0 + _tanh(0.7978845608 * (y01 + 0.044715 * y01 * y01 * y01)))
        )
        y02 = (
            0.5 * y02 * (1.0 + _tanh(0.7978845608 * (y02 + 0.044715 * y02 * y02 * y02)))
        )
        y03 = (
            0.5 * y03 * (1.0 + _tanh(0.7978845608 * (y03 + 0.044715 * y03 * y03 * y03)))
        )
        y10 = (
            0.5 * y10 * (1.0 + _tanh(0.7978845608 * (y10 + 0.044715 * y10 * y10 * y10)))
        )
        y11 = (
            0.5 * y11 * (1.0 + _tanh(0.7978845608 * (y11 + 0.044715 * y11 * y11 * y11)))
        )
        y12 = (
            0.5 * y12 * (1.0 + _tanh(0.7978845608 * (y12 + 0.044715 * y12 * y12 * y12)))
        )
        y13 = (
            0.5 * y13 * (1.0 + _tanh(0.7978845608 * (y13 + 0.044715 * y13 * y13 * y13)))
        )
        y20 = (
            0.5 * y20 * (1.0 + _tanh(0.7978845608 * (y20 + 0.044715 * y20 * y20 * y20)))
        )
        y21 = (
            0.5 * y21 * (1.0 + _tanh(0.7978845608 * (y21 + 0.044715 * y21 * y21 * y21)))
        )
        y22 = (
            0.5 * y22 * (1.0 + _tanh(0.7978845608 * (y22 + 0.044715 * y22 * y22 * y22)))
        )
        y23 = (
            0.5 * y23 * (1.0 + _tanh(0.7978845608 * (y23 + 0.044715 * y23 * y23 * y23)))
        )
        y30 = (
            0.5 * y30 * (1.0 + _tanh(0.7978845608 * (y30 + 0.044715 * y30 * y30 * y30)))
        )
        y31 = (
            0.5 * y31 * (1.0 + _tanh(0.7978845608 * (y31 + 0.044715 * y31 * y31 * y31)))
        )
        y32 = (
            0.5 * y32 * (1.0 + _tanh(0.7978845608 * (y32 + 0.044715 * y32 * y32 * y32)))
        )
        y33 = (
            0.5 * y33 * (1.0 + _tanh(0.7978845608 * (y33 + 0.044715 * y33 * y33 * y33)))
        )

    # Decode tile indices and compute per-tile base pointers
    n_idx = offs_t // (tile_H * tile_W)
    rem = offs_t % (tile_H * tile_W)
    th = rem // tile_W
    tw = rem % tile_W
    p_start = th * 4
    q_start = tw * 4

    y_base = Y + n_idx * stride_y_n
    y_ptrs_base = y_base[:, None] + offs_k[None, :] * stride_y_k
    full_mask = t_mask[:, None] & k_mask[None, :]

    # Store 4x4 output tile using 2D scatter stores
    for di in tl.static_range(4):
        for dj in tl.static_range(4):
            mask_ij = (
                full_mask & (p_start + di < P)[:, None] & (q_start + dj < Q)[:, None]
            )
            ptrs_ij = (
                y_ptrs_base
                + (p_start[:, None] + di) * stride_y_p
                + (q_start[:, None] + dj) * stride_y_q
            )
            if di == 0:
                if dj == 0:
                    tl.store(ptrs_ij, y00, mask=mask_ij)
                elif dj == 1:
                    tl.store(ptrs_ij, y01, mask=mask_ij)
                elif dj == 2:
                    tl.store(ptrs_ij, y02, mask=mask_ij)
                else:
                    tl.store(ptrs_ij, y03, mask=mask_ij)
            elif di == 1:
                if dj == 0:
                    tl.store(ptrs_ij, y10, mask=mask_ij)
                elif dj == 1:
                    tl.store(ptrs_ij, y11, mask=mask_ij)
                elif dj == 2:
                    tl.store(ptrs_ij, y12, mask=mask_ij)
                else:
                    tl.store(ptrs_ij, y13, mask=mask_ij)
            elif di == 2:
                if dj == 0:
                    tl.store(ptrs_ij, y20, mask=mask_ij)
                elif dj == 1:
                    tl.store(ptrs_ij, y21, mask=mask_ij)
                elif dj == 2:
                    tl.store(ptrs_ij, y22, mask=mask_ij)
                else:
                    tl.store(ptrs_ij, y23, mask=mask_ij)
            else:
                if dj == 0:
                    tl.store(ptrs_ij, y30, mask=mask_ij)
                elif dj == 1:
                    tl.store(ptrs_ij, y31, mask=mask_ij)
                elif dj == 2:
                    tl.store(ptrs_ij, y32, mask=mask_ij)
                else:
                    tl.store(ptrs_ij, y33, mask=mask_ij)
