# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""Gluon block-scaled FP8 GEMM for M=32, N=7168, K=2048 on MI355X."""

from __future__ import annotations

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon


@gluon.jit
def _carry_accumulator(x):
    """Keep the FP32 loop carry ordered without emitting a VGPR move."""
    return gl.inline_asm_elementwise(
        "",
        "=v,0",
        [x],
        dtype=gl.float32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _block_scaled_mm_n7168_k2048_m32(x_ptr, w_ptr, xs_ptr, ws_ptr, out_ptr):
    block_m: gl.constexpr = 32
    block_n: gl.constexpr = 32
    block_k: gl.constexpr = 128

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=False,
        warps_per_cta=[2, 2],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, 16)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, 16)

    copy_a: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [8, 8],
        [4, 1],
        [1, 0],
    )
    copy_b: gl.constexpr = gl.BlockedLayout(
        [16, 1],
        [8, 8],
        [1, 4],
        [0, 1],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 16, [0, 1])

    pid_n = gl.program_id(0)
    a_m = gl.arange(0, block_m, layout=gl.SliceLayout(1, copy_a))
    a_k = gl.arange(0, block_k, layout=gl.SliceLayout(0, copy_a))
    b_k = gl.arange(0, block_k, layout=gl.SliceLayout(1, copy_b))
    b_n = pid_n * block_n + gl.arange(
        0,
        block_n,
        layout=gl.SliceLayout(0, copy_b),
    )

    a_offsets = a_m[:, None] * 2048 + a_k[None, :]
    b_offsets = b_k[:, None] + b_n[None, :] * 2048

    # Separate descriptors preserve async-copy and MFMA ordering on gfx950.
    a0 = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        [block_m, block_k],
        layout=shared_a,
    )
    a1 = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        [block_m, block_k],
        layout=shared_a,
    )
    b0 = gl.allocate_shared_memory(
        w_ptr.type.element_ty,
        [block_k, block_n],
        layout=shared_b,
    )
    b1 = gl.allocate_shared_memory(
        w_ptr.type.element_ty,
        [block_k, block_n],
        layout=shared_b,
    )
    a2 = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        [block_m, block_k],
        layout=shared_a,
    )
    a3 = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        [block_m, block_k],
        layout=shared_a,
    )
    b2 = gl.allocate_shared_memory(
        w_ptr.type.element_ty,
        [block_k, block_n],
        layout=shared_b,
    )
    b3 = gl.allocate_shared_memory(
        w_ptr.type.element_ty,
        [block_k, block_n],
        layout=shared_b,
    )

    next_a = x_ptr
    next_b = w_ptr
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a0,
        next_a,
        a_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b0,
        next_b,
        b_offsets,
        cache_modifier=".ca",
    )
    next_a += block_k
    next_b += block_k
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a1,
        next_a,
        a_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b1,
        next_b,
        b_offsets,
        cache_modifier=".ca",
    )
    next_a += block_k
    next_b += block_k
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a2,
        next_a,
        a_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b2,
        next_b,
        b_offsets,
        cache_modifier=".ca",
    )
    next_a += block_k
    next_b += block_k
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a3,
        next_a,
        a_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b3,
        next_b,
        b_offsets,
        cache_modifier=".ca",
    )
    gl.amd.cdna4.async_copy.commit_group()

    scale_m = gl.arange(0, block_m, layout=gl.SliceLayout(1, mfma_layout))
    scale_group = (pid_n * block_n // 128) * 16
    s0 = gl.load(xs_ptr + scale_m * 16)
    q0 = gl.load(ws_ptr + scale_group)
    s1 = gl.load(xs_ptr + scale_m * 16 + 1)
    q1 = gl.load(ws_ptr + scale_group + 1)
    s2 = gl.load(xs_ptr + scale_m * 16 + 2)
    q2 = gl.load(ws_ptr + scale_group + 2)
    s3 = gl.load(xs_ptr + scale_m * 16 + 3)
    q3 = gl.load(ws_ptr + scale_group + 3)
    gl.amd.cdna4.async_copy.wait_group(0)
    acc = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)

    for turn in gl.static_range(0, 3):
        av0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, layout=dot_a)
        bv0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, layout=dot_b)
        term0 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
        term0 = gl.amd.cdna4.mfma_scaled(
            av0,
            None,
            "e4m3",
            bv0,
            None,
            "e4m3",
            term0,
        )
        av1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, layout=dot_a)
        bv1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, layout=dot_b)
        term1 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
        term1 = gl.amd.cdna4.mfma_scaled(
            av1,
            None,
            "e4m3",
            bv1,
            None,
            "e4m3",
            term1,
        )
        av2 = gl.amd.cdna4.async_copy.load_shared_relaxed(a2, layout=dot_a)
        bv2 = gl.amd.cdna4.async_copy.load_shared_relaxed(b2, layout=dot_b)
        term2 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
        term2 = gl.amd.cdna4.mfma_scaled(
            av2,
            None,
            "e4m3",
            bv2,
            None,
            "e4m3",
            term2,
        )
        av3 = gl.amd.cdna4.async_copy.load_shared_relaxed(a3, layout=dot_a)
        bv3 = gl.amd.cdna4.async_copy.load_shared_relaxed(b3, layout=dot_b)
        term3 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
        term3 = gl.amd.cdna4.mfma_scaled(
            av3,
            None,
            "e4m3",
            bv3,
            None,
            "e4m3",
            term3,
        )

        next_a += block_k
        next_b += block_k
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b0,
            next_b,
            b_offsets,
            cache_modifier=".ca",
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a0,
            next_a,
            a_offsets,
            cache_modifier=".ca",
        )
        next_a += block_k
        next_b += block_k
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b1,
            next_b,
            b_offsets,
            cache_modifier=".ca",
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a1,
            next_a,
            a_offsets,
            cache_modifier=".ca",
        )
        next_a += block_k
        next_b += block_k
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b2,
            next_b,
            b_offsets,
            cache_modifier=".ca",
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a2,
            next_a,
            a_offsets,
            cache_modifier=".ca",
        )
        next_a += block_k
        next_b += block_k
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b3,
            next_b,
            b_offsets,
            cache_modifier=".ca",
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a3,
            next_a,
            a_offsets,
            cache_modifier=".ca",
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc += term0 * s0[:, None] * q0
        acc = _carry_accumulator(acc)
        acc += term1 * s1[:, None] * q1
        acc = _carry_accumulator(acc)
        acc += term2 * s2[:, None] * q2
        acc = _carry_accumulator(acc)
        acc += term3 * s3[:, None] * q3
        acc = _carry_accumulator(acc)

        scale_offset = 4 * turn + 4
        s0 = gl.load(xs_ptr + scale_m * 16 + scale_offset)
        q0 = gl.load(ws_ptr + scale_group + scale_offset)
        s1 = gl.load(xs_ptr + scale_m * 16 + scale_offset + 1)
        q1 = gl.load(ws_ptr + scale_group + scale_offset + 1)
        s2 = gl.load(xs_ptr + scale_m * 16 + scale_offset + 2)
        q2 = gl.load(ws_ptr + scale_group + scale_offset + 2)
        s3 = gl.load(xs_ptr + scale_m * 16 + scale_offset + 3)
        q3 = gl.load(ws_ptr + scale_group + scale_offset + 3)
        gl.amd.cdna4.async_copy.wait_group(0)

    av0 = gl.amd.cdna4.async_copy.load_shared_relaxed(a0, layout=dot_a)
    bv0 = gl.amd.cdna4.async_copy.load_shared_relaxed(b0, layout=dot_b)
    term0 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
    term0 = gl.amd.cdna4.mfma_scaled(
        av0,
        None,
        "e4m3",
        bv0,
        None,
        "e4m3",
        term0,
    )
    av1 = gl.amd.cdna4.async_copy.load_shared_relaxed(a1, layout=dot_a)
    bv1 = gl.amd.cdna4.async_copy.load_shared_relaxed(b1, layout=dot_b)
    term1 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
    term1 = gl.amd.cdna4.mfma_scaled(
        av1,
        None,
        "e4m3",
        bv1,
        None,
        "e4m3",
        term1,
    )
    av2 = gl.amd.cdna4.async_copy.load_shared_relaxed(a2, layout=dot_a)
    bv2 = gl.amd.cdna4.async_copy.load_shared_relaxed(b2, layout=dot_b)
    term2 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
    term2 = gl.amd.cdna4.mfma_scaled(
        av2,
        None,
        "e4m3",
        bv2,
        None,
        "e4m3",
        term2,
    )
    av3 = gl.amd.cdna4.async_copy.load_shared_relaxed(a3, layout=dot_a)
    bv3 = gl.amd.cdna4.async_copy.load_shared_relaxed(b3, layout=dot_b)
    term3 = gl.zeros((block_m, block_n), gl.float32, layout=mfma_layout)
    term3 = gl.amd.cdna4.mfma_scaled(
        av3,
        None,
        "e4m3",
        bv3,
        None,
        "e4m3",
        term3,
    )

    acc += term0 * s0[:, None] * q0
    acc = _carry_accumulator(acc)
    acc += term1 * s1[:, None] * q1
    acc = _carry_accumulator(acc)
    acc += term2 * s2[:, None] * q2
    acc = _carry_accumulator(acc)
    acc += term3 * s3[:, None] * q3

    out_m = gl.arange(0, block_m, layout=gl.SliceLayout(1, mfma_layout))
    out_n = pid_n * block_n + gl.arange(
        0,
        block_n,
        layout=gl.SliceLayout(0, mfma_layout),
    )
    gl.amd.cdna4.buffer_store(
        acc.to(out_ptr.type.element_ty),
        out_ptr,
        out_m[:, None] * 7168 + out_n[None, :],
    )


def block_scaled_mm_n7168_k2048_m32(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Run the registered `(32, 7168, 2048, 2048)` implementation."""
    assert output.shape == (32, 7168)
    assert output.dtype == torch.bfloat16
    assert output.device == a.device and output.is_contiguous()
    _block_scaled_mm_n7168_k2048_m32[(224,)](
        a,
        b,
        a_scale,
        b_scale,
        output,
        num_warps=4,
        num_stages=2,
        waves_per_eu=1,
    )
    return output


__all__ = ["block_scaled_mm_n7168_k2048_m32"]
