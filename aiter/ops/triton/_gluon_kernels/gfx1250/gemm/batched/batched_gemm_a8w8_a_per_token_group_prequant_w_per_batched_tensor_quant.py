# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon (gfx1250) batched a8w8 GEMM with in-kernel per-token-group activation
quantization and a single per-batched-tensor weight scale.

    Y[b] = (Xq[b] * a_scale) @ WQ[b]^T * w_scale

A (X)  : (B, M, K) bf16/fp16, quantized to fp8 in-kernel, per token group.
B (WQ) : (B, N, K) fp8, pre-quantized. NOTE: unlike the triton kernel this one
         reads WQ as (N, K) tiles directly -- the caller must NOT transpose it.
C (Y)  : (B, M, N), or (M, B, N) when transpose_bm=True (expressed via strides).
"""

import math

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_MAX_PAD_INTERVAL_DWORDS = 256

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "QGROUP",
    "NUM_BUFFERS",
    "num_warps",
    "waves_per_eu",
    "QUANT_IN_DOT",
    "HAS_BIAS",
]

_batched_gemm_a8w8_ptg_repr = make_kernel_repr(
    "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_gfx1250_kernel",
    _GLUON_REPR_KEYS,
)


def _pad_interval(extent, elem_bits):
    return min(extent, _MAX_PAD_INTERVAL_DWORDS * 32 // elem_bits)


def create_wmma_layouts(num_warps, BLOCK_M, BLOCK_N, instr_k=128):
    """WMMA layout whose warps are laid out along N first, then M.

    The stock aiter helper always puts one warp bit on N and the rest on M,
    which replicates work when BLOCK_M is tiny (the MLA decode case).
    """
    m_tiles = max(BLOCK_M // 16, 1)
    n_tiles = max(BLOCK_N // 16, 1)
    bases = []
    mi = 1
    ni = 1
    for _ in range(int(math.log2(num_warps))):
        if ni < n_tiles:
            bases.append((0, ni))
            ni *= 2
        elif mi < m_tiles:
            bases.append((mi, 0))
            mi *= 2
        else:  # more warps than tiles -> replicate along N
            bases.append((0, ni))
            ni *= 2
    warp_bases = tuple(bases)
    wmma_layout = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, instr_k]
    )
    operand_a = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)
    return wmma_layout, operand_a, operand_b


def create_blocked_a(BLOCK_M, QGROUP, num_warps):
    """Blocked layout for the bf16 activation tile.

    K (the quantization-group axis) is kept inside a single warp so the
    per-token-group amax reduction is a short intra-wave shuffle chain.
    Only used when QUANT_IN_DOT is off.
    """
    per_thread_k = 8 if QGROUP >= 8 * 16 else max(1, QGROUP // 16)
    threads_k = min(32, max(1, QGROUP // per_thread_k))
    threads_m = 32 // threads_k
    return gl.BlockedLayout(
        size_per_thread=[1, per_thread_k],
        threads_per_warp=[threads_m, threads_k],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )


def create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, a_bits=16, b_bits=8):
    """Padded shared-memory layouts for the A (bf16) and B (fp8) staging tiles."""
    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BLOCK_K, a_bits), 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BLOCK_K, b_bits), 16]], [BLOCK_N, BLOCK_K], [1, 0]
    )
    return shared_a, shared_b


@gluon.jit
def _quant_wmma(
    sa_tile,
    sb_tile,
    acc,
    zeros,
    DTYPE_MAX: gl.constexpr,
    FP8_TY: gl.constexpr,
    BLOCKED_A: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    QUANT_IN_DOT: gl.constexpr,
):
    # QUANT_IN_DOT: read the bf16 tile straight into the fp8 dot-operand layout.
    # The element->(lane, register) map of DotOperandLayout does not depend on
    # the element width, so quantizing in place produces the fp8 operand with
    # zero layout conversion (saves an LDS round trip per A tile). The amax
    # reduction along K then happens inside the dot layout, where each lane
    # already owns a contiguous run of K.
    if QUANT_IN_DOT:
        a_bf = sa_tile.load(layout=OPA)
    else:
        a_bf = sa_tile.load(layout=BLOCKED_A)
    a_f = a_bf.to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(a_f), axis=1), 1e-10)
    a_scale = amax * (1.0 / DTYPE_MAX)
    a_q = gl.clamp(a_f * (1.0 / a_scale)[:, None], -DTYPE_MAX, DTYPE_MAX).to(FP8_TY)
    if QUANT_IN_DOT:
        a_dot = a_q
    else:
        a_dot = gl.convert_layout(a_q, OPA)
    b_dot = sb_tile.permute((1, 0)).load(layout=OPB)
    res = gl.amd.gfx1250.wmma(a_dot, b_dot, zeros)
    s = gl.convert_layout(a_scale, gl.SliceLayout(1, WMMA_LAYOUT))
    return acc + res * s[:, None]


@gluon.jit(repr=_batched_gemm_a8w8_ptg_repr)
def _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_gluon_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    w_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bn,
    stride_bk,
    stride_cb,
    stride_cm,
    stride_cn,
    stride_biasb,
    HAS_BIAS: gl.constexpr,
    DTYPE_MAX: gl.constexpr,
    FP8_TY: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    QGROUP: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    K_GROUPS: gl.constexpr,
    BLOCKED_A: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPA: gl.constexpr,
    OPB: gl.constexpr,
    QUANT_IN_DOT: gl.constexpr,
    num_warps: gl.constexpr,
    waves_per_eu: gl.constexpr,
):
    batch_id = gl.program_id(axis=0).to(gl.int64)
    pid = gl.program_id(axis=1)

    num_pid_n = gl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + batch_id * stride_ab,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + batch_id * stride_bb,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_B,
    )

    sa = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], layout=SHARED_A
    )
    sb = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], layout=SHARED_B
    )

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    zeros = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    nl = 0
    nc = 0

    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [off_m, nl * BLOCK_K], sa.index(nl % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [off_n, nl * BLOCK_K], sb.index(nl % NUM_BUFFERS)
        )
        nl += 1

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [off_m, nl * BLOCK_K], sa.index(nl % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [off_n, nl * BLOCK_K], sb.index(nl % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        nl += 1

        buf = nc % NUM_BUFFERS
        for g in gl.static_range(K_GROUPS):
            acc = _quant_wmma(
                sa.index(buf).slice(g * QGROUP, QGROUP, dim=1),
                sb.index(buf).slice(g * QGROUP, QGROUP, dim=1),
                acc,
                zeros,
                DTYPE_MAX,
                FP8_TY,
                BLOCKED_A,
                OPA,
                OPB,
                WMMA_LAYOUT,
                QUANT_IN_DOT,
            )
        nc += 1

    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        buf = nc % NUM_BUFFERS
        for g in gl.static_range(K_GROUPS):
            acc = _quant_wmma(
                sa.index(buf).slice(g * QGROUP, QGROUP, dim=1),
                sb.index(buf).slice(g * QGROUP, QGROUP, dim=1),
                acc,
                zeros,
                DTYPE_MAX,
                FP8_TY,
                BLOCKED_A,
                OPA,
                OPB,
                WMMA_LAYOUT,
                QUANT_IN_DOT,
            )
        nc += 1

    w_scale = gl.load(w_scale_ptr)
    acc = acc * w_scale

    offs_cm = off_m + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = off_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))

    if HAS_BIAS:
        bias_vals = gl.load(
            bias_ptr + batch_id * stride_biasb + offs_cn,
            mask=offs_cn < N,
            other=0.0,
        )
        acc = acc + bias_vals[None, :]

    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(
        acc.to(c_ptr.type.element_ty),
        c_ptr + batch_id * stride_cb,
        offs_c,
        mask=mask_c,
    )
