# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config

import triton

_batched_gemm_a8wfp4_repr = make_kernel_repr(
    "_batched_gemm_a8wfp4_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "A_PACK",
        "A_FORMAT",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
    ],
)

_batched_gemm_a8wfp4_reduce_repr = make_kernel_repr(
    "_batched_gemm_a8wfp4_reduce_kernel",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "ACTUAL_KSPLIT",
        "MAX_KSPLIT",
    ],
)


@triton.heuristics(
    {
        # K is in true (unpacked) element units here.
        "EVEN_K": lambda args: (args["K"] % args["BLOCK_SIZE_K"] == 0)
        and (args["SPLITK_BLOCK_SIZE"] % args["BLOCK_SIZE_K"] == 0)
        and (args["K"] % args["SPLITK_BLOCK_SIZE"] == 0),
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_batched_gemm_a8wfp4_repr)
def _batched_gemm_a8wfp4_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scales_ptr,
    b_scales_ptr,
    M,
    N,
    K,
    stride_in_ab,
    stride_in_am,
    stride_in_ak,
    stride_in_bb,
    stride_in_bk,
    stride_in_bn,
    stride_in_cb,
    stride_in_ck,
    stride_in_cm,
    stride_in_cn,
    stride_in_asb,
    stride_in_asm,
    stride_in_ask,
    stride_in_bsb,
    stride_in_bsn,
    stride_in_bsk,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    A_PACK: tl.constexpr,
    A_FORMAT: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """Strided-batched matmul C[b] = A[b] x B[b].

    A is MXFP8 (e4m3, A_PACK=1) or MXFP4 (e2m1, A_PACK=2); B is MXFP4
    (e2m1). A_scales and B_scales are per-1x32 e8m0 microscales folded into a
    scaled MFMA. K is in true (unpacked) element units; A packs A_PACK elems
    per byte, B packs 2 elems per byte.
    """
    tl.assume(stride_in_ab > 0)
    tl.assume(stride_in_am > 0)
    tl.assume(stride_in_ak > 0)
    tl.assume(stride_in_bb > 0)
    tl.assume(stride_in_bk > 0)
    tl.assume(stride_in_bn > 0)
    tl.assume(stride_in_cb > 0)
    tl.assume(stride_in_cm > 0)
    tl.assume(stride_in_cn > 0)
    tl.assume(stride_in_asb > 0)
    tl.assume(stride_in_asm > 0)
    tl.assume(stride_in_ask > 0)
    tl.assume(stride_in_bsb > 0)
    tl.assume(stride_in_bsk > 0)
    tl.assume(stride_in_bsn > 0)

    pid_batch = tl.program_id(axis=0)
    pid_unified = tl.program_id(axis=1)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Cast batch id and batch-dim strides to int64 to avoid int32 overflow.
    stride_ab = tl.cast(stride_in_ab, tl.int64)
    stride_am = tl.cast(stride_in_am, tl.int64)
    stride_ak = tl.cast(stride_in_ak, tl.int64)
    stride_bb = tl.cast(stride_in_bb, tl.int64)
    stride_bk = tl.cast(stride_in_bk, tl.int64)
    stride_bn = tl.cast(stride_in_bn, tl.int64)
    stride_cb = tl.cast(stride_in_cb, tl.int64)
    stride_ck = tl.cast(stride_in_ck, tl.int64)
    stride_cm = tl.cast(stride_in_cm, tl.int64)
    stride_cn = tl.cast(stride_in_cn, tl.int64)
    stride_asb = tl.cast(stride_in_asb, tl.int64)
    stride_asm = tl.cast(stride_in_asm, tl.int64)
    stride_ask = tl.cast(stride_in_ask, tl.int64)
    stride_bsb = tl.cast(stride_in_bsb, tl.int64)
    stride_bsk = tl.cast(stride_in_bsk, tl.int64)
    stride_bsn = tl.cast(stride_in_bsn, tl.int64)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_batch >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    SCALE_GROUP_SIZE: tl.constexpr = 32

    if (pid_k * SPLITK_BLOCK_SIZE) < K:
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

        # Byte columns per K block: A packs A_PACK elems/byte, B packs 2.
        offs_ka = tl.arange(0, BLOCK_SIZE_K // A_PACK)
        offs_ka_split = pid_k * (SPLITK_BLOCK_SIZE // A_PACK) + offs_ka
        offs_kb = tl.arange(0, BLOCK_SIZE_K // 2)
        offs_kb_split = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_kb
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            pid_batch * stride_ab
            + offs_am[:, None] * stride_am
            + offs_ka_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            pid_batch * stride_bb
            + offs_kb_split[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

        offs_ks = (pid_k * (SPLITK_BLOCK_SIZE // SCALE_GROUP_SIZE)) + tl.arange(
            0, BLOCK_SIZE_K // SCALE_GROUP_SIZE
        )
        a_scale_ptrs = (
            a_scales_ptr
            + pid_batch * stride_asb
            + offs_am[:, None] * stride_asm
            + offs_ks[None, :] * stride_ask
        )
        # B scales are N x K even though B operand is K x N.
        b_scale_ptrs = (
            b_scales_ptr
            + pid_batch * stride_bsb
            + offs_bn[:, None] * stride_bsn
            + offs_ks[None, :] * stride_bsk
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            a_scales = tl.load(a_scale_ptrs)
            b_scales = tl.load(b_scale_ptrs)
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs,
                    mask=offs_ka[None, :] < (K - k * BLOCK_SIZE_K) // A_PACK,
                    other=0,
                )
                b = tl.load(
                    b_ptrs,
                    mask=offs_kb[:, None] < (K - k * BLOCK_SIZE_K) // 2,
                    other=0,
                )

            accumulator = tl.dot_scaled(
                a, a_scales, A_FORMAT, b, b_scales, "e2m1", acc=accumulator
            )

            a_ptrs += (BLOCK_SIZE_K // A_PACK) * stride_ak
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            a_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_ask
            b_scale_ptrs += (BLOCK_SIZE_K // SCALE_GROUP_SIZE) * stride_bsk

        c = accumulator.to(c_ptr.type.element_ty)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + pid_batch * stride_cb
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(repr=_batched_gemm_a8wfp4_reduce_repr)
def _batched_gemm_a8wfp4_reduce_kernel(
    c_in_ptr,
    c_out_ptr,
    M,
    N,
    stride_c_in_b,
    stride_c_in_k,
    stride_c_in_m,
    stride_c_in_n,
    stride_c_out_b,
    stride_c_out_m,
    stride_c_out_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACTUAL_KSPLIT: tl.constexpr,
    MAX_KSPLIT: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, MAX_KSPLIT)
    c_in_ptrs = (
        c_in_ptr
        + pid_batch * stride_c_in_b
        + (offs_k[:, None, None] * stride_c_in_k)
        + (offs_m[None, :, None] * stride_c_in_m)
        + (offs_n[None, None, :] * stride_c_in_n)
    )

    if ACTUAL_KSPLIT == MAX_KSPLIT:
        c = tl.load(c_in_ptrs)
    else:
        c = tl.load(c_in_ptrs, mask=offs_k[:, None, None] < ACTUAL_KSPLIT)
    c = tl.sum(c, axis=0)

    c = c.to(c_out_ptr.type.element_ty)

    c_out_ptrs = (
        c_out_ptr
        + pid_batch * stride_c_out_b
        + (offs_m[:, None] * stride_c_out_m)
        + (offs_n[None, :] * stride_c_out_n)
    )

    tl.store(c_out_ptrs, c)


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # Heuristics to make EVEN_K == True as much as possible. K is in true
    # unpacked element units.
    NUM_KSPLIT_STEP = 2
    BLOCK_SIZE_K_STEP = 2
    SPLITK_BLOCK_SIZE = (
        triton.cdiv(triton.cdiv(K, NUM_KSPLIT), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % SPLITK_BLOCK_SIZE == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % BLOCK_SIZE_K == 0
        ):
            break
        elif K % SPLITK_BLOCK_SIZE != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        elif K % BLOCK_SIZE_K != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        else:
            break

        SPLITK_BLOCK_SIZE = (
            triton.cdiv(triton.cdiv(K, NUM_KSPLIT), BLOCK_SIZE_K) * BLOCK_SIZE_K
        )

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


def _get_config(
    M: int,
    N: int,
    K: int,
):
    # Config files use K=2*K in their naming (FP4 weights are packed 2/byte),
    # so the file's K corresponds to 2*K unpacked elements.
    config, is_tunned = get_gemm_config(
        "BATCHED_GEMM-A8WFP4",
        M,
        N,
        2 * K,
        bounds=(4, 8, 16, 32, 64, 128, 256, 320, 512, 1024, 2048, 4096, 8192),
    )

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )
        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT
    else:
        config["SPLITK_BLOCK_SIZE"] = K

    if config["BLOCK_SIZE_K"] >= K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(K)
        config["SPLITK_BLOCK_SIZE"] = K

    return config, is_tunned
