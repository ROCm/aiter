# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import argparse
import itertools
import os
from unittest.mock import patch

import torch

import aiter
from aiter import dtypes
from aiter.benchmark_data_init import add_data_init_args, fill, make_generator
from aiter.benchmark_reporting import print_json_table
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    run_perftest,
)

try:
    from aiter.ops.mhc import mhc_fused_post_pre_large_m
except ImportError:
    mhc_fused_post_pre_large_m = None

# gfx950 large-M path (mhc_fused_post_pre_large_m) applies when M > 1024.
LARGE_M_MIN = 1025

try:
    from aiter.ops.triton.fusions.mhc import mhc_post_pre as triton_mhc_post_pre

    _HAS_TRITON_MHC_POST_PRE = True
except ImportError:
    triton_mhc_post_pre = None
    _HAS_TRITON_MHC_POST_PRE = False


# Triton ``mhc_post_pre`` is only validated for M <= 4096 (hangs on larger M).
TRITON_MHC_POST_PRE_MAX_M = 4096

torch.set_default_device("cuda")
# torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)


# copy from tilelang/examples/deepseek_mhc/example_mhc_pre.py
def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor; TileLang version of mhc_pre_gemm_sqrsum doesn't support this

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    import math

    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size: int,
        rms_eps: float,
        hc_pre_eps: float,
        hc_sinkhorn_eps: float,
        hc_post_mult_value: float,
        sinkhorn_repeat: int,
        n_splits: int = 16,
        hc_mult: int = 4,
    ):
        """Deeply fused kernels, everything other than gemm & sqrsum in mHC pre block."""
        num_tokens = T.dynamic("num_tokens")
        hc_mult3 = hc_mult * (2 + hc_mult)
        hidden_block = math.gcd(512, hidden_size)

        gemm_out_mul: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]
        gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
        hc_scale: T.Tensor[[3], T.float32]
        hc_base: T.Tensor[[hc_mult3], T.float32]
        residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
        # outputs
        post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
        comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
        layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

        with T.Kernel(num_tokens, threads=96) as i:
            ##################################################################
            # _pre_norm_fn_fwd_norm
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                ##################################################################
                # _pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = (
                        T.sigmoid(
                            mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]
                        )
                        * hc_post_mult_value
                    )
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = (
                        mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                        + hc_base[j * hc_mult + k + hc_mult * 2]
                    )

                ##################################################################
                # _sinkhorn_fwd
                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * hc_scale[0] + hc_base[j],
                        )
                        + hc_pre_eps
                    )
                ###################################################################
                # _pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]

                    T.copy(ol, layer_input[i, i0_h * hidden_block])

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hc_mult3: int,
        hc_hidden_size: int,
        token_block: int = 32,
        hidden_block: int = 256,
    ) -> tilelang.JITKernel:
        """Not highly optimized TileLang implementation of fused gemm and sqrsum in mHC pre block."""
        assert hc_mult3 <= 32  # should be 24 usually
        num_tokens = T.dynamic("num_tokens")
        assert hc_hidden_size % hidden_block == 0

        x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
        fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
        out: T.Tensor((num_tokens, hc_mult3), T.float32)
        sqrsum: T.Tensor((num_tokens), T.float32)

        with T.Kernel(T.ceildiv(num_tokens, token_block), threads=256) as px:
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout(
                    {x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)}
                )

                T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
                T.copy(fn[0, pz * hidden_block], fn_smem)

                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem_16, x_frag_16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_frag_16, x_frag)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += (
                            x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]
                        )

                # should be TF32 gemm
                T.gemm(
                    x_frag,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    wg_wait=0,
                    clear_accum=False,
                )
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                sqrsum[px * token_block + i] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                if j < hc_mult3:
                    out[px * token_block + i, j] = out_frag[i, j]

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    assert (
        n_splits == 1
    ), "The simple TileLang version gemm_sqrsum doesn't support split-k"
    mhc_pre_gemm_sqrsum_tilelang(
        residual_flat.view(num_tokens, hc_mult * hidden_size),
        fn_flat,
        gemm_out_mul.squeeze(0),
        gemm_out_sqrsum.squeeze(0),
        hc_mult3,
        hc_mult * hidden_size,
        hidden_block=128,
    )

    mhc_pre_big_fuse_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
    )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


# copy from tilelang/examples/deepseek_mhc/example_mhc_pre.py
def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    test_hc_head: bool = False,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    out = residual_flat @ fn.T
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

    if not test_hc_head:
        hc_scale = torch.cat(
            [
                hc_scale[0].expand(hc_mult),
                hc_scale[1].expand(hc_mult),
                hc_scale[2].expand(hc_mult * hc_mult),
            ],
        )
        mixes = mixes * hc_scale + hc_base

        pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
        post_mix = (
            mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
        ).unsqueeze(-1)
        res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

        def sinkhorn_normalize_ref(
            x: torch.Tensor, repeat: int, eps: float
        ) -> torch.Tensor:
            x = x.softmax(-1) + eps
            x = x / (x.sum(-2, keepdim=True) + eps)
            for _ in range(repeat - 1):
                x = x / (x.sum(-1, keepdim=True) + eps)
                x = x / (x.sum(-2, keepdim=True) + eps)
            return x

        res_mix = sinkhorn_normalize_ref(
            res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
        )
    else:
        hc_scale = hc_scale[0].expand(hc_mult)
        mixes = mixes * hc_scale + hc_base
        pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
        post_mix = None
        res_mix = None

    layer_input = (residual * pre_mix).sum(-2)

    if norm_weight is not None:
        x = layer_input
        rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = (x.float() * rms * norm_weight.float()).bfloat16()

    return post_mix, res_mix, layer_input.bfloat16()


def mhc_pre_norm_split_hip(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    post_mix, res_mix, layer_input = aiter.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )
    out = torch.empty_like(layer_input)
    aiter.rmsnorm(out, layer_input, norm_weight, norm_eps)
    return post_mix, res_mix, out


@benchmark()
def test_mhc_pre(
    m,
    hidden_size,
    hc_mult,
    test_hc_head=False,
    fuse_rmsnorm=False,
    w_preshuffle_bf16=False,
    dtype=dtypes.bf16,
    data_init="norm",
    seed=0,
):
    if fuse_rmsnorm and test_hc_head:
        raise ValueError("fuse_rmsnorm and hc_head are mutually exclusive")

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2 if not test_hc_head else hc_mult
    hc_hidden_size = hc_mult * hidden_size
    gen = make_generator(seed)
    residual = fill((m, hc_mult, hidden_size), data_init, gen, dtype=dtype)
    fn = fill((hc_mult3, hc_hidden_size), data_init, gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), data_init, gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), data_init, gen, dtype=dtypes.fp32) * 0.1
    norm_weight = None
    if fuse_rmsnorm:
        norm_weight = fill((hidden_size,), data_init, gen, dtype=dtype)
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 1.0,
        "sinkhorn_repeat": 20 if not test_hc_head else 0,
    }
    if fuse_rmsnorm:
        extra_args["norm_eps"] = 1e-6

    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_ref(
        residual,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
        test_hc_head=test_hc_head,
        norm_weight=norm_weight,
    )

    hip_nofuse_us = None
    if fuse_rmsnorm:
        _, hip_nofuse_us = run_perftest(
            mhc_pre_norm_split_hip,
            residual,
            fn,
            hc_scale,
            hc_base,
            norm_weight=norm_weight,
            **extra_args,
        )

    hip_kwargs = {**extra_args}
    if fuse_rmsnorm:
        hip_kwargs["norm_weight"] = norm_weight
    (post_mix_hip, comb_mix_hip, layer_input_hip), hip_us = run_perftest(
        aiter.mhc_pre,
        residual,
        fn,
        hc_scale,
        hc_base,
        **hip_kwargs,
    )
    if not test_hc_head:
        checkAllclose(post_mix_ref, post_mix_hip, msg="post_mix")
        checkAllclose(comb_mix_ref, comb_mix_hip, msg="comb_mix")
    hip_err = checkAllclose(layer_input_ref, layer_input_hip, msg="layer_input")
    ret = {"fuse_rmsnorm": fuse_rmsnorm, "test_hc_head": test_hc_head}
    ret["hip_err"] = hip_err
    ret["hip_us"] = hip_us

    # bf16 GEMM path: pre-pack fn (fp32) -> int32 BF16 hi/lo ONCE via mhc_shuffle_fn
    # (fn are constant weights), then run mhc_pre with w_preshuffle_bf16=1 so the gemm
    # bit-extracts hi/lo instead of recomputing the fp32->bf16 split per (m_block, k).
    # On gfx950 this uses the native bf16 MFMA; gfx1250 the wave32 bf16 WMMA (UNVERIFIED).
    if w_preshuffle_bf16:
        from aiter.ops.mhc import mhc_shuffle_fn

        fn_packed = mhc_shuffle_fn(fn)
        (
            _post_mix_bf16,
            _comb_mix_bf16,
            layer_input_bf16,
        ), hip_bf16_us = run_perftest(
            aiter.mhc_pre,
            residual,
            fn_packed,
            hc_scale,
            hc_base,
            w_preshuffle_bf16=1,
            **hip_kwargs,
        )
        ret["hip_bf16_err"] = checkAllclose(
            layer_input_ref, layer_input_bf16, msg="layer_input(bf16)"
        )
        ret["hip_bf16_us"] = hip_bf16_us
    if fuse_rmsnorm:
        ret["hip_nofuse_us"] = hip_nofuse_us
        ret["TB/s"] = (
            (
                layer_input_ref.numel() * layer_input_ref.dtype.itemsize
                + residual.numel() * residual.dtype.itemsize
                + norm_weight.numel() * norm_weight.dtype.itemsize
            )
            / 1e6
            / hip_us
        )
    # ret["TFLOPS * us"] = 2.0 * m * hidden_size * hc_mult * hc_mult3 / 1e6
    # ret["GB"] = (m * hc_mult3 * dtypes.fp32.itemsize + (m * hc_mult + m) * hidden_size * dtypes.bf16.itemsize) / 1e6
    # try:
    #     (post_mix_tilelang, comb_mix_tilelang, layer_input_tilelang), tilelang_us = run_perftest(mhc_pre_tilelang, residual, fn, hc_scale, hc_base, **extra_args)
    #     checkAllclose(post_mix_ref, post_mix_tilelang, msg="post_mix")
    #     tilelang_err = checkAllclose(comb_mix_ref, comb_mix_tilelang, msg="comb_mix")
    #     checkAllclose(layer_input_ref, layer_input_tilelang, msg="layer_input")
    #     ret["tilelang_err"] = tilelang_err
    #     ret["tilelang_us"] = tilelang_us
    # except Exception as e:
    #     tilelang_err = str(e)
    #     print(f"tilelang error: {tilelang_err}")

    return ret


# copy from tilelang/examples/deepseek_mhc/example_mhc_post.py
def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    import math

    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        },
    )
    def mhc_post_tilelang(
        a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
    ) -> tilelang.JITKernel:
        # rename for shorter code
        n = T.dynamic("num_tokens")
        h = hidden

        h_blk = math.gcd(hidden, h_blk)
        a: T.Tensor((n, hc, hc), T.float32)
        b: T.Tensor((n, hc, h), T.bfloat16)
        c: T.Tensor((n, hc), T.float32)
        d: T.Tensor((n, h), T.bfloat16)
        x: T.Tensor((n, hc, h), T.bfloat16)
        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((hc, h_blk), T.float32)
            b_local = T.alloc_fragment((hc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((hc, hc), T.float32)
            c_local = T.alloc_fragment(hc, T.float32)
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[i_n, i0_h * h_blk], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += (
                            a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                        )
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def mhc_post_hip(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(residual)
    aiter.mhc_post(
        out,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    return out


# copy from tilelang/examples/deepseek_mhc/example_mhc_post.py
def mhc_post_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).to(x.dtype)


@benchmark()
def test_mhc_post(m, hidden_size, hc_mult, dtype=dtypes.bf16, data_init="norm", seed=0):
    gen = make_generator(seed)
    x = fill((m, hidden_size), data_init, gen, dtype=dtype)
    residual = fill((m, hc_mult, hidden_size), data_init, gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), data_init, gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), data_init, gen, dtype=dtypes.fp32)
    out_ref = mhc_post_ref(x, residual, post_layer_mix, comb_res_mix)
    out_hip, hip_us = run_perftest(
        mhc_post_hip,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    hip_err = checkAllclose(out_ref, out_hip, msg="out")
    # checkAllclose only reports; a wrong mhc_post (e.g. at a fallback tile:
    # hidden 512 runs 256-column tiles, 1024 runs 512) must fail the test.
    assert hip_err <= 0.05, (
        f"mhc_post m={m} hidden={hidden_size} {dtype}: {hip_err:.2%} of "
        "elements outside tolerance"
    )
    ret = {}
    ret["hip_err"] = hip_err
    ret["hip_us"] = hip_us
    ret["TB/s"] = (
        (
            out_ref.numel() * out_ref.dtype.itemsize
            + x.numel() * x.dtype.itemsize
            + residual.numel() * residual.dtype.itemsize
            + post_layer_mix.numel() * post_layer_mix.dtype.itemsize
            + comb_res_mix.numel() * comb_res_mix.dtype.itemsize
        )
        / 1e6
        / hip_us
    )
    # try:
    #     (out_tilelang), tilelang_us = run_perftest(mhc_post_tilelang, x, residual, post_layer_mix, comb_res_mix)
    #     tilelang_err = checkAllclose(out_ref, out_tilelang, msg="out")
    #     ret["tilelang_err"] = tilelang_err
    #     ret["tilelang_us"] = tilelang_us
    # except Exception as e:
    #     tilelang_err = str(e)
    #     print(f"tilelang error: {tilelang_err}")

    return ret


def mhc_post_reduce_ref(
    streams: torch.Tensor, pre_mix: torch.Tensor | None
) -> torch.Tensor:
    """The streams `mhc_post` wrote, reduced in FP32 with h ascending from 0.

    Eager torch runs every product and every sum as its own kernel, so none of
    them is contracted into an FMA -- the rounding `mhc_post_reduce` promises.
    """
    y = streams.float()
    acc = torch.zeros_like(y[:, 0])
    for h in range(y.shape[1]):
        acc = acc + (y[:, h] if pre_mix is None else pre_mix[:, h : h + 1] * y[:, h])
    if pre_mix is None:
        acc = acc * (1.0 / y.shape[1])
    return acc.to(streams.dtype)


@benchmark()
def test_mhc_post_reduce(
    m, hidden_size, hc_mult, reduce, dtype=dtypes.bf16, data_init="norm", seed=0
):
    """`mhc_post_reduce` against `mhc_post` + the FP32 reduction, to the bit.

    ``reduce`` is "pre_mix" (the pre-mix weighted sum) or "mean".
    """
    gen = make_generator(seed)
    x = fill((m, hidden_size), data_init, gen, dtype=dtype)
    residual = fill((m, hc_mult, hidden_size), data_init, gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), data_init, gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), data_init, gen, dtype=dtypes.fp32)
    pre_mix = (
        fill((m, hc_mult), data_init, gen, dtype=dtypes.fp32)
        if reduce == "pre_mix"
        else None
    )
    streams, post_us = run_perftest(
        mhc_post_hip, x, residual, post_layer_mix, comb_res_mix
    )
    out_ref = mhc_post_reduce_ref(streams, pre_mix)

    def run_reduce():
        out = torch.empty(m, hidden_size, dtype=dtype)
        aiter.mhc_post_reduce(out, x, residual, post_layer_mix, comb_res_mix, pre_mix)
        return out

    out, reduce_us = run_perftest(run_reduce)
    bit_exact = torch.equal(out.view(torch.int16), out_ref.view(torch.int16))
    mismatches = int((out.view(torch.int16) != out_ref.view(torch.int16)).sum())
    assert bit_exact, (
        f"mhc_post_reduce({reduce}) m={m} hidden={hidden_size}: "
        f"{mismatches} elements differ from mhc_post + the FP32 reduction"
    )
    return {
        "reduce": reduce,
        "bit_exact": bit_exact,
        "mhc_post_us": post_us,
        "reduce_us": reduce_us,
    }


def _launch_mhc_post(x, residual, post, comb):
    aiter.mhc_post(torch.empty_like(residual), x, residual, post, comb)


def _launch_mhc_post_reduce(x, residual, post, comb):
    out = torch.empty(x.shape, dtype=x.dtype)
    aiter.mhc_post_reduce(out, x, residual, post, comb)


def test_mhc_post_reduce_operand_guards():
    """`mhc_post_reduce` refuses an ``out`` that overlaps an input and operands
    off ``residual``'s GPU, and on a second GPU launches on that GPU's stream
    even while another device is current."""
    m, hidden_size, hc_mult = 2, 1024, 4
    x = torch.randn(m, hidden_size).bfloat16()
    residual = torch.randn(m, hc_mult, hidden_size).bfloat16()
    post = torch.rand(m, hc_mult, 1)
    comb = torch.rand(m, hc_mult, hc_mult)
    refusals = {
        "out overlaps residual": lambda: aiter.mhc_post_reduce(
            residual.view(-1)[: m * hidden_size].view(m, hidden_size),
            x,
            residual,
            post,
            comb,
        ),
        "out is x": lambda: aiter.mhc_post_reduce(x, x, residual, post, comb),
        "post_layer_mix on the CPU": lambda: aiter.mhc_post_reduce(
            torch.empty(m, hidden_size, dtype=x.dtype), x, residual, post.cpu(), comb
        ),
        "comb_res_mix on the CPU": lambda: aiter.mhc_post_reduce(
            torch.empty(m, hidden_size, dtype=x.dtype), x, residual, post, comb.cpu()
        ),
        "fp32 with no rows": lambda: aiter.mhc_post_reduce(
            torch.empty(0, hidden_size),
            torch.empty(0, hidden_size),
            torch.empty(0, hc_mult, hidden_size),
            torch.empty(0, hc_mult, 1),
            torch.empty(0, hc_mult, hc_mult),
        ),
    }
    rows = []
    for name, launch in refusals.items():
        try:
            launch()
            torch.cuda.synchronize()
            raised = False
        except RuntimeError:
            raised = True
        assert raised, f"mhc_post_reduce accepted: {name}"
        rows.append({"case": f"{name} raises", "ok": raised})
    if torch.cuda.device_count() > 1:
        expected = torch.empty(m, hidden_size, dtype=x.dtype)
        aiter.mhc_post_reduce(expected, x, residual, post, comb)
        dev = torch.device("cuda", 1)
        args = [t.to(dev) for t in (x, residual, post, comb)]
        out = torch.empty(m, hidden_size, dtype=x.dtype, device=dev)
        side = torch.cuda.Stream(device=0)
        with torch.cuda.device(0), torch.cuda.stream(side):
            aiter.mhc_post_reduce(out, *args)
        torch.cuda.synchronize(dev)
        same = torch.equal(out.cpu(), expected.cpu())
        assert same, "mhc_post_reduce on cuda:1 with cuda:0 current differs"
        rows.append(
            {"case": "cuda:1 inputs, cuda:0 current on a side stream", "ok": same}
        )
    return rows


def test_mhc_post_support_queries():
    """`mhc_post_supported` / `mhc_post_reduce_supported` against the launch:
    over a sweep of hidden sizes and dtypes, a query is False exactly where
    the op refuses the shape (raises), and True exactly where it runs."""
    from aiter.ops.mhc import mhc_post_reduce_supported, mhc_post_supported

    arch = get_gfx_runtime()
    m, hc_mult = 4, 4
    ops = (
        ("mhc_post", mhc_post_supported, _launch_mhc_post),
        ("mhc_post_reduce", mhc_post_reduce_supported, _launch_mhc_post_reduce),
    )
    hidden_sizes = sorted(set(range(256, 8193, 256)) | {320, 640, 2560})
    rows, mismatches = [], []
    for dtype in (dtypes.bf16, dtypes.fp16, dtypes.fp32):
        for hidden_size in hidden_sizes:
            x = torch.randn(m, hidden_size).to(dtype)
            residual = torch.randn(m, hc_mult, hidden_size).to(dtype)
            post = torch.rand(m, hc_mult, 1)
            comb = torch.rand(m, hc_mult, hc_mult)
            for name, query, launch in ops:
                supported = query(hidden_size, hc_mult, dtype, arch)
                try:
                    launch(x, residual, post, comb)
                    torch.cuda.synchronize()
                    ran = True
                except RuntimeError:
                    ran = False
                rows.append(
                    {
                        "op": name,
                        "dtype": str(dtype),
                        "hidden_size": hidden_size,
                        "supported": supported,
                        "ran": ran,
                    }
                )
                if supported != ran:
                    mismatches.append((name, str(dtype), hidden_size, supported, ran))
    assert not mismatches, f"query and launch disagree on {arch}: {mismatches}"
    return rows


def mhc_post_pre_ref(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused torch reference: mhc_post then mhc_pre."""
    next_residual = mhc_post_ref(layer_input, residual_in, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input_out = mhc_pre_ref(
        next_residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return post_mix, comb_mix, layer_input_out, next_residual


def mhc_post_pre_unfused_hip(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    **extra_args,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    next_residual = torch.empty_like(residual_in)
    aiter.mhc_post(
        next_residual,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
    )
    post_mix, comb_mix, layer_input_out = aiter.mhc_pre(
        next_residual,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
    )
    return post_mix, comb_mix, layer_input_out, next_residual


@benchmark()
def test_mhc_post_pre(
    m,
    hidden_size,
    hc_mult,
    fuse_rmsnorm=False,
    large_m=False,
    w_preshuffle_bf16=False,
    res_preshuffle=False,
    dtype=dtypes.bf16,
    data_init="norm",
    seed=0,
):
    """Fused mhc_post + mhc_pre: HIP ``mhc_fused_post_pre`` vs ref / unfused HIP / Triton.

    --w_preshuffle_bf16 selects packed BF16 hi/lo compute for all HIP paths.
    --res_shuffle requests the runtime fuse-aligned residual layout on gfx1250.
    """
    if hidden_size < 512:
        aiter.logger.info(
            "skip mhc_post_pre: hidden_size=%s < 512 (big_fuse dispatch)", hidden_size
        )
        return {"skipped": True}
    if not hasattr(aiter, "mhc_fused_post_pre"):
        aiter.logger.info("skip mhc_post_pre: aiter.mhc_fused_post_pre not available")
        return {"skipped": True}

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size

    gen = make_generator(seed)
    layer_input = fill((m, hidden_size), data_init, gen, dtype=dtype)
    residual_in = fill((m, hc_mult, hidden_size), data_init, gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), data_init, gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), data_init, gen, dtype=dtypes.fp32)
    fn = fill((hc_mult3, hc_hidden_size), data_init, gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), data_init, gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), data_init, gen, dtype=dtypes.fp32) * 0.1
    norm_weight = None
    if fuse_rmsnorm:
        norm_weight = fill((hidden_size,), data_init, gen, dtype=dtype)

    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
    }
    if fuse_rmsnorm:
        extra_args["norm_eps"] = 1e-6

    post_mix_ref, comb_mix_ref, layer_input_ref, next_residual_ref = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=extra_args["rms_eps"],
        hc_pre_eps=extra_args["hc_pre_eps"],
        hc_sinkhorn_eps=extra_args["hc_sinkhorn_eps"],
        hc_post_mult_value=extra_args["hc_post_mult_value"],
        sinkhorn_repeat=extra_args["sinkhorn_repeat"],
        norm_weight=norm_weight,
        norm_eps=extra_args.get("norm_eps", 1e-6),
    )

    hip_kwargs = {**extra_args}
    if fuse_rmsnorm:
        hip_kwargs["norm_weight"] = norm_weight

    from aiter.ops.mhc import (
        mhc_res_ks,
        mhc_res_shuffle,
        mhc_res_shuffle_enabled,
        mhc_res_unshuffle,
        mhc_shuffle_fn,
    )

    packed = w_preshuffle_bf16
    shuffled = res_preshuffle and mhc_res_shuffle_enabled(m)
    ret = {
        "fuse_rmsnorm": fuse_rmsnorm,
        "w_preshuffle_bf16": bool(packed),
        "res_preshuffle": bool(shuffled),
    }
    pack_flag = int(packed)
    # Pack weights and change residual layout independently, outside the timed call.
    if packed:
        fn_gemm = mhc_shuffle_fn(fn)
    else:
        fn_gemm = fn
    # The HIP kernels' shuffle block (the API picks Gluon for some shapes by default).
    hip_ks = mhc_res_ks(m, hidden_size, backend="hip")
    residual_in_fused = (
        mhc_res_shuffle(residual_in, hip_ks) if shuffled else residual_in
    )

    (
        post_mix_unfused,
        comb_mix_unfused,
        layer_input_unfused,
        next_residual_unfused,
    ), unfused_us = run_perftest(
        mhc_post_pre_unfused_hip,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn_gemm,
        hc_scale,
        hc_base,
        w_preshuffle_bf16=pack_flag,
        **hip_kwargs,
    )

    (
        post_mix_fused,
        comb_mix_fused,
        layer_input_fused,
        next_residual_fused,
    ), fused_us = run_perftest(
        aiter.mhc_fused_post_pre,
        layer_input,
        residual_in_fused,
        post_layer_mix,
        comb_res_mix,
        fn_gemm,
        hc_scale,
        hc_base,
        force_fused=True,
        w_preshuffle_bf16=w_preshuffle_bf16,
        res_preshuffle=res_preshuffle,
        backend="hip",
        **hip_kwargs,
    )

    # The public call without force_fused (HIP backend, like the other HIP candidates).
    (
        post_mix_default,
        comb_mix_default,
        layer_input_default,
        next_residual_default,
    ), default_us = run_perftest(
        aiter.mhc_fused_post_pre,
        layer_input,
        residual_in_fused,
        post_layer_mix,
        comb_res_mix,
        fn_gemm,
        hc_scale,
        hc_base,
        w_preshuffle_bf16=w_preshuffle_bf16,
        res_preshuffle=res_preshuffle,
        backend="hip",
        **hip_kwargs,
    )

    checkAllclose(post_mix_ref, post_mix_unfused, msg="unfused/post_mix")
    checkAllclose(comb_mix_ref, comb_mix_unfused, msg="unfused/comb_mix")
    hip_unfused_err = checkAllclose(
        layer_input_ref, layer_input_unfused, msg="unfused/layer_input"
    )
    checkAllclose(next_residual_ref, next_residual_unfused, msg="unfused/next_residual")
    if shuffled:
        next_residual_fused = mhc_res_unshuffle(next_residual_fused, hip_ks)
    checkAllclose(post_mix_ref, post_mix_fused, msg="fused/post_mix")
    checkAllclose(comb_mix_ref, comb_mix_fused, msg="fused/comb_mix")
    hip_fused_err = checkAllclose(
        layer_input_ref, layer_input_fused, msg="fused/layer_input"
    )
    checkAllclose(next_residual_ref, next_residual_fused, msg="fused/next_residual")
    if shuffled:
        next_residual_default = mhc_res_unshuffle(next_residual_default, hip_ks)
    checkAllclose(post_mix_ref, post_mix_default, msg="default/post_mix")
    checkAllclose(comb_mix_ref, comb_mix_default, msg="default/comb_mix")
    hip_default_err = checkAllclose(
        layer_input_ref, layer_input_default, msg="default/layer_input"
    )
    checkAllclose(next_residual_ref, next_residual_default, msg="default/next_residual")
    ret["unfused_us"] = unfused_us
    ret["hip_unfused_err"] = hip_unfused_err
    ret["fused_us"] = fused_us
    ret["hip_fused_err"] = hip_fused_err
    ret["default_us"] = default_us
    ret["hip_default_err"] = hip_default_err
    # print(f"next_residual_ref: {next_residual_ref}")
    # print(f"next_residual_fused: {next_residual_fused}")

    # gfx1250: Triton mhc_post_pre runs the Gluon kernels and takes the same
    # HIP-side inputs as the fused call above (packed fn, shuffled residual,
    # fused RMSNorm) -- the ATOM configuration. Other archs: plain fp32 inputs.
    triton_hip_layouts = get_gfx_runtime() == "gfx1250"
    run_triton = (
        _HAS_TRITON_MHC_POST_PRE
        and (triton_hip_layouts or not fuse_rmsnorm)
        and m <= TRITON_MHC_POST_PRE_MAX_M
    )
    if fuse_rmsnorm and not triton_hip_layouts:
        aiter.logger.info(
            "skip Triton mhc_post_pre: fuse_rmsnorm (gfx1250 Gluon path only)"
        )
    elif m > TRITON_MHC_POST_PRE_MAX_M:
        aiter.logger.info(
            "skip Triton mhc_post_pre: m=%s > %s",
            m,
            TRITON_MHC_POST_PRE_MAX_M,
        )

    if run_triton and triton_hip_layouts:
        # The Gluon kernels through aiter.mhc_fused_post_pre(backend="gluon"), with
        # the HIP-side inputs of the fused call above and the Gluon shuffle block.
        gluon_ks = mhc_res_ks(m, hidden_size, backend="gluon")
        gluon_residual_in = (
            mhc_res_shuffle(residual_in, gluon_ks) if shuffled else residual_in
        )
        (h_post_t, h_res_t, layer_input_t, residual_out_t), triton_us = run_perftest(
            aiter.mhc_fused_post_pre,
            layer_input,
            gluon_residual_in,
            post_layer_mix,
            comb_res_mix,
            fn_gemm,
            hc_scale,
            hc_base,
            force_fused=True,
            w_preshuffle_bf16=bool(packed),
            res_preshuffle=bool(res_preshuffle),
            backend="gluon",
            **hip_kwargs,
        )
        if shuffled:
            residual_out_t = mhc_res_unshuffle(residual_out_t, gluon_ks)
    elif run_triton:
        # phi layout (K, N) = (n*C, hc_mult3): the C-contiguous view of fn.
        post_mix_2d = post_layer_mix.squeeze(-1)
        (h_post_t, h_res_t, layer_input_t, residual_out_t), triton_us = run_perftest(
            triton_mhc_post_pre,
            layer_input,
            residual_in,
            post_mix_2d,
            comb_res_mix,
            fn.T,
            hc_scale,
            hc_base,
            hc_mult,
            eps=extra_args["rms_eps"],
            hc_pre_eps=extra_args["hc_pre_eps"],
            hc_post_mult_value=extra_args["hc_post_mult_value"],
            sinkhorn_iters=extra_args["sinkhorn_repeat"],
            asymmetric_exp_domain=True,
            hc_sinkhorn_eps=extra_args["hc_sinkhorn_eps"],
        )
    if run_triton:
        h_post_t = h_post_t.to(post_mix_ref.dtype)
        h_res_t = h_res_t.to(comb_mix_ref.dtype)
        layer_input_t = layer_input_t.to(layer_input_ref.dtype)
        residual_out_t = residual_out_t.to(next_residual_ref.dtype)
        ret["triton_us"] = triton_us
        checkAllclose(post_mix_ref, h_post_t, msg="triton/post_mix")
        checkAllclose(comb_mix_ref, h_res_t, msg="triton/comb_mix")
        ret["triton_fused_err"] = checkAllclose(
            layer_input_ref, layer_input_t, msg="triton/layer_input", rtol=2e-2
        )
        checkAllclose(next_residual_ref, residual_out_t, msg="triton/next_residual")
    elif not _HAS_TRITON_MHC_POST_PRE:
        aiter.logger.info("skip Triton mhc_post_pre: import unavailable")

    if large_m:
        if m < LARGE_M_MIN:
            aiter.logger.info("skip large_m_us: m=%s < %s", m, LARGE_M_MIN)
        elif get_gfx_runtime() != "gfx950":
            aiter.logger.info(
                "skip large_m_us: gfx=%s (gfx950 only)", get_gfx_runtime()
            )
        elif mhc_fused_post_pre_large_m is None:
            aiter.logger.info("skip large_m_us: mhc_fused_post_pre_large_m unavailable")
        else:
            (_, _, layer_input_large_m, _), large_m_us = run_perftest(
                mhc_fused_post_pre_large_m,
                layer_input,
                residual_in,
                post_layer_mix,
                comb_res_mix,
                fn_gemm,
                hc_scale,
                hc_base,
                w_preshuffle_bf16=w_preshuffle_bf16,
                res_preshuffle=False,
                **hip_kwargs,
            )
            ret["large_m_us"] = large_m_us
            ret["hip_large_m_err"] = checkAllclose(
                layer_input_ref, layer_input_large_m, msg="large_m/layer_input"
            )

    return ret


# Stream-cancellation cases of mhc_post_pre: (amplitude, pre-gate logits, perturb).
# The share of layer-output elements outside tolerance depends on how the stream
# fold is rounded; the cases are asserted where that share has been measured.
MHC_CANCELLATION_GFX = ["gfx950"]
MHC_CANCELLATION_CASES = [
    (amplitude, gate_logits, perturb)
    for amplitude in (1.0, 16.0, 256.0)
    for gate_logits in (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1e-4, 0.0, -1e-4),
        (0.0, 0.0, 0.0, -1e-6),
    )
    for perturb in (False, True)
]


@benchmark()
def test_mhc_post_pre_cancellation(
    m,
    hidden_size,
    hc_mult,
    amplitude,
    gate_logits,
    perturb,
    w_preshuffle_bf16=False,
    dtype=dtypes.bf16,
    seed=0,
):
    """Opposing residual stream pairs: the layer output is what their cancellation leaves.

    Streams 1 and 3 are the negated streams 0 and 2, the post step is the identity
    and the projection is zero, so next_residual equals residual_in and the pre-gate
    logits are exactly ``gate_logits``. The fused RMSNorm scales the rounding residue
    of the stream fold to O(1), so a fold that rounds differently shows up as
    layer-output elements outside tolerance. Unfused HIP and the public default call
    are both held to the reference at the standard tolerances.
    """
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    gen = make_generator(seed)
    residual_in = fill((m, hc_mult, hidden_size), "norm", gen, dtype=dtype)
    residual_in[:, 0].mul_(amplitude)
    residual_in[:, 1].copy_(-residual_in[:, 0])
    residual_in[:, 2].mul_(amplitude)
    residual_in[:, 3].copy_(-residual_in[:, 2])
    if perturb:
        residual_in[:, 3, ::257].add_(amplitude / 128)
    layer_input = torch.zeros((m, hidden_size), dtype=dtype)
    post_layer_mix = torch.zeros((m, hc_mult, 1), dtype=dtypes.fp32)
    comb_res_mix = torch.eye(hc_mult, dtype=dtypes.fp32).repeat(m, 1, 1)
    fn = torch.zeros((hc_mult3, hc_mult * hidden_size), dtype=dtypes.fp32)
    hc_scale = torch.zeros((3,), dtype=dtypes.fp32)
    hc_base = torch.zeros((hc_mult3,), dtype=dtypes.fp32)
    hc_base[:hc_mult] = torch.tensor(gate_logits, dtype=dtypes.fp32)
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 1.0,
        "sinkhorn_repeat": 20,
        "norm_weight": torch.ones((hidden_size,), dtype=dtype),
        "norm_eps": 1e-6,
    }
    refs = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
    )
    fn_gemm = aiter.mhc_shuffle_fn(fn) if w_preshuffle_bf16 else fn
    inputs = (
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn_gemm,
        hc_scale,
        hc_base,
    )
    candidates = {
        "unfused": lambda: mhc_post_pre_unfused_hip(
            *inputs, w_preshuffle_bf16=int(w_preshuffle_bf16), **extra_args
        ),
        # The public call without force_fused.
        "default": lambda: aiter.mhc_fused_post_pre(
            *inputs, w_preshuffle_bf16=w_preshuffle_bf16, backend="hip", **extra_args
        ),
    }
    if dtype == dtypes.bf16 and mhc_gram_route_available(m):
        # The default call on the Gram route, whatever its tuned table lists: one
        # producer split per 128 hidden elements, four finisher segments per row.
        def gram():
            outs, launches = mhc_post_pre_gram_route(
                *inputs,
                split_k=hidden_size // 128,
                parts=4,
                w_preshuffle_bf16=w_preshuffle_bf16,
                **extra_args,
            )
            assert launches == [(hidden_size // 128, 4)], launches
            return outs

        candidates["gram"] = gram
    ret = {}
    for name, candidate in candidates.items():
        # A numerical case: a short timing run is enough for the table.
        outs, us = run_perftest(candidate, num_iters=11, num_warmup=1)
        for what, ref, out in zip(
            ("post_mix", "comb_mix", "layer_input", "next_residual"), refs, outs
        ):
            err = assertAllclose(ref, out, msg=f"cancellation/{name}/{what}")
            if what == "layer_input":
                ret[f"{name}_err"] = err
        ret[f"{name}_us"] = us
    # No other candidate may leave the reference further than post + pre does.
    for name in candidates:
        assert ret[f"{name}_err"] <= ret["unfused_err"] + 5e-3, (
            f"cancellation: {name} {ret[f'{name}_err']:.3%} of elements outside "
            f"tolerance against {ret['unfused_err']:.3%} for post + pre"
        )
    return ret


def mhc_gram_route_available(m):
    """Whether the Gram route of mhc_fused_post_pre can serve m rows on this device:
    gfx950 with 256 CUs and no CU mask, a decode row count."""
    from aiter.ops.mhc import MHC_GRAM_M_MAX_GFX950

    return (
        get_gfx_runtime() == "gfx950"
        and get_cu_num() == 256
        and not any(
            name in os.environ for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")
        )
        and 0 < m <= MHC_GRAM_M_MAX_GFX950
    )


def mhc_post_pre_gram_route(
    *inputs,
    split_k=None,
    parts=None,
    w_preshuffle_bf16=False,
    table_m=None,
    **extra_args,
):
    """The default mhc_fused_post_pre call. With ``split_k`` and ``parts`` the table of
    the Gram route is replaced by one row, (split_k, parts) for ``table_m`` rows
    (default: the rows of this call), so that the call does not depend on the tuned
    table; without them the tuned table is used. Returns the outputs and the
    (split_k, parts) of every launch of the Gram kernels."""
    from aiter.ops import mhc as mhc_ops

    m, hidden_size = inputs[0].shape
    table_m = m if table_m is None else table_m
    key = (hidden_size, bool(w_preshuffle_bf16))
    launches = []
    gram_kernels = mhc_ops.mhc_post_pre_gram_gfx950

    def record(*args, **kwargs):
        launches.append((args[4].size(0), args[18]))
        return gram_kernels(*args, **kwargs)

    table = mhc_ops.MHC_GRAM_CONFIG_GFX950_256
    if split_k is not None:
        table = {key: ((table_m, table_m, split_k, parts),)}
    with (
        patch.object(mhc_ops, "MHC_GRAM_CONFIG_GFX950_256", table),
        patch.object(mhc_ops, "mhc_post_pre_gram_gfx950", record),
    ):
        outs = aiter.mhc_fused_post_pre(
            *inputs, w_preshuffle_bf16=w_preshuffle_bf16, **extra_args
        )
    return outs, launches


def mhc_gram_case(m, hidden_size, hc_mult, w_preshuffle_bf16, dtype, seed):
    """Inputs of one Gram route test (the call's tensors, its keyword arguments) and
    the reference outputs. Weights and mixes are random and differ per row, so a
    projection or a mix read from the wrong row, slab or segment fails the comparison
    with the reference."""
    from aiter.ops import mhc as mhc_ops

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    gen = make_generator(seed)
    layer_input = fill((m, hidden_size), "norm", gen, dtype=dtype)
    residual_in = fill((m, hc_mult, hidden_size), "norm", gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), "norm", gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), "norm", gen, dtype=dtypes.fp32)
    fn = fill((hc_mult3, hc_mult * hidden_size), "norm", gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    # Four different epsilons, large enough that a swapped one fails the comparison.
    extra_args = {
        "rms_eps": 0.3,
        "hc_pre_eps": 0.2,
        "hc_sinkhorn_eps": 0.1,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_weight": fill((hidden_size,), "norm", gen, dtype=dtype),
        "norm_eps": 0.5,
    }
    inputs = (layer_input, residual_in, post_layer_mix, comb_res_mix)
    refs = mhc_post_pre_ref(*inputs, fn, hc_scale, hc_base, **extra_args)
    inputs += (
        mhc_ops.mhc_shuffle_fn(fn) if w_preshuffle_bf16 else fn,
        hc_scale,
        hc_base,
    )
    return inputs, extra_args, refs


def mhc_gram_errors(refs, outs, msg, tol_err_ratio=1e-4):
    """Compare the four outputs with the reference; the layer output's mismatch ratio.
    The default bound is for the Gram kernels: one wrong row of the largest row count
    here is 1e-3 of the elements."""
    ret = {}
    for what, ref, out in zip(
        ("post_mix", "comb_mix", "layer_input", "next_residual"), refs, outs
    ):
        err = assertAllclose(ref, out, tol_err_ratio=tol_err_ratio, msg=f"{msg}/{what}")
        if what == "layer_input":
            ret["err"] = err
    return ret


@benchmark()
def test_mhc_post_pre_gram_dispatch(
    m, hidden_size, hc_mult, slab, parts, w_preshuffle_bf16, dtype=dtypes.bf16, seed=0
):
    """gfx950 with 256 CUs: the Gram route of the default call with fused RMSNorm.

    With a table row for m rows (one producer split per ``slab`` hidden elements,
    ``parts`` finisher segments per row) the call must launch the Gram kernels once
    with that configuration; with the row moved to m + 1 rows it must not launch
    them.
    """
    inputs, extra_args, refs = mhc_gram_case(
        m, hidden_size, hc_mult, w_preshuffle_bf16, dtype, seed
    )
    route = {
        "split_k": hidden_size // slab,
        "parts": parts,
        "w_preshuffle_bf16": w_preshuffle_bf16,
    }
    outs, launches = mhc_post_pre_gram_route(*inputs, **route, **extra_args)
    assert launches == [(route["split_k"], parts)], launches
    _, launches = mhc_post_pre_gram_route(*inputs, **route, table_m=m + 1, **extra_args)
    assert not launches, launches
    return mhc_gram_errors(refs, outs, "gram dispatch")


@benchmark()
def test_mhc_post_pre_gram_table(
    m, hidden_size, hc_mult, w_preshuffle_bf16, dtype=dtypes.bf16, seed=0
):
    """gfx950 with 256 CUs: the tuned table of the Gram route.

    At a row count of MHC_GRAM_CONFIG_GFX950_256 the default call must launch the
    Gram kernels once with that row's configuration; at any other row count it must
    not launch them. The outputs are compared with the reference: on the Gram route
    with the bound of the dispatch test, on the existing path with the bound that the
    other candidates of this file use.
    """
    from aiter.ops import mhc as mhc_ops

    inputs, extra_args, refs = mhc_gram_case(
        m, hidden_size, hc_mult, w_preshuffle_bf16, dtype, seed
    )
    row = mhc_ops._mhc_gram_config_gfx950_256(m, hidden_size, w_preshuffle_bf16)
    outs, launches = mhc_post_pre_gram_route(
        *inputs, w_preshuffle_bf16=w_preshuffle_bf16, **extra_args
    )
    assert launches == ([tuple(row)] if row else []), (launches, row)
    ret = {"route": "gram" if row else "existing"}
    if row:
        ret["split_k"], ret["parts"] = row
    ret.update(
        mhc_gram_errors(refs, outs, "gram table", tol_err_ratio=1e-4 if row else 0.05)
    )
    return ret


@benchmark()
def test_mhc_post_pre_prefill_dispatch(
    m, hidden_size, hc_mult, dtype=dtypes.bf16, seed=0
):
    """gfx950 with 256 CUs: the path of the default call for packed weights.

    Inside the row range of ``aiter.ops.mhc`` (MHC_FUSED_PREFILL_*_GFX950) the call
    must launch the fused GEMM once, with 32-row, 64-wide tiles; outside it must not
    launch it. For hidden size 7168 the two-kernel route takes the row counts from
    MHC_GFX950_PREFILL_M_MIN to MHC_GFX950_PREFILL_M_MAX instead. Weights and mixes
    are random, so a projection or a mix read from the wrong row fails the comparison
    with the reference.
    """
    from aiter.ops import mhc as mhc_ops

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    gen = make_generator(seed)
    layer_input = fill((m, hidden_size), "norm", gen, dtype=dtype)
    residual_in = fill((m, hc_mult, hidden_size), "norm", gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), "norm", gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), "norm", gen, dtype=dtypes.fp32)
    fn = fill((hc_mult3, hc_mult * hidden_size), "norm", gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_weight": fill((hidden_size,), "norm", gen, dtype=dtype),
        "norm_eps": 1e-6,
    }
    refs = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
    )

    launches = []
    routed = []
    fused_gemm = mhc_ops.mhc_fused_post_pre_gemm_sqrsum
    post_project = mhc_ops.mhc_post_project_gfx950

    def record(*args, **kwargs):
        # (split_k, tile_m, tile_n, tile_k) of the launch
        launches.append((args[1].size(0), *args[8:11]))
        return fused_gemm(*args, **kwargs)

    def record_route(*args, **kwargs):
        routed.append(args[3].size(0))
        return post_project(*args, **kwargs)

    with (
        patch.object(mhc_ops, "mhc_fused_post_pre_gemm_sqrsum", record),
        patch.object(mhc_ops, "mhc_post_project_gfx950", record_route),
    ):
        outs = aiter.mhc_fused_post_pre(
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
            mhc_ops.mhc_shuffle_fn(fn),
            hc_scale,
            hc_base,
            w_preshuffle_bf16=True,
            **extra_args,
        )
    in_range = (
        hidden_size in mhc_ops.MHC_FUSED_PREFILL_HIDDEN_SIZES_GFX950
        and mhc_ops.MHC_FUSED_PREFILL_M_MIN_GFX950
        <= m
        <= mhc_ops.MHC_FUSED_PREFILL_M_MAX_GFX950
    )
    two_kernel = (
        hidden_size == 7168
        and mhc_ops.MHC_GFX950_PREFILL_M_MIN <= m <= mhc_ops.MHC_GFX950_PREFILL_M_MAX
    )
    if two_kernel:
        assert routed == [m] and not launches, (routed, launches)
    elif in_range:
        assert not routed, routed
        assert len(launches) == 1 and launches[0][1:] == (32, 32, 64), launches
    else:
        assert not routed and not launches, (routed, launches)
    ret = {
        "fused_gemm": str(launches[0]) if launches else "not launched",
        "two_kernel": "launched" if routed else "not launched",
    }
    for what, ref, out in zip(
        ("post_mix", "comb_mix", "layer_input", "next_residual"), refs, outs
    ):
        # One wrong row of the largest row count here is 1.2e-4 of the elements.
        err = assertAllclose(
            ref, out, tol_err_ratio=1e-5, msg=f"prefill dispatch/{what}"
        )
        if what == "layer_input":
            ret["err"] = err
    return ret


def test_mhc_prefill_launchers_raise():
    """The two launchers of the gfx950 packed prefill route answer a tensor they
    cannot take with RuntimeError. They are exported, so a direct call with a
    wrong shape must not end the process."""
    from aiter.ops import mhc as mhc_ops

    m, eps = 4096, 1e-6
    f32, b16, i32 = dtypes.fp32, dtypes.bf16, torch.int32

    def z(dtype, *shape):
        return torch.zeros(shape, dtype=dtype, device="cuda")

    # hidden 4096 where the producer takes 7168, and a (m, 4, 2) post mix
    producer = [z(f32, 4, m, 32), z(f32, 4, m, 4), z(b16, m, 4, 4096), z(b16, m, 4096)]
    producer += [z(b16, m, 4, 4096), z(i32, 24, 16384), z(f32, m, 4), z(f32, m, 4, 4)]
    finisher = [z(f32, m, 4, 2), z(f32, m, 4, 4), z(b16, m, 7168), z(f32, 4, m, 32)]
    finisher += [z(f32, 4, m, 4), z(f32, 3), z(f32, 24), z(b16, m, 4, 7168)]
    finisher += [z(b16, 7168), eps, eps, eps, eps, 2.0, 20]
    calls = {"mhc_post_project_gfx950": producer, "mhc_finalize_gfx950": finisher}
    for name, args in calls.items():
        try:
            getattr(mhc_ops, name)(*args)
        except RuntimeError as e:
            assert "mHC gfx950" in str(e), str(e)
        else:
            raise AssertionError(f"{name} accepted a tensor of the wrong shape")


@benchmark()
def test_mhc_post_pre_decode_dispatch(
    m, hidden_size, hc_mult, w_preshuffle_bf16=False, dtype=dtypes.bf16, seed=0
):
    """gfx950 with 256 CUs: the setting of the fused post+pre GEMM below 1024 rows.

    At a row count of MHC_FUSED_DECODE_CONFIG_GFX950_256 the default call must launch
    the fused GEMM once with that row's split count and tiles; at every other row
    count with the setting that get_mhc_fused_post_pre_config computes. A row count
    of the table is called as it is, so the test also fails when the Gram route
    answers there. The other row counts are called with the Gram route's table
    emptied, because that route lists some of them. Weights and mixes are random and
    the outputs are compared with the reference.
    """
    from aiter.ops import mhc as mhc_ops

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    gen = make_generator(seed)
    layer_input = fill((m, hidden_size), "norm", gen, dtype=dtype)
    residual_in = fill((m, hc_mult, hidden_size), "norm", gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), "norm", gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), "norm", gen, dtype=dtypes.fp32)
    fn = fill((hc_mult3, hc_mult * hidden_size), "norm", gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_weight": fill((hidden_size,), "norm", gen, dtype=dtype),
        "norm_eps": 1e-6,
    }
    refs = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
    )

    launches = []
    fused_gemm = mhc_ops.mhc_fused_post_pre_gemm_sqrsum

    def record(*args, **kwargs):
        launches.append((args[1].size(0), *args[8:11]))
        return fused_gemm(*args, **kwargs)

    row = mhc_ops._mhc_fused_decode_config_gfx950_256(m, hidden_size, w_preshuffle_bf16)
    # A row count of the table keeps the Gram route's table: the default call has to
    # reach the fused GEMM there. Next to the rows that route may answer first, so
    # its table is empty for those calls.
    gram_rows = dict(mhc_ops.MHC_GRAM_CONFIG_GFX950_256) if row else {}
    with patch.object(mhc_ops, "mhc_fused_post_pre_gemm_sqrsum", record), patch.dict(
        mhc_ops.MHC_GRAM_CONFIG_GFX950_256, gram_rows, clear=True
    ):
        outs = aiter.mhc_fused_post_pre(
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
            mhc_ops.mhc_shuffle_fn(fn) if w_preshuffle_bf16 else fn,
            hc_scale,
            hc_base,
            w_preshuffle_bf16=w_preshuffle_bf16,
            **extra_args,
        )
    # No row: the setting that the config function computes for this row count.
    expected = row or mhc_ops.get_mhc_fused_post_pre_config(
        m, hidden_size, w_preshuffle_bf16
    )
    assert launches == [tuple(expected)], (launches, expected)
    ret = {"split_k": expected[0], "tile_m": expected[1], "tile_k": expected[3]}
    ret["setting"] = "table" if row else "computed"
    for what, ref, out in zip(
        ("post_mix", "comb_mix", "layer_input", "next_residual"), refs, outs
    ):
        err = assertAllclose(ref, out, msg=f"decode dispatch/{what}")
        if what == "layer_input":
            ret["err"] = err
    return ret


@benchmark()
def test_mhc_post_pre_store_dispatch(
    m, hidden_size, hc_mult, w_preshuffle_bf16=False, dtype=dtypes.bf16, seed=0
):
    """gfx950 with 256 CUs: the store policy of the unfused post + pre of the default call.

    With FP32 projection weights, from 8 * 256 + 1 rows up to
    MHC_POST_PRE_REGULAR_STORE_M_MAX_GFX950 the call must pass regular stores (0) to
    ``mhc_post``; outside that range, and with packed weights, it must leave the
    policy to ``mhc_post`` (-1). Weights and mixes are random and the outputs are
    compared with the reference.
    """
    from aiter.ops import mhc as mhc_ops

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    gen = make_generator(seed)
    layer_input = fill((m, hidden_size), "norm", gen, dtype=dtype)
    residual_in = fill((m, hc_mult, hidden_size), "norm", gen, dtype=dtype)
    post_layer_mix = fill((m, hc_mult, 1), "norm", gen, dtype=dtypes.fp32)
    comb_res_mix = fill((m, hc_mult, hc_mult), "norm", gen, dtype=dtypes.fp32)
    fn = fill((hc_mult3, hc_mult * hidden_size), "norm", gen, dtype=dtypes.fp32)
    hc_scale = fill((3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    hc_base = fill((hc_mult3,), "norm", gen, dtype=dtypes.fp32) * 0.1
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
        "norm_weight": fill((hidden_size,), "norm", gen, dtype=dtype),
        "norm_eps": 1e-6,
    }
    refs = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **extra_args,
    )

    stores = []
    post = mhc_ops.mhc_post

    def record(*args, **kwargs):
        stores.append(args[5] if len(args) > 5 else kwargs.get("store_nt", -1))
        return post(*args, **kwargs)

    with patch.object(mhc_ops, "mhc_post", record):
        outs = aiter.mhc_fused_post_pre(
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
            mhc_ops.mhc_shuffle_fn(fn) if w_preshuffle_bf16 else fn,
            hc_scale,
            hc_base,
            w_preshuffle_bf16=w_preshuffle_bf16,
            **extra_args,
        )
    regular = (
        not w_preshuffle_bf16
        and 8 * 256 < m <= mhc_ops.MHC_POST_PRE_REGULAR_STORE_M_MAX_GFX950
    )
    assert stores == [0 if regular else -1], stores
    ret = {"store_nt": stores[0]}
    for what, ref, out in zip(
        ("post_mix", "comb_mix", "layer_input", "next_residual"), refs, outs
    ):
        err = assertAllclose(ref, out, tol_err_ratio=1e-5, msg=f"store dispatch/{what}")
        if what == "layer_input":
            ret["err"] = err
    return ret


def test_mhc_res_layout(m, hidden_size, hc_mult, dtype):
    from aiter.ops.mhc import (
        mhc_res_ks,
        mhc_res_repeat,
        mhc_res_shuffle,
        mhc_res_unshuffle,
    )

    # The block width the ops pick by default for these rows / hidden size.
    ks = mhc_res_ks(m, hidden_size)
    hidden_states = torch.randn(m, hidden_size, dtype=dtype)
    residual = hidden_states.unsqueeze(1).repeat(1, hc_mult, 1)

    def repeat_ref(x, mult):
        rows, width = x.shape
        return (
            x.view(rows, width // ks, ks)
            .permute(1, 0, 2)
            .unsqueeze(1)
            .expand(-1, mult, -1, -1)
            .contiguous()
            .view(rows, mult, width)
        )

    def shuffle_ref(x):
        rows, mult, width = x.shape
        return (
            x.view(rows, mult, width // ks, ks)
            .permute(2, 1, 0, 3)
            .contiguous()
            .view(rows, mult, width)
        )

    def unshuffle_ref(x):
        rows, mult, width = x.shape
        return (
            x.view(width // ks, mult, rows, ks)
            .permute(2, 1, 0, 3)
            .contiguous()
            .view(rows, mult, width)
        )

    repeated_ref, torch_repeat_us = run_perftest(repeat_ref, hidden_states, hc_mult)
    repeated_out, flydsl_repeat_us = run_perftest(
        mhc_res_repeat, hidden_states, hc_mult, True
    )
    assert torch.equal(repeated_out, repeated_ref)

    shuffled_ref, torch_shuffle_us = run_perftest(shuffle_ref, residual)
    shuffled_out, flydsl_shuffle_us = run_perftest(mhc_res_shuffle, residual)
    assert torch.equal(shuffled_out, shuffled_ref)

    unshuffled_ref, torch_unshuffle_us = run_perftest(unshuffle_ref, shuffled_ref)
    unshuffled_out, flydsl_unshuffle_us = run_perftest(mhc_res_unshuffle, shuffled_out)
    assert torch.equal(unshuffled_out, unshuffled_ref)
    assert torch.equal(unshuffled_out, residual)

    return {
        "M": m,
        "dtype": str(dtype),
        "N": hidden_size,
        "torch_repeat_us": torch_repeat_us,
        "flydsl_repeat_us": flydsl_repeat_us,
        "repeat_speedup": torch_repeat_us / flydsl_repeat_us,
        "torch_shuffle_us": torch_shuffle_us,
        "flydsl_shuffle_us": flydsl_shuffle_us,
        "shuffle_speedup": torch_shuffle_us / flydsl_shuffle_us,
        "torch_unshuffle_us": torch_unshuffle_us,
        "flydsl_unshuffle_us": flydsl_unshuffle_us,
        "unshuffle_speedup": torch_unshuffle_us / flydsl_unshuffle_us,
    }


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp16, bf16}",
    default=[dtypes.bf16],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 32, 64, 128, 256, 512, 1024, 2048, 8192, 65536],
    help="""M.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    "--hidden_size",
    type=int,
    nargs="*",
    choices=[4096, 5120, 7168],
    default=[4096, 5120, 7168],
    help="""hidden_size.
    e.g.: -n 5120""",
)
_mode_group = parser.add_mutually_exclusive_group()
_mode_group.add_argument(
    "--hc_head",
    action="store_true",
    help="Test mhc_pre for hc_head only (mutually exclusive with --fuse_rmsnorm).",
)
_mode_group.add_argument(
    "--fuse_rmsnorm",
    action="store_true",
    help="Fuse RMSNorm into mhc_pre / mhc_post_pre HIP paths (mutually exclusive with --hc_head).",
)
parser.add_argument(
    "--largeM",
    action="store_true",
    help="In mhc_post_pre summary, add large_m_us / hip_large_m_err columns "
    "(gfx950, M>1024, mhc_fused_post_pre_large_m).",
)
parser.add_argument(
    "--w_preshuffle_bf16",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable BF16 hi/lo GEMM with pre-packed weights, independently of residual layout.",
)
parser.add_argument(
    "--res_shuffle",
    "--res_preshuffle",
    dest="res_preshuffle",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Request shuffled residual input/output when the runtime policy uses "
    "the fused path (gfx1250 only), independently of GEMM compute.",
)
parser.add_argument(
    "--cancellation_m",
    type=int,
    nargs="*",
    default=[64, 8192],
    help="""M of the mhc_post_pre stream-cancellation cases (decode and prefill).
    e.g.: --cancellation_m 64""",
)
add_data_init_args(parser, default_dist="norm")

args = parser.parse_args()
if args.res_preshuffle and get_gfx_runtime() != "gfx1250":
    parser.error("residual shuffle is only supported on gfx1250; use --no-res_shuffle")

if args.res_preshuffle and not args.hc_head:
    from aiter.ops.mhc import mhc_res_shuffle_enabled

    df = []
    for dtype in args.dtype:
        for hidden_size in args.hidden_size:
            for m in args.m:
                if not mhc_res_shuffle_enabled(m):
                    continue
                for hc_mult in [4]:
                    df.append(
                        test_mhc_res_layout(
                            m=m,
                            hidden_size=hidden_size,
                            hc_mult=hc_mult,
                            dtype=dtype,
                        )
                    )
    print_json_table("mhc_res_layout summary", df)


df = []
for dtype in args.dtype:
    for data_init in args.data_init:
        for hidden_size in args.hidden_size:
            for m in args.m:
                for hc_mult in [4]:
                    ret = test_mhc_pre(
                        m=m,
                        hidden_size=hidden_size,
                        hc_mult=hc_mult,
                        test_hc_head=args.hc_head,
                        fuse_rmsnorm=args.fuse_rmsnorm,
                        w_preshuffle_bf16=args.w_preshuffle_bf16,
                        dtype=dtype,
                        data_init=data_init,
                        seed=args.seed,
                    )
                    df.append(ret)
print_json_table("mhc_pre summary", df)

if not args.hc_head:
    df = []
    for dtype in args.dtype:
        for data_init in args.data_init:
            for hidden_size in sorted(set(args.hidden_size) | {512, 1024}):
                for m in args.m:
                    for hc_mult in [4]:
                        ret = test_mhc_post(
                            m=m,
                            hidden_size=hidden_size,
                            hc_mult=hc_mult,
                            dtype=dtype,
                            data_init=data_init,
                            seed=args.seed,
                        )
                        df.append(ret)
    print_json_table("mhc_post summary", df)

    # The query sweep, then the reduce at the 1024- and 512-column tiles (1024 and
    # 2560 take the 512 one; gfx942 takes it for every size) in both dtypes.
    print_json_table("mhc_post support queries", test_mhc_post_support_queries())
    print_json_table("mhc_post_reduce guards", test_mhc_post_reduce_operand_guards())

    df = []
    for dtype in (dtypes.bf16, dtypes.fp16):
        for data_init in args.data_init:
            for hidden_size in sorted(set(args.hidden_size) | {1024, 2560}):
                for m in args.m:
                    for reduce in ("pre_mix", "mean"):
                        ret = test_mhc_post_reduce(
                            m=m,
                            hidden_size=hidden_size,
                            hc_mult=4,
                            reduce=reduce,
                            dtype=dtype,
                            data_init=data_init,
                            seed=args.seed,
                        )
                        df.append(ret)
    print_json_table("mhc_post_reduce summary", df)

    df = []
    for dtype in args.dtype:
        for data_init in args.data_init:
            for hidden_size in args.hidden_size:
                for m in args.m:
                    for hc_mult in [4]:
                        ret = test_mhc_post_pre(
                            m=m,
                            hidden_size=hidden_size,
                            hc_mult=hc_mult,
                            fuse_rmsnorm=args.fuse_rmsnorm,
                            large_m=args.largeM,
                            w_preshuffle_bf16=args.w_preshuffle_bf16,
                            res_preshuffle=args.res_preshuffle,
                            dtype=dtype,
                            data_init=data_init,
                            seed=args.seed,
                        )
                        if ret.get("skipped"):
                            continue
                        df.append(ret)
    if df:
        print_json_table("mhc_post_pre summary", df)
    else:
        aiter.logger.info("mhc_post_pre: all cases skipped")

    if get_gfx_runtime() in MHC_CANCELLATION_GFX:
        # Both projection modes: models that pack fn take a different dispatch.
        df = []
        for (
            dtype,
            hidden_size,
            m,
            packed,
            (amplitude, gate_logits, perturb),
        ) in itertools.product(
            args.dtype,
            args.hidden_size,
            args.cancellation_m,
            (False, True),
            MHC_CANCELLATION_CASES,
        ):
            df.append(
                test_mhc_post_pre_cancellation(
                    m=m,
                    hidden_size=hidden_size,
                    hc_mult=4,
                    amplitude=amplitude,
                    gate_logits=gate_logits,
                    perturb=perturb,
                    w_preshuffle_bf16=packed,
                    dtype=dtype,
                    seed=args.seed,
                )
            )
        print_json_table("mhc_post_pre cancellation summary", df)
    else:
        aiter.logger.warning(
            "skip mhc_post_pre cancellation cases: asserted on %s only, this is %s",
            MHC_CANCELLATION_GFX,
            get_gfx_runtime(),
        )

    if (
        get_gfx_runtime() == "gfx950"
        and get_cu_num() == 256
        and not any(
            name in os.environ for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK")
        )
    ):
        # Just below the range, its first row count (odd, partial row tile), the
        # row counts around the two-kernel route of hidden size 7168 (4095 and
        # 16385 outside, 4099 with a partial row tile and an odd last row inside),
        # and a full grid.
        df = [
            test_mhc_post_pre_prefill_dispatch(
                m=m, hidden_size=hidden_size, hc_mult=4, seed=args.seed
            )
            for hidden_size, m in itertools.product(
                args.hidden_size, (2048, 2049, 4095, 4099, 8192, 16385)
            )
        ]
        print_json_table("mhc_post_pre prefill dispatch summary", df)
        test_mhc_prefill_launchers_raise()
    else:
        aiter.logger.warning(
            "skip mhc_post_pre prefill dispatch cases: unmasked 256-CU gfx950 only"
        )

    from aiter.ops import mhc as mhc_ops

    # The tuned table of the Gram route: ascending row ranges, legal configurations.
    for (hidden_size, _), rows in mhc_ops.MHC_GRAM_CONFIG_GFX950_256.items():
        last = 0
        for m_first, m_last, split_k, parts in rows:
            slab = hidden_size // split_k
            assert last < m_first <= m_last <= mhc_ops.MHC_GRAM_M_MAX_GFX950, rows
            assert slab * split_k == hidden_size, rows
            assert slab == 64 or slab % 128 == 0, rows
            assert hidden_size % (8 * parts) == 0, rows
            last = m_last

    if mhc_gram_route_available(1):
        # (m, slab, parts): one row, partial producer tiles, both MFMA widths, up to
        # four producer steps per split, 1 to 8 segments, the largest row count.
        df = [
            test_mhc_post_pre_gram_dispatch(
                m=m,
                hidden_size=hidden_size,
                hc_mult=4,
                slab=slab,
                parts=parts,
                w_preshuffle_bf16=packed,
                seed=args.seed,
            )
            for hidden_size, packed, (m, slab, parts) in itertools.product(
                args.hidden_size,
                (False, True),
                (
                    (1, 64, 8),
                    (5, 128, 4),
                    (33, 256, 2),
                    (64, 128, 4),
                    (129, 512, 1),
                    (mhc_ops.MHC_GRAM_M_MAX_GFX950, 128, 1),
                ),
            )
        ]
        print_json_table("mhc_post_pre gram dispatch summary", df)
        # The tuned table: the first and the last row count of every row, and the row
        # counts just outside it (in the next row, or on the existing path).
        df = []
        for (hidden_size, packed), rows in sorted(
            mhc_ops.MHC_GRAM_CONFIG_GFX950_256.items()
        ):
            listed = {m for m_first, m_last, *_ in rows for m in (m_first, m_last)}
            nearby = {
                m for m_first, m_last, *_ in rows for m in (m_first - 1, m_last + 1)
            }
            limit = mhc_ops.MHC_GRAM_M_MAX_GFX950
            for m in sorted(listed | {m for m in nearby if 0 < m <= limit}):
                df.append(
                    test_mhc_post_pre_gram_table(
                        m=m,
                        hidden_size=hidden_size,
                        hc_mult=4,
                        w_preshuffle_bf16=packed,
                        seed=args.seed,
                    )
                )
        print_json_table("mhc_post_pre gram table summary", df)
    else:
        aiter.logger.warning(
            "skip mhc_post_pre gram dispatch cases: unmasked 256-CU gfx950 only"
        )


# The table of the gfx950 decode settings: ascending row ranges below the fused bound,
# settings that the fused GEMM has, and no row count that the Gram route lists (the
# default call with fused RMSNorm would never ask for its setting).
for key, rows in aiter.ops.mhc.MHC_FUSED_DECODE_CONFIG_GFX950_256.items():
    hidden_size = key[0]
    gram_rows = aiter.ops.mhc.MHC_GRAM_CONFIG_GFX950_256.get(key, ())
    last = 0
    for m_first, m_last, split_k, tile_m, tile_k in rows:
        assert last < m_first <= m_last < 1024, rows
        assert tile_m in (16, 32) and tile_k in (32, 64), rows
        assert hidden_size % (split_k * tile_k) == 0, rows
        assert hidden_size // split_k >= 2 * tile_k, rows
        for gram_first, gram_last, *_ in gram_rows:
            assert m_last < gram_first or gram_last < m_first, (key, rows, gram_rows)
        last = m_last

if get_gfx_runtime() == "gfx950" and get_cu_num() == 256:
    # Per table row: its first and last row count, and the row counts next to them
    # that no row lists (they keep the computed setting).
    df = []
    for (hidden_size, packed), rows in sorted(
        aiter.ops.mhc.MHC_FUSED_DECODE_CONFIG_GFX950_256.items()
    ):
        listed = {m for m_first, m_last, *_ in rows for m in (m_first, m_last)}
        nearby = {m for m_first, m_last, *_ in rows for m in (m_first - 1, m_last + 1)}
        for m in sorted(listed | {m for m in nearby if 0 < m < 1024}):
            df.append(
                test_mhc_post_pre_decode_dispatch(
                    m=m,
                    hidden_size=hidden_size,
                    hc_mult=4,
                    w_preshuffle_bf16=packed,
                    seed=args.seed,
                )
            )
    print_json_table("mhc_post_pre decode dispatch summary", df)
else:
    aiter.logger.warning("skip mhc_post_pre decode dispatch cases: 256-CU gfx950 only")


if (
    get_gfx_runtime() == "gfx950"
    and get_cu_num() == 256
    and not any(name in os.environ for name in ("HSA_CU_MASK", "ROC_GLOBAL_CU_MASK"))
):
    # FP32 projection weights keep post + pre at these row counts: the row count
    # below the regular-store range, its first and last row counts, and the first
    # one above it.
    store_m_max = aiter.ops.mhc.MHC_POST_PRE_REGULAR_STORE_M_MAX_GFX950
    df = [
        test_mhc_post_pre_store_dispatch(
            m=m, hidden_size=hidden_size, hc_mult=4, seed=args.seed
        )
        for hidden_size, m in itertools.product(
            args.hidden_size, (2048, 2049, 4099, store_m_max, store_m_max + 1)
        )
    ]
    # Packed weights keep post + pre with the store policy of mhc_post.
    df.append(
        test_mhc_post_pre_store_dispatch(
            m=4099, hidden_size=5120, hc_mult=4, w_preshuffle_bf16=True, seed=args.seed
        )
    )
    print_json_table("mhc_post_pre store dispatch summary", df)
else:
    aiter.logger.warning(
        "skip mhc_post_pre store dispatch cases: unmasked 256-CU gfx950 only"
    )
