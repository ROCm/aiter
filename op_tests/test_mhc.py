# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
import torch
import aiter
from aiter import dtypes
import argparse
import pandas as pd
import logging
from aiter.jit.utils.chip_info import get_cu_num

logging.getLogger("KernelCache").disabled = True

torch.set_default_device("cuda")
# torch.cuda.manual_seed_all(0)
# torch.set_printoptions(precision=3, linewidth=200, sci_mode=False)


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

        with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hc_mult3 = fn.shape[0]
    hc_hidden_size = fn.shape[1]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    out = residual_flat @ fn.T
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

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

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_pre_hip(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    m = residual.size(0)
    hc_mult = residual.size(-2)
    hc_mult3 = fn.size(0)
    hc_hidden_size = fn.size(1)

    prefetch_stages = 2
    tile_m = 16 * 4
    tile_k_tg_dict = {
        128: 2,
        64: 4,
    }
    num_cu = get_cu_num()
    selected_splitk = 1
    selected_tile_k = 128
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg_dict[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, tg_per_cu in tile_k_tg_dict.items():
        if (hc_hidden_size % tile_k) != 0:
            continue
        meanwhile_tg = num_cu * tg_per_cu
        for splitk in range(1, 17):
            if hc_hidden_size % (splitk * tile_k) != 0 or (hc_hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            if num_tg > meanwhile_tg * 4:
                selected_splitk = splitk
                selected_tile_k = tile_k
                break
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            # print(f"{selected_score=}, {score=} {splitk=} {tile_k=}")
            if selected_score < score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score

    out_pad = torch.empty(
        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32
    )
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32)
    aiter.mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, selected_tile_k)
    out = out.sum(0)
    sqrsum = sqrsum.sum(0)
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

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

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


@benchmark()
def test_mhc_pre(m, hidden_size, hc_mult):
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    residual = torch.randn(m, hc_mult, hidden_size, dtype=dtypes.bf16)
    # residual[:, :2] = 0
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=dtypes.fp32)
    hc_scale = torch.randn((3,), dtype=dtypes.fp32) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=dtypes.fp32) * 0.1
    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 1.0,
        "sinkhorn_repeat": 20,
    }
    # print(f"{fn[:, :128]=}")

    post_mix_ref, comb_mix_ref, layer_input_ref = mhc_pre_ref(
        residual, fn, hc_scale, hc_base, **extra_args
    )
    (post_mix_hip, comb_mix_hip, layer_input_hip), hip_us = run_perftest(
        mhc_pre_hip, residual, fn, hc_scale, hc_base, **extra_args
    )  # , num_iters=2, num_warmup=0)

    checkAllclose(post_mix_ref, post_mix_hip, msg="post_mix")
    hip_err = checkAllclose(comb_mix_ref, comb_mix_hip, msg="comb_mix")
    checkAllclose(layer_input_ref, layer_input_hip, msg="layer_input")
    ret = {}
    ret["hip_err"] = hip_err
    ret["hip_us"] = hip_us

    # try:
    #     (post_mix_tilelang, comb_mix_tilelang, layer_input_tilelang), tilelang_us = run_perftest(mhc_pre_tilelang, residual, fn, hc_scale, hc_base, **extra_args)
    #     checkAllclose(post_mix_tilelang, post_mix_hip, msg="post_mix")
    #     tilelang_err = checkAllclose(comb_mix_tilelang, comb_mix_hip, msg="comb_mix")
    #     checkAllclose(layer_input_tilelang, layer_input_hip, msg="layer_input")
    #     ret["tilelang_err"] = tilelang_err
    #     ret["tilelang_us"] = tilelang_us
    # except Exception as e:
    #     tilelang_err = str(e)
    #     print(f"tilelang error: {tilelang_err}")

    return ret


df = []
for m in [200 * 64, 512, 1024, 2048, 8192][1:2]:
    for hidden_size in [1280, 2560, 4096][:1]:
        for hc_mult in [4]:
            ret = test_mhc_pre(m=m, hidden_size=hidden_size, hc_mult=hc_mult)
            df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("mhc summary (markdown):\n%s", df_md)
