# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import math
from torch import Tensor
from aiter import dtypes
from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num


@compile_ops("module_mhc")
def mhc_pre_gemm_sqrsum(
    out: Tensor,
    sqrsum: Tensor,
    x: Tensor,
    fn: Tensor,
    tile_k: int = 128,  # 64 or 128
) -> None: ...


@compile_ops("module_mhc")
def mhc_pre_big_fuse(
    post_mix: Tensor,
    comb_mix: Tensor,
    layer_input: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> None: ...


def mhc_pre(
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
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    prefetch_stages = 2
    tile_m = 16 * 4
    tile_k_tg_dict = {
        128: 2,
        64: 4,
    }
    num_cu = get_cu_num()
    selected_splitk = 1
    selected_tile_k = 64
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg_dict[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, tg_per_cu in tile_k_tg_dict.items():
        if (hc_hidden_size % tile_k) != 0:
            continue
        meanwhile_tg = num_cu * tg_per_cu
        for splitk in range(1, 33):
            if hc_hidden_size % (splitk * tile_k) != 0 or (hc_hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            if selected_score < score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            # print(f"{selected_score=} {selected_splitk=} {selected_tile_k=} {score=} {splitk=} {tile_k=}")
            if num_tg > meanwhile_tg * 4:
                break

    out_pad = torch.empty(
        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32
    )
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32)
    mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, selected_tile_k)
    # out = out.sum(0)
    # sqrsum = sqrsum.sum(0)

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16)
    mhc_pre_big_fuse(
        post_mix,
        comb_mix,
        layer_input,
        out,
        sqrsum,
        hc_scale,
        hc_base,
        residual,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
    )

    return post_mix, comb_mix, layer_input


@compile_ops("module_mhc")
def mhc_post(
    out: Tensor,
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> None: ...
