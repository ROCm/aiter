# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch
import functools
from aiter import dtypes
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard


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


@compile_ops("module_mhc")
def mhc_pre_big_fuse_rmsnorm(
    post_mix: Tensor,
    comb_mix: Tensor,
    out: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> None: ...


@functools.lru_cache(maxsize=1024)
def get_mhc_pre_splitk(m: int, hc_hidden_size: int) -> tuple[int, int]:
    prefetch_stages = 2
    tile_m = 16 * 4
    num_cu = get_cu_num()
    arch = get_gfx_runtime()
    tile_k_tg_dict = (
        {
            128: 2 * num_cu,
            64: 4 * num_cu,
        }
        if arch.startswith("gfx9")
        else {
            64: 4 * num_cu,
        }
    )
    selected_splitk = 1
    selected_tile_k = 64
    num_tg_m = (m + tile_m - 1) // tile_m
    selected_score = num_tg_m / (num_cu * tile_k_tg_dict[selected_tile_k])
    selected_score = selected_score / math.ceil(selected_score)
    for tile_k, meanwhile_tg in tile_k_tg_dict.items():
        if (hc_hidden_size % tile_k) != 0:
            continue
        for splitk in range(1, num_cu + 1):
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
            if num_tg > meanwhile_tg * 2:
                break

    return selected_splitk, selected_tile_k


def _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu, prefetch_stages=2):
    return [
        sk
        for sk in range(1, num_cu + 1)
        if hidden_size % (sk * tile_k) == 0
        and (hidden_size // sk) >= tile_k * prefetch_stages
    ]


def _mhc_fused_fill_splitk(m, valid_splitk, num_cu):
    """Pick the split-k whose total grid best fills the device a few waves deep.

    Empirically the optimum sits near total_blocks ~= 32*num_cu, i.e.
    splitk ~= 32*num_cu/m, snapped (geometrically) to the nearest valid divisor.
    """
    ideal = max(1.0, 32.0 * num_cu / m)
    return min(valid_splitk, key=lambda sk: (abs(math.log(sk) - math.log(ideal)), -sk))


def _mhc_fused_config_gfx950_256(m, hidden_size, num_cu):
    tile_k = 64 if m >= 2 * hidden_size else 32
    if hidden_size % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64

    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)
    if not valid:
        return 1, 16, 32, tile_k
    splitk = _mhc_fused_fill_splitk(m, valid, num_cu)

    tile_n = 32  # tile_n=16 never wins on this chip
    if tile_k == 32:
        # large-m underfill: geom fill at split_k 2..4 leaves ~1 wave; a 2nd
        # K-reduction wave measured faster. Excludes geom>=8 (small m) and geom=1.
        if 2 <= splitk <= 4 and (2 * splitk) in valid:
            splitk = 2 * splitk
        tile_m = 32 if (m + 31) // 32 * splitk >= num_cu else 16
    else:  # tile_k == 64: tile_m=32 would overflow LDS, keep 16
        tile_m = 16
        # the compute-bound wide-k path wants >=2 K-reduction waves; fill gives
        # sk=1 at this m but sk=2 is ~2-5% faster (measured m>=8192).
        if splitk < 2 and 2 in valid:
            splitk = 2
    return splitk, tile_m, tile_n, tile_k


def _mhc_fused_config_gfx942_80(m, hidden_size, num_cu):
    tile_k = 32 if (hidden_size <= 4096 and m <= 128) else 64
    if hidden_size % tile_k != 0:
        tile_k = 32 if tile_k == 64 else 64

    valid = _mhc_fused_valid_splitk(hidden_size, tile_k, num_cu)
    if not valid:
        return 1, 16, 32, tile_k

    tile_n = 32  # tile_n=16 never wins on this chip
    tile_m = 16  # tile_m=32 (fn-reuse) never wins on this low-CU part
    if tile_k == 64:
        # deep fill: optimum ~= 12.8*num_cu total blocks; small m saturates the cap.
        m_blocks = (m + 15) // 16
        ideal = max(1.0, 12.8 * num_cu / m_blocks)
        splitk = min(valid, key=lambda sk: (abs(math.log(sk) - math.log(ideal)), -sk))
        if splitk < 2 and 2 in valid:  # large-m underfill; sk>=2 measured faster
            splitk = 2
    else:  # tile_k == 32 small-problem path: shallow ~2-wave fill
        splitk = _mhc_fused_fill_splitk(m, valid, num_cu)
    return splitk, tile_m, tile_n, tile_k


def _mhc_fused_config_default(m, hidden_size, num_cu):
    """Generic fallback for untuned chips: pick (split_k, tile_k) by the occupancy
    scoring search (how many thread-groups fit vs. how many the device can run at
    once), with tile_m fixed at 16 (single mfma band, no fn-reuse path) to avoid the
    tile_m=32 regression seen on low-CU parts that aren't tuned yet. tile_n is then
    chosen to fill the grid."""
    prefetch_stages = 2
    tile_m = 16
    # thread-groups the device can keep in flight per tile_k (smaller tile_k => more)
    tile_k_tg_dict = {
        64: 2 * num_cu,
        32: 4 * num_cu,
    }
    num_tg_m = (m + tile_m - 1) // tile_m

    selected_splitk = 1
    selected_tile_k = 32 if hidden_size % 32 == 0 else 64
    selected_score = -1.0
    for tile_k, meanwhile_tg in tile_k_tg_dict.items():
        if hidden_size % tile_k != 0:
            continue
        for splitk in range(1, num_cu + 1):
            if hidden_size % (splitk * tile_k) != 0 or (hidden_size // splitk) < (
                tile_k * prefetch_stages
            ):
                continue
            num_tg = num_tg_m * splitk
            # occupancy fill ratio, penalize the partial last wave (closer to 1 = better)
            score = num_tg / meanwhile_tg
            score = score / math.ceil(score)
            if score > selected_score:
                selected_splitk = splitk
                selected_tile_k = tile_k
                selected_score = score
            if num_tg > meanwhile_tg * 2:
                break

    m_blocks = (m + tile_m - 1) // tile_m
    tile_n = 16 if num_cu * 2 > m_blocks * selected_splitk else 32
    return selected_splitk, tile_m, tile_n, selected_tile_k


# Per-chip tuned config registry, keyed by (gfx_arch, cu_num).
# Each entry: (m, hidden_size, num_cu) -> (splitk, tile_m, tile_n, tile_k).
_MHC_FUSED_POST_PRE_CONFIG = {
    ("gfx950", 256): _mhc_fused_config_gfx950_256,
    ("gfx942", 80): _mhc_fused_config_gfx942_80,
}


@functools.lru_cache(maxsize=1024)
def get_mhc_fused_post_pre_config(
    m: int, hidden_size: int
) -> tuple[int, int, int, int]:
    """Select (split_k, tile_m, tile_n, tile_k) for the fused post+pre GEMM.

    Looks up a per-chip tuned policy keyed by (gfx_arch, cu_num); falls back to a
    conservative default for untuned chips. K = hidden_size per stream.
    """
    num_cu = get_cu_num()
    try:
        arch = get_gfx_runtime()
    except Exception:
        arch = "unknown"
    policy = _MHC_FUSED_POST_PRE_CONFIG.get((arch, num_cu), _mhc_fused_config_default)
    return policy(m, hidden_size, num_cu)


def mhc_pre_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    device = residual.device
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    return post_mix, comb_mix, layer_input


@torch_compile_guard(gen_fake=mhc_pre_fake)
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,  # if 0, only do pre for hc_head
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    large_m_splitk: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = residual.size(0)
    hc_mult = residual.size(1)
    hidden_size = residual.size(2)
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult or (
        hc_mult3 == hc_mult and sinkhorn_repeat == 0
    )
    hc_hidden_size = hc_mult * hidden_size
    if large_m_splitk:
        selected_splitk, selected_tile_k = get_mhc_pre_splitk_large_m(m, hc_hidden_size)
    else:
        selected_splitk, selected_tile_k = get_mhc_pre_splitk(m, hc_hidden_size)
    device = residual.device
    out_pad = torch.empty(
        selected_splitk, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    out = out_pad[:, :, :hc_mult3]
    sqrsum = torch.empty(selected_splitk, m, dtype=dtypes.fp32, device=device)
    mhc_pre_gemm_sqrsum(out, sqrsum, residual, fn, selected_tile_k)
    # out = out.sum(0)
    # sqrsum = sqrsum.sum(0)

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    if norm_weight is not None:
        mhc_pre_big_fuse_rmsnorm(
            post_mix,
            comb_mix,
            layer_input,
            out,
            sqrsum,
            hc_scale,
            hc_base,
            residual,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            norm_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
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
    store_nt: int = -1,
) -> None: ...


def get_mhc_pre_splitk_large_m(m: int, hc_hidden_size: int) -> tuple[int, int]:
    """Split-K policy for gfx950 large-M post_pre kernel (M > 1024)."""
    if get_gfx_runtime() == "gfx950" and m >= 8192 and hc_hidden_size % (8 * 64) == 0:
        return 8, 64
    return get_mhc_pre_splitk(m, hc_hidden_size)


@compile_ops("module_mhc")
def mhc_fused_post_pre_gemm_sqrsum(
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    next_residual: Tensor,
    layer_input: Tensor,
    residual_in: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    fn: Tensor,
    tile_m: int = 16,  # 16, 32 or 64
    tile_n: int = 32,  # 16 or 32
    tile_k: int = 32,  # 32 or 64
) -> None: ...


def mhc_fused_post_pre_fake(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    force_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = layer_input.size(0)
    hc_mult = residual_in.size(1)
    hidden_size = residual_in.size(2)
    device = layer_input.device
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input_out = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    next_residual = torch.empty_like(residual_in)
    return post_mix, comb_mix, layer_input_out, next_residual


@torch_compile_guard(gen_fake=mhc_fused_post_pre_fake)
def mhc_fused_post_pre_large_m(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """gfx950 large-M post+pre (M > 1024): upstream ``mhc_post`` + ``mhc_pre``."""
    m = residual_in.size(0)

    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.contiguous()
    elif not post_layer_mix.is_contiguous():
        post_layer_mix = post_layer_mix.contiguous()
    if not comb_res_mix.is_contiguous():
        comb_res_mix = comb_res_mix.contiguous()
    if not residual_in.is_contiguous():
        residual_in = residual_in.contiguous()
    if not layer_input.is_contiguous():
        layer_input = layer_input.contiguous()
    if not fn.is_contiguous():
        fn = fn.contiguous()
    if norm_weight is not None and not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    next_residual = torch.empty_like(residual_in)
    post_store_nt = 0 if m > 8 * get_cu_num() else -1
    mhc_post(
        next_residual,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        post_store_nt,
    )
    post_mix, comb_mix, layer_input_out = mhc_pre(
        next_residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
        large_m_splitk=True,
    )
    return post_mix, comb_mix, layer_input_out, next_residual


@torch_compile_guard(gen_fake=mhc_fused_post_pre_fake)
def mhc_fused_post_pre(
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_weight: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-6,
    force_fused: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused mhc_post + next mhc_pre (HIP), mirroring ``mhc_pre`` with post-step inputs.

    Post step (from preceding layer's pre):
        ``layer_input`` (attn/ffn output), ``residual_in``, ``post_layer_mix``, ``comb_res_mix``.

    Pre step (next layer): same ``fn`` / ``hc_scale`` / ``hc_base`` as ``mhc_pre``.

    Returns ``(post_mix, comb_mix, layer_input_out, next_residual)`` -- next pre mixes,
    folded layer input, and the new residual stream for the following layer's post.

    ``force_fused``: when True, always use the fused HIP kernel. When False (default),
    use the fused path only for smaller ``m`` (threshold depends on the detected GPU arch);
    larger ``m`` falls back to the unfused ``mhc_post`` + ``mhc_pre`` path.
    """
    m = layer_input.size(0)
    hc_mult = residual_in.size(1)
    hidden_size = residual_in.size(2)
    arch = get_gfx_runtime()
    fused_m_upper_bound = {
        "gfx950": 1024,
        "gfx942": 128,
    }.get(arch, 1024)

    if not force_fused and m >= fused_m_upper_bound:
        next_residual = torch.empty_like(residual_in)
        mhc_post(
            next_residual,
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
        )
        post_mix, comb_mix, layer_input_out = mhc_pre(
            next_residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )
        return post_mix, comb_mix, layer_input_out, next_residual

    if force_fused and arch == "gfx950" and m > fused_m_upper_bound:
        return mhc_fused_post_pre_large_m(
            layer_input,
            residual_in,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_weight,
            norm_eps,
        )

    assert layer_input.shape == (
        m,
        hidden_size,
    ), f"layer_input shape mismatch: expected ({m}, {hidden_size}), got {tuple(layer_input.shape)}"
    assert residual_in.shape == (m, hc_mult, hidden_size), (
        f"residual_in shape mismatch: expected ({m}, {hc_mult}, {hidden_size}), "
        f"got {tuple(residual_in.shape)}"
    )
    hc_mult3 = fn.size(0)
    assert hc_mult3 == hc_mult * 2 + hc_mult * hc_mult or (
        hc_mult3 == hc_mult and sinkhorn_repeat == 0
    )
    hc_hidden_size = hc_mult * hidden_size
    assert fn.size(1) == hc_hidden_size

    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)
    assert post_layer_mix.shape == (
        m,
        hc_mult,
    ), f"post_layer_mix shape mismatch: expected ({m}, {hc_mult}), got {tuple(post_layer_mix.shape)}"
    assert comb_res_mix.shape == (m, hc_mult, hc_mult), (
        f"comb_res_mix shape mismatch: expected ({m}, {hc_mult}, {hc_mult}), "
        f"got {tuple(comb_res_mix.shape)}"
    )

    selected_splitk, selected_tile_m, selected_tile_n, selected_tile_k = (
        get_mhc_fused_post_pre_config(m, hidden_size)
    )
    n_splits = selected_splitk
    device = layer_input.device

    gemm_out_pad = torch.empty(
        n_splits, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    gemm_out = gemm_out_pad[:, :, :hc_mult3]
    gemm_out_sqrsum = torch.empty(n_splits, m, dtype=dtypes.fp32, device=device)
    next_residual = torch.empty_like(residual_in)

    mhc_fused_post_pre_gemm_sqrsum(
        gemm_out,
        gemm_out_sqrsum,
        next_residual,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        selected_tile_m,
        selected_tile_n,
        selected_tile_k,
    )

    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input_out = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)
    if norm_weight is not None:
        mhc_pre_big_fuse_rmsnorm(
            post_mix,
            comb_mix,
            layer_input_out,
            gemm_out,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            next_residual,
            norm_weight,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            norm_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        mhc_pre_big_fuse(
            post_mix,
            comb_mix,
            layer_input_out,
            gemm_out,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            next_residual,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    return post_mix, comb_mix, layer_input_out, next_residual
