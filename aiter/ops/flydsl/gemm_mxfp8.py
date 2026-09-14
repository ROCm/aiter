# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 MXFP8 GEMM, with native (16,16) weight preshuffle.

scale_block=32: A scales [M,K/32], B scales [N,K/32].
scale_block=128: A scales [M,K/128], B scales [ceil(N/128),K/128].
Scales are E8M0 bytes, never numerically converted to uint8. The public API
takes logical scale tensors; the model blockscale adapter handles the legacy
column-major *bytes in a row-major shaped tensor* convention.
"""

import functools
import itertools
import re

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.scaled_gemm_gfx950 import (
    SCALED_GEMM_DTYPE_BF16,
    SCALED_GEMM_DTYPE_FP32,
    SCALED_GEMM_DTYPE_MXFP8,
    make_scaled_gemm_param_and_validate,
    scaled_gemm,
)

DEFAULT_CONFIG = {
    "block_m": 32,
    "block_n": 64,
    "block_k": 128,
    "stages": 2,
    "split_k": 1,
    "m_waves": 1,
    "n_waves": 2,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
    "mma_m": 16,
    "mma_n": 16,
    "mma_k": 128,
}
CONFIG_KEYS = tuple(DEFAULT_CONFIG)
_NAME_RE = re.compile(
    r"^flydsl_mxfp8_(?P<out_dtype>bf16|fp32)_sb(?P<scale_block>32|128)"
    r"_bp(?P<bpreshuffle>[01])_sat(?P<scale_a_transposed>[01])"
    r"_t(?P<block_m>\d+)x(?P<block_n>\d+)x(?P<block_k>\d+)x(?P<stages>\d+)"
    r"_ks(?P<split_mode>1|d)_w(?P<m_waves>\d+)x(?P<n_waves>\d+)x(?P<k_waves>\d+)"
    r"_mma(?P<mma_m>\d+)x(?P<mma_n>\d+)x(?P<mma_k>\d+)"
    r"_bias(?P<has_bias>[01])_gm(?P<group_m>\d+)_p(?P<policy>ft|hti)"
    r"_(?P<target_gfx>gfx950)$"
)


def flydsl_mxfp8_kernel_name(
    config,
    *,
    out_dtype=torch.bfloat16,
    has_bias=False,
    scale_block=32,
    bpreshuffle=False,
    scale_a_transposed=False,
):
    c = {**DEFAULT_CONFIG, **config}
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("MXFP8 output must be bf16 or fp32")
    if scale_block not in (32, 128):
        raise ValueError("scale_block must be 32 or 128")
    dt = "bf16" if out_dtype == torch.bfloat16 else "fp32"
    return (
        f"flydsl_mxfp8_{dt}_sb{scale_block}_bp{int(bpreshuffle)}"
        f"_sat{int(scale_a_transposed)}"
        f"_t{c['block_m']}x{c['block_n']}x{c['block_k']}x{c['stages']}"
        f"_ks{'d' if c['split_k'] > 1 else 1}_w{c['m_waves']}x{c['n_waves']}x{c['k_waves']}"
        f"_mma{c['mma_m']}x{c['mma_n']}x{c['mma_k']}"
        f"_bias{int(has_bias)}_gm{c['group_m']}"
        f"_p{'hti' if c['use_half_tile_interleaved'] else 'ft'}_gfx950"
    )


def get_flydsl_mxfp8_kernel_params(name, split_k=1):
    match = _NAME_RE.fullmatch(str(name))
    if match is None:
        return None
    p = match.groupdict()
    try:
        value = int(split_k)
    except (TypeError, ValueError, OverflowError):
        return None
    if str(split_k) not in (str(value), f"{value}.0") or value < 1:
        return None
    if (p.pop("split_mode") == "d") != (value > 1):
        return None
    p["split_k"] = value
    for key in CONFIG_KEYS:
        if key != "use_half_tile_interleaved":
            p[key] = int(p[key])
            if key != "group_m" and p[key] <= 0:
                return None
    p["use_half_tile_interleaved"] = p.pop("policy") == "hti"
    p["scale_block"] = int(p["scale_block"])
    for key in ("has_bias", "bpreshuffle", "scale_a_transposed"):
        p[key] = p[key] == "1"
    return p


def mxfp8_kernel_config(
    config, out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
):
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("MXFP8 output must be bf16 or fp32")
    return {
        **DEFAULT_CONFIG,
        **config,
        "in_dtype_id": SCALED_GEMM_DTYPE_MXFP8,
        "out_dtype_id": (
            SCALED_GEMM_DTYPE_BF16
            if out_dtype == torch.bfloat16
            else SCALED_GEMM_DTYPE_FP32
        ),
        "has_bias": has_bias,
        "scale_block": scale_block,
        "bpreshuffle": bpreshuffle,
        "scale_a_transposed": scale_a_transposed,
        "a_is_transposed": False,
        "b_is_transposed": True,
    }


def flydsl_mxfp8_gemm(
    a,
    b,
    scale_a,
    scale_b,
    out=None,
    bias=None,
    *,
    config=None,
    out_dtype=None,
    bpreshuffle=False,
    scale_block=32,
    scale_a_transposed=False,
    stream=None,
):
    """C = dequant(A[M,K]) @ dequant(B[N,K]).T + bias, on gfx950.

    B may be ``shuffle_weight(B, layout=(16,16))``; scales remain unshuffled.
    ``scale_a_transposed`` describes logical column-major A scales, not packed
    bytes. Call the blockscale model entry for the latter convention.
    """
    if get_gfx() != "gfx950":
        raise RuntimeError("FlyDSL MXFP8 GEMM requires gfx950")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("expected A[M,K], B[N,K]")
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        raise ValueError("MXFP8 data must be torch.float8_e4m3fn")
    if scale_a.dtype not in (
        torch.uint8,
        torch.float8_e8m0fnu,
    ) or scale_b.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("MXFP8 scales must be E8M0 or raw uint8 bytes")
    if scale_block not in (32, 128):
        raise ValueError("scale_block must be 32 or 128")
    if scale_a_transposed and scale_block == 32:
        raise ValueError("1x32 scales must be row-major")
    if bpreshuffle and (not b.is_contiguous() or b.shape[0] % 16):
        raise ValueError("preshuffled B must be contiguous with N divisible by 16")
    if out is not None and (
        out.shape != (a.shape[0], b.shape[0]) or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous [M,N]")
    launch_stream = torch.cuda.current_stream(a.device) if stream is None else stream
    if launch_stream.device != a.device:
        raise ValueError("stream must be on the input device")
    dtype = out_dtype or (out.dtype if out is not None else torch.bfloat16)
    c = mxfp8_kernel_config(
        config or {},
        dtype,
        bias is not None,
        scale_block,
        bpreshuffle,
        scale_a_transposed,
    )
    # Copies, allocations and split-K zero initialization must share the launch
    # stream (including first use on a non-default stream).
    with torch.cuda.stream(launch_stream):
        return scaled_gemm(
            a.contiguous(),
            b.contiguous().t(),
            scale_a,
            scale_b,
            out=out,
            bias=bias,
            user_kwargs=c,
            out_dtype=dtype,
            stream=launch_stream,
        )


@functools.lru_cache(maxsize=16)
def _mxfp8_base_configs(
    out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
):
    """Shape-independent legality pass over the shared HGEMM search axes."""
    from .gemm_a16w16_policy import gemm_config_space
    from .kernels.scaled_gemm_gfx950 import make_scaled_gemm_gfx950_param

    selections = gemm_config_space(1, block_k=(128, 256, 512), k_waves=(1, 2, 4))
    # Only the presence of split-K is constexpr. Expand its actual count later.
    selections["split_k"] = (1, 2)
    valid = []
    for combo in itertools.product(*selections.values()):
        c = dict(zip(selections, combo))
        bm, bn, bk = c["block_m"], c["block_n"], c["block_k"]
        mw, nw, kw = c["m_waves"], c["n_waves"], c["k_waves"]
        if bm % (mw * 16) or bn % (nw * 16) or bk % (kw * 128):
            continue
        if c["use_half_tile_interleaved"]:
            if c["stages"] != 2 or mw != 2 or nw < 2 or kw != 1:
                continue
        elif bm // mw // 16 > 4 or bn // nw // 16 > 4:
            continue
        c = dict(DEFAULT_CONFIG, **c)
        try:
            make_scaled_gemm_gfx950_param(
                **mxfp8_kernel_config(
                    c, out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
                )
            )
        except (ValueError, AssertionError):
            continue
        valid.append(c)
    return tuple(valid)


@functools.lru_cache(maxsize=256)
def get_flydsl_mxfp8_configs(
    m,
    n,
    k,
    out_dtype=torch.bfloat16,
    has_bias=False,
    scale_block=32,
    bpreshuffle=False,
    scale_a_transposed=False,
):
    """HGEMM search space adapted to scaled MFMA: BK=128/256/512, slice=1/2/4."""
    from .gemm_a16w16_policy import GemmConfigPruner, gemm_config_space

    if get_gfx() != "gfx950" or min(m, n, k) <= 0:
        return ()
    split_candidates = gemm_config_space(k)["split_k"]
    configs = []
    for base in _mxfp8_base_configs(
        out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
    ):
        for split_k in ((1,) if base["split_k"] == 1 else split_candidates[1:]):
            c = dict(base, split_k=split_k)
            if (
                make_scaled_gemm_param_and_validate(
                    m,
                    n,
                    k,
                    mxfp8_kernel_config(
                        c,
                        out_dtype,
                        has_bias,
                        scale_block,
                        bpreshuffle,
                        scale_a_transposed,
                    ),
                )
                is not None
            ):
                configs.append(c)
    # Preserve all legal slice-K choices: the BF16 occupancy estimate does not
    # model FP8 scaled-MFMA throughput or our FP32 slice reduction storage.
    pruner = GemmConfigPruner(
        m,
        n,
        k,
        torch.cuda.get_device_properties(torch.cuda.current_device()),
        1,
        prune_slice_k=False,
    )
    configs = pruner.prune(configs)
    # K=64 uses the alternate MFMA atom; it is not part of the peak-MFMA sweep.
    if scale_block == 32 and k % 128:
        c = dict(
            DEFAULT_CONFIG,
            block_m=32,
            block_n=32,
            block_k=64,
            n_waves=1,
            mma_m=32,
            mma_n=32,
            mma_k=64,
        )
        if (
            make_scaled_gemm_param_and_validate(
                m,
                n,
                k,
                mxfp8_kernel_config(
                    c, out_dtype, has_bias, scale_block, bpreshuffle, scale_a_transposed
                ),
            )
            is not None
        ):
            configs.append(c)
    return tuple(configs)
