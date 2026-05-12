# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations


import os
import csv
import torch


from pathlib import Path
from flydsl.runtime.device import get_rocm_arch
from .kernels.gdr_decode import create_vk_gdr_decode_kernel
from .kernels.tensor_shim import get_dtype_str, _run_compiled

__all__ = [
    "flydsl_gdr_decode",
]


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()


def get_default_kwargs(
    dtype_str,
    state_dtype_str,
    batch_size,
    seq_length,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    d = {}
    d["NUM_BLOCKS_PER_V_DIM"] = 1
    d["NUM_WARPS"] = 4
    d["WARP_THREADS_K"] = 8
    global GDR_GLOBAL_CONFIG_MAP
    global GDR_GPU_ARCH
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        fname = os.path.join(Path(__file__).resolve().parent, "gdr_decode_tuned.csv")
        with open(fname, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                obj = dict(row)
                arch, b, sq, nkh, nvh, khd, vhd = (
                    obj["arch"],
                    int(obj["b"]),
                    int(obj["sq"]),
                    int(obj["num_k_heads"]),
                    int(obj["num_v_heads"]),
                    int(obj["head_k_dim"]),
                    int(obj["head_v_dim"]),
                )
                d_str, sd_str = obj["dtype"], obj["state_dtype"]
                if float(obj["duration"]) < 10000.0:
                    _dict[(d_str, sd_str, arch, b, sq, nkh, nvh, khd, vhd)] = {
                        "NUM_BLOCKS_PER_V_DIM": int(obj["NUM_BLOCKS_PER_V_DIM"]),
                        "NUM_WARPS": int(obj["NUM_WARPS"]),
                        "WARP_THREADS_K": int(obj["WARP_THREADS_K"]),
                    }
        GDR_GLOBAL_CONFIG_MAP = _dict
    config = GDR_GLOBAL_CONFIG_MAP.get(
        (
            dtype_str,
            state_dtype_str,
            GDR_GPU_ARCH,
            batch_size,
            seq_length,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
        ),
        None,
    )
    if config:
        d.update(config)
    return d


def flydsl_gdr_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream = None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    device = query.device
    dtype = query.dtype
    for input in [query, key, value, a, b, dt_bias, A_log, indices, out]:
        assert input.device == device
    assert state.data_ptr() % 16 == 0
    for input in [key, value, a, b, dt_bias, out]:
        assert input.dtype == dtype
    assert state.dtype in [torch.float, torch.bfloat16]
    assert A_log.dtype in [torch.float, torch.bfloat16]
    assert indices.dtype == torch.int32

    if need_shuffle_state:
        state_ = state.permute(0, 1, 3, 2).contiguous()
    else:
        state_ = state
    batch_size, seq_length, num_k_heads, head_k_dim = query.shape
    num_v_heads = value.shape[-2]
    head_v_dim = value.shape[-1]
    kwargs_ = get_default_kwargs(
        str(dtype),
        str(state_.dtype),
        batch_size,
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    exe = create_vk_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state_.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state_.stride(),
        use_qk_l2norm,
        **kwargs_,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            a.contiguous(),
            b.contiguous(),
            dt_bias.contiguous(),
            A_log.contiguous(),
            indices.contiguous(),
            state_,
            out,
            batch_size,
            stream,
        )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)
