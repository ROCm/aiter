# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention APIs."""

from __future__ import annotations


import os
import json
import torch


from pathlib import Path
from flydsl.runtime.device import get_rocm_arch
from .kernels.gdr_decode import create_shuffle_gdr_decode_kernel
from .kernels.tensor_shim import get_dtype_str, _run_compiled

__all__ = [
    "flydsl_gdr_decode",
]


GDR_GLOBAL_CONFIG_MAP = None
GDR_GPU_ARCH = get_rocm_arch()


def get_default_kwargs(
    batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim
):
    d = {}
    d["NUM_BLOCKS_PER_V_DIM"] = 1
    d["NUM_WARPS"] = 4
    d["WARP_THREADS_K"] = 8
    global GDR_GLOBAL_CONFIG_MAP
    global GDR_GPU_ARCH
    if GDR_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        fname = os.path.join(Path(__file__).resolve().parent, "gdr_decode_tuned.jsonl")
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 10:
                    obj = json.loads(line)
                    arch, b, sq, nkh, nvh, khd, vhd = (
                        obj["arch"],
                        obj["b"],
                        obj["sq"],
                        obj["num_k_heads"],
                        obj["num_v_heads"],
                        obj["head_k_dim"],
                        obj["head_v_dim"],
                    )
                    _dict[(arch, b, sq, nkh, nvh, khd, vhd)] = obj["config"]
        GDR_GLOBAL_CONFIG_MAP = _dict
    config = GDR_GLOBAL_CONFIG_MAP.get(
        (
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
    stream: torch.cuda.Stream = torch.cuda.current_stream(),
):
    device = query.device
    dtype = query.dtype
    for input in [key, value, a, b, dt_bias, A_log, indices, out]:
        assert input.is_contiguous()
        assert input.data_ptr() % 16 == 0
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
        batch_size, seq_length, num_k_heads, num_v_heads, head_k_dim, head_v_dim
    )
    exe = create_shuffle_gdr_decode_kernel(
        get_dtype_str(query.dtype),
        get_dtype_str(A_log.dtype),
        get_dtype_str(state.dtype),
        seq_length,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        state.stride(),
        use_qk_l2norm,
        **kwargs_,
    )
    with torch.cuda.device(query.device.index):
        _run_compiled(
            exe,
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            indices,
            state_,
            out,
            batch_size,
            stream,
        )
    if need_shuffle_state:
        state_ = state_.permute(0, 1, 3, 2).contiguous()
        state.copy_(state_)
