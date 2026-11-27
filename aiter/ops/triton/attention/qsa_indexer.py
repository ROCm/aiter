# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gemma norm/MRoPE and cache preparation for the gfx942 QSA indexer."""

import math
from numbers import Real

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton._triton_kernels.attention.qsa_indexer import (
    _indexer_decode_prep,
    _indexer_k_compress,
    _indexer_q_prep,
    _indexer_ring_compress,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def _config():
    folder = resolve_config_dir("attention", "QSA_INDEXER_PREP", backend="triton")
    return load_config_json(f"{folder}/DEFAULT.json")


def _check(tensor, name, dtype, shape, *, contiguous=True):
    dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    if (
        tensor.dtype not in dtypes
        or tuple(tensor.shape) != tuple(shape)
        or (contiguous and not tensor.is_contiguous())
    ):
        raise ValueError(
            f"{name} must have dtype {dtypes} and shape {shape}, contiguous={contiguous}"
        )


def _same_device(device, tensors):
    if device.type != "cuda" or any(t.device != device for t in tensors):
        raise ValueError("All QSA indexer tensors must be on the same GPU")


def _span(tensor):
    if tensor.numel() == 0:
        return (0, 0)
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return tensor.data_ptr(), tensor.data_ptr() + extent * tensor.element_size()


def _no_overlap(outputs, inputs):
    for i, output in enumerate(outputs):
        lo, hi = _span(output)
        for other in (*outputs[:i], *inputs):
            a, b = _span(other)
            if lo < b and a < hi:
                raise ValueError(
                    "QSA indexer outputs and cache writes must not overlap other inputs"
                )


def _check_shared(
    qk,
    positions,
    state_slots,
    key_state,
    rope_state,
    compressed,
    cos_sin_cache,
    axis_map,
    q_weight,
    k_weight,
    q_eps,
    k_eps,
    q_out,
):
    if get_gfx_runtime() != "gfx942":
        raise ValueError("QSA indexer preparation requires gfx942")
    if qk.ndim != 2 or qk.shape[0] == 0:
        raise ValueError("qk must be nonempty BF16 [rows,640]")
    rows = qk.shape[0]
    _check(qk, "qk", torch.bfloat16, (rows, 640))
    if (
        positions.ndim not in (1, 2)
        or positions.dtype != torch.int64
        or positions.shape[-1] != rows
        or positions.stride(-1) != 1
        or (positions.ndim == 2 and positions.shape[0] != 3)
    ):
        raise ValueError(
            "positions must be int64 [rows] or [3,rows] with unit token stride"
        )
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[0] == 0:
        raise ValueError("cos_sin_cache must be a nonempty [positions,64] table")
    _check(
        cos_sin_cache,
        "cos_sin_cache",
        (torch.bfloat16, torch.float32),
        (cos_sin_cache.shape[0], 64),
    )
    if key_state.ndim != 3 or compressed.ndim != 3:
        raise ValueError("key_state and compressed must be [slots,1,128]")
    if key_state.shape[0] < 8 or compressed.shape[0] == 0:
        raise ValueError(
            "QSA state needs reserved raw slots and a compressed dump slot"
        )
    _check(state_slots, "state_slots", torch.int64, (rows,))
    _check(key_state, "key_state", torch.bfloat16, (key_state.shape[0], 1, 128))
    _check(rope_state, "rope_state", torch.int64, (key_state.shape[0], 3))
    _check(compressed, "compressed", torch.bfloat16, (compressed.shape[0], 1, 128))
    _check(axis_map, "axis_map", torch.int32, (32,))
    _check(q_weight, "q_weight", torch.bfloat16, (128,))
    _check(k_weight, "k_weight", torch.bfloat16, (128,))
    if compressed.numel() * compressed.element_size() >= 2**31:
        raise ValueError("The compressed key pool must be smaller than 2 GiB")
    if any(
        not isinstance(e, Real) or not math.isfinite(e) or e <= 0
        for e in (q_eps, k_eps)
    ):
        raise ValueError("QSA RMSNorm epsilons must be positive finite host scalars")
    inputs = (qk, positions, state_slots, cos_sin_cache, axis_map, q_weight, k_weight)
    outputs = (key_state, rope_state, compressed)
    if q_out is not None:
        _check(q_out, "q_out", torch.bfloat16, (rows, 4, 128))
        outputs += (q_out,)
    _same_device(qk.device, (*inputs, *outputs))
    _no_overlap(outputs, inputs)


def qsa_indexer_prefill_prep(
    qk,
    *,
    positions,
    state_slots,
    key_state,
    rope_state,
    member_rows,
    write_locs,
    rope_matrix,
    compressed,
    cos_sin_cache,
    axis_map,
    q_weight,
    k_weight,
    q_eps=1e-6,
    k_eps=1e-6,
    q_out=None,
    stats_out=None,
):
    """Return index Q and min/max-stat storage; update raw and compressed caches.

    QK is BF16 [M,640]. Each nonzero write_locs entry compresses the four
    consecutive QK rows starting at member_rows. RoPE uses the first member's
    rope_matrix row. The caller must provide distinct live state/write slots.
    The supplied cos/sin table may include YaRN scaling; it is not rebuilt.
    """
    _check_shared(
        qk,
        positions,
        state_slots,
        key_state,
        rope_state,
        compressed,
        cos_sin_cache,
        axis_map,
        q_weight,
        k_weight,
        q_eps,
        k_eps,
        q_out,
    )
    rows, groups = qk.shape[0], write_locs.numel()
    _check(member_rows, "member_rows", torch.int64, (groups,))
    _check(write_locs, "write_locs", torch.int32, (groups,))
    _check(rope_matrix, "rope_matrix", torch.int64, (rows, 3))
    tensors = (member_rows, write_locs, rope_matrix)
    if stats_out is not None:
        _check(stats_out, "stats_out", torch.int32, (rows, 2))
        tensors += (stats_out,)
    _same_device(qk.device, tensors)
    outputs = (key_state, rope_state, compressed) + (() if q_out is None else (q_out,))
    _no_overlap(outputs, (member_rows, write_locs, rope_matrix))
    if stats_out is not None:
        _no_overlap(
            (stats_out,),
            (
                qk,
                positions,
                state_slots,
                *outputs,
                cos_sin_cache,
                axis_map,
                q_weight,
                k_weight,
                member_rows,
                write_locs,
                rope_matrix,
            ),
        )
    q = (
        torch.empty((rows, 4, 128), dtype=torch.bfloat16, device=qk.device)
        if q_out is None
        else q_out
    )
    stats = (
        torch.empty((rows, 2), dtype=torch.int32, device=qk.device)
        if stats_out is None
        else stats_out
    )
    config = _config()
    q_config, k_config = config["q_prep"], config["k_compress"]
    _indexer_q_prep[(-(-rows // q_config["TB"]),)](
        qk,
        q,
        q_weight,
        key_state,
        rope_state,
        state_slots,
        positions,
        positions.stride(0) if positions.ndim == 2 else 0,
        cos_sin_cache,
        axis_map,
        stats,
        rows,
        128,
        q_eps,
        TB=q_config["TB"],
        H=4,
        D=128,
        ROT=64,
        CACHE_STRIDE=cos_sin_cache.stride(0),
        num_warps=q_config["num_warps"],
    )
    if groups:
        _indexer_k_compress[(-(-groups // k_config["GB"]),)](
            qk,
            member_rows,
            write_locs,
            rope_matrix,
            cos_sin_cache,
            axis_map,
            k_weight,
            compressed,
            groups,
            128,
            k_eps,
            GB=k_config["GB"],
            H=4,
            D=128,
            ROT=64,
            RATIO=4,
            CACHE_STRIDE=cos_sin_cache.stride(0),
            num_warps=k_config["num_warps"],
        )
    return q, stats


def qsa_indexer_decode_prep(
    qk,
    *,
    positions,
    state_slots,
    key_state,
    rope_state,
    group_locs,
    write_locs,
    compressed,
    cos_sin_cache,
    axis_map,
    q_weight,
    k_weight,
    q_eps=1e-6,
    k_eps=1e-6,
    verify=False,
    q_out=None,
    stats_out=None,
):
    """Return index Q and update ring/compressed state, including YaRN RoPE.

    group_locs is int32 [rows,4], oldest member first; write_locs is int32
    [rows], with zero as the dump slot. With multiple query rows per request,
    publish every ring write before the separate compression launch.
    Warm this mode before graph capture. All metadata values remain on GPU.
    """
    _check_shared(
        qk,
        positions,
        state_slots,
        key_state,
        rope_state,
        compressed,
        cos_sin_cache,
        axis_map,
        q_weight,
        k_weight,
        q_eps,
        k_eps,
        q_out,
    )
    rows = qk.shape[0]
    _check(group_locs, "group_locs", torch.int32, (rows, 4))
    _check(write_locs, "write_locs", torch.int32, (rows,))
    if type(verify) is not bool:
        raise ValueError("verify must be a host bool")
    tensors = (group_locs, write_locs)
    if stats_out is not None:
        _check(stats_out, "stats_out", torch.int32, (rows, 256, 2))
        tensors += (stats_out,)
    _same_device(qk.device, tensors)
    outputs = (key_state, rope_state, compressed) + (() if q_out is None else (q_out,))
    _no_overlap(outputs, (group_locs, write_locs))
    if stats_out is not None:
        _no_overlap(
            (stats_out,),
            (
                qk,
                positions,
                state_slots,
                *outputs,
                cos_sin_cache,
                axis_map,
                q_weight,
                k_weight,
                group_locs,
                write_locs,
            ),
        )
    q = (
        torch.empty((rows, 4, 128), dtype=torch.bfloat16, device=qk.device)
        if q_out is None
        else q_out
    )
    stats = stats_out if stats_out is not None else write_locs
    config = _config()
    _indexer_decode_prep[(rows,)](
        qk,
        q,
        q_weight,
        k_weight,
        key_state,
        rope_state,
        compressed,
        state_slots,
        group_locs,
        write_locs,
        positions,
        positions.stride(0) if positions.ndim == 2 else 0,
        cos_sin_cache,
        axis_map,
        stats,
        512 if stats_out is not None else 0,
        128,
        q_eps,
        k_eps,
        H=4,
        D=128,
        ROT=64,
        RATIO=4,
        CACHE_STRIDE=cos_sin_cache.stride(0),
        COMPRESS=not verify,
        STAT_SLOTS=256,
        num_warps=config["decode_prep"]["num_warps"],
    )
    if verify:
        _indexer_ring_compress[(rows,)](
            key_state,
            rope_state,
            compressed,
            group_locs,
            write_locs,
            cos_sin_cache,
            axis_map,
            k_weight,
            128,
            k_eps,
            D=128,
            ROT=64,
            RATIO=4,
            CACHE_STRIDE=cos_sin_cache.stride(0),
            num_warps=config["ring_compress"]["num_warps"],
        )
    return q
