# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode QKV chain in one launch: the native-MXFP8 QKV GEMM, the q/k Gemma
RMS norm + partial NeoX RoPE, bf16 q / index-q and the NVFP4 K / V / index-K row
inserts (kernels/mxfp8_qkv_rope_nvfp4_gfx1250.py).

The bytes are those of ``gemm_a8w8_blockscale(isBpreshuffled=True)`` followed by
``fused_qknorm_idxrqknorm`` (NVFP4 row-layout K/V, NVFP4 index rows) on the same
pre-norm rows: the tail reproduces fused_qknorm_idxrqknorm bit for bit, the GEMM
rows differ from the tuned GEMM's only by its FP32 K order (~1e-4 of the bf16
values flip one ulp).

Launch configs come from the tuned table ``AITER_CONFIG_MXFP8_QKV_CHAIN``
(``aiter/configs/mxfp8_qkv_chain_tuned.csv``, plus any
``model_configs/*mxfp8_qkv_chain_tuned*.csv``), keyed by (gfx, cu_num, head shape,
hidden size). A shape's rows give the config up to each ``m``; past its last row
the fused chain is not used (the unfused chain was as fast there when tuned). A
shape without rows takes ``_default_config`` up to ``DEFAULT_MAX_M`` rows.
"""

from __future__ import annotations

import csv
import functools
import struct

import torch

from aiter import logger
from aiter.jit.core import AITER_CONFIGS
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

from .kernels.mxfp8_qkv_rope_nvfp4_gfx1250 import (
    chain_config_error,
    chain_heads,
    chain_shape_error,
)

DEFAULT_MAX_M = 64
_SHAPE_COLS = (
    "num_heads",
    "num_kv_heads",
    "num_index_heads",
    "head_dim",
    "rotary_dim",
    "hidden_size",
)
# tile_m: rows of an m-tile; ns x ks: column groups x K splits of a head (one
# cluster); kw x wm x wn: K x m x n waves of a workgroup; depth: weight K steps in
# flight; astage: activation staging (1 through VGPRs, 2 by TDM, 0 none); wcpol:
# weight cache policy (1 non-temporal); owner_keeps: a rank keeps its own split-K
# rows in LDS.
_CFG_COLS = (
    "tile_m",
    "ns",
    "ks",
    "kw",
    "wm",
    "wn",
    "depth",
    "astage",
    "wcpol",
    "owner_keeps",
)


@functools.lru_cache(maxsize=1)
def _tuned_table() -> dict[tuple, list[tuple[int, tuple]]]:
    """``{(gfx, cu_num, *shape): [(m, config), ...]}``, ascending ``m``."""
    path = AITER_CONFIGS.AITER_CONFIG_MXFP8_QKV_CHAIN_FILE
    table: dict[tuple, list[tuple[int, tuple]]] = {}
    try:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                key = (row["gfx"].strip(), int(row["cu_num"])) + tuple(
                    int(row[c]) for c in _SHAPE_COLS
                )
                cfg = tuple(int(row[c]) for c in _CFG_COLS)
                table.setdefault(key, []).append((int(row["m"]), cfg))
    except FileNotFoundError:
        return {}
    except (KeyError, ValueError) as exc:
        logger.warning(
            f"[mxfp8_qkv_chain] ignoring malformed tuned table {path}: {exc}"
        )
        return {}
    for rows in table.values():
        rows.sort()
    return table


def _default_config(m: int, hidden_size: int):
    """Untuned shapes, up to DEFAULT_MAX_M rows: one m-tile; 4 K splits of the
    largest K-wave count up to 6 that divides the K steps (fewer splits if K needs
    them); 4 column groups a head up to 16 rows, else 2; TDM activation staging
    from 49 rows."""
    if not 0 < m <= DEFAULT_MAX_M:
        return None
    tile_m = (m + 15) // 16 * 16
    ns = 4 if m <= 16 else 2
    for ks in (4, 2, 1):
        for kw in range(6, 0, -1):
            for astage in (2, 1) if tile_m > 48 else (1,):
                cfg = (tile_m, ns, ks, kw, 1, 2, 2, astage, 1, 0)
                if chain_config_error(hidden_size, *cfg[:8]) is None:
                    return cfg
    return None


@functools.lru_cache(maxsize=1024)
def _chain_config(key: tuple, m: int):
    shape, hidden_size = key[2:7], key[7]
    if m <= 0 or chain_shape_error(*shape, hidden_size) is not None:
        return None
    rows = _tuned_table().get(key)
    if rows is None:
        cfg = _default_config(m, hidden_size)
    else:
        cfg = next((c for max_m, c in rows if m <= max_m), None)
    if cfg is None or chain_config_error(hidden_size, *cfg[:8]) is not None:
        return None
    return cfg


def mxfp8_qkv_chain_config(
    m: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    hidden_size: int,
    num_index_heads: int = 1,
):
    """The chain config for ``m`` tokens (the 10 fields of ``_CFG_COLS``) on this
    GPU, or None where the unfused GEMM + fused_qknorm_idxrqknorm keeps the call."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    shape = (num_heads, num_kv_heads, num_index_heads, head_dim, rotary_dim)
    return _chain_config((get_gfx_runtime(), get_cu_num(), *shape, hidden_size), int(m))


def mxfp8_qkv_chain_supported(
    m: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    hidden_size: int,
    num_index_heads: int = 1,
) -> bool:
    """Whether the fused chain takes this call: gfx1250, a head shape the kernel
    accepts (``chain_shape_error``: head_dim 128, rotary_dim a multiple of 8 up to
    head_dim, hidden_size a multiple of 128, at least one q, kv and index-q head)
    and a valid config for ``m`` (a tuned row, or the default up to DEFAULT_MAX_M
    rows for a shape without rows)."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return (
        get_gfx_runtime() == "gfx1250"
        and mxfp8_qkv_chain_config(
            m,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            hidden_size,
            num_index_heads,
        )
        is not None
    )


@functools.cache
def _launcher():
    from .kernels.mxfp8_qkv_rope_nvfp4_gfx1250 import (
        launch_mxfp8_qkv_norm_rope_nvfp4_insert,
    )

    return launch_mxfp8_qkv_norm_rope_nvfp4_insert


@capture_scoped_cache
def _slab_cached(device: torch.device, stream_id: int, numel: int) -> torch.Tensor:
    """FP32 split partial planes, one per (device, stream); a capture gets its own."""
    del stream_id
    with persistent_alloc(device):
        return torch.empty(numel, dtype=torch.float32, device=device)


def _slab(device, numel):
    bucket = 1 << max(20, (numel - 1).bit_length())  # token counts share a bucket
    return _slab_cached(device, torch.cuda.current_stream(device).cuda_stream, bucket)


def mxfp8_qkv_norm_rope_nvfp4_insert(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    index_q_norm_weight: torch.Tensor,
    index_k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    index_slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    index_cache: torch.Tensor,
    index_scale: torch.Tensor,
    q_out: torch.Tensor,
    index_q_out: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    *,
    num_index_heads: int = 1,
    qkv_out: torch.Tensor | None = None,
    config: tuple | None = None,
) -> None:
    """One launch of the QKV chain. H = head_dim (128), K = hidden size,
    N = (num_heads + 2 * num_kv_heads + num_index_heads + 1) * H.

    Args:
        x, x_scale: e4m3 [M, K] (unit column stride, 16-byte row stride) and its
            row-major e8m0 [M, K / 32] (the AR + RMSNorm + MXFP8 producer's output).
        w, w_scale: the (16, 16)-preshuffled e4m3 [N, K] QKV weight
            ([q | k | v | index-q | index-k] heads) and its 32x4-shuffled e8m0
            [N, K / 32] (shuffle_mxfp8_bpreshuffle_scale).
        q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight:
            bf16 [H] Gemma RMSNorm weights.
        cos_sin_cache: bf16 [max_pos, rotary_dim] ([cos | sin]); positions,
            slot_mapping, index_slot_mapping: int64 [M] (slot -1 skips the token).
        k_cache, v_cache: u8 [pages, num_kv_heads, 128, H / 2] NVFP4 payloads;
            k_scale, v_scale: their E4M3 planes [pages, num_kv_heads, 128, H / 16]
            (u8 / e4m3 views, shared strides).
        index_cache, index_scale: u8 [pages, 128, H / 2] and [pages, 128, H / 16].
        q_out: bf16 [M, num_heads * H]; index_q_out: bf16 [M, num_index_heads * H].
        num_heads, num_kv_heads, rotary_dim, eps: as fused_qknorm_idxrqknorm.
        qkv_out: optional bf16 [M, N], the reduced pre-norm rows (for checks).
        config: the 10 fields of ``_CFG_COLS`` instead of the tuned one.
    """
    import flydsl.expr as fx

    from .kernels.tensor_shim import _run_compiled, ptr_arg

    m, kdim = x.shape
    if m == 0:
        return
    nq, nkv, niq, rot = num_heads, num_kv_heads, num_index_heads, rotary_dim
    hd = 2 * k_cache.shape[-1]
    err = chain_shape_error(nq, nkv, niq, hd, rot, kdim)
    if err is not None:
        raise ValueError(f"mxfp8_qkv_norm_rope_nvfp4_insert: {err}")
    n = chain_heads(nq, nkv, niq) * hd
    cfg = tuple(config or mxfp8_qkv_chain_config(m, nq, nkv, hd, rot, kdim, niq) or ())
    if len(cfg) != len(_CFG_COLS) or chain_config_error(kdim, *cfg[:8]) is not None:
        raise ValueError(f"mxfp8_qkv_norm_rope_nvfp4_insert: no config for M={m}")
    if (
        tuple(w.shape) != (n, kdim)
        or x.stride(1) != 1
        or x.stride(0) % 16
        or tuple(x_scale.shape) != (m, kdim // 32)
        or tuple(w_scale.shape) != (n, kdim // 32)
        or not (
            x_scale.is_contiguous() and w_scale.is_contiguous() and w.is_contiguous()
        )
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: operand shapes / strides")
    # 16-bit payload stores: even bases and strides
    even = [k_cache.data_ptr(), v_cache.data_ptr(), index_cache.data_ptr()]
    even += [k_cache.stride(0), index_cache.stride(0)]
    even += [k_cache.stride(1)] if nkv > 1 else []
    if (
        k_cache.shape[1:] != (nkv, 128, hd // 2)
        or k_scale.shape[1:] != (nkv, 128, hd // 16)
        or k_cache.stride() != v_cache.stride()
        or k_scale.stride() != v_scale.stride()
        or k_cache.stride()[2:] != (hd // 2, 1)
        or k_scale.stride()[2:] != (hd // 16, 1)
        or index_cache.shape[1:] != (128, hd // 2)
        or index_scale.shape[1:] != (128, hd // 16)
        or index_cache.stride()[1:] != (hd // 2, 1)
        or index_scale.stride()[1:] != (hd // 16, 1)
        or any(v % 2 for v in even)
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: NVFP4 row-layout caches")
    if (
        q_out.dtype != torch.bfloat16
        or not q_out.is_contiguous()
        or tuple(q_out.shape) != (m, nq * hd)
        or index_q_out.dtype != torch.bfloat16
        or not index_q_out.is_contiguous()
        or tuple(index_q_out.shape) != (m, niq * hd)
        or cos_sin_cache.dtype != torch.bfloat16
        or cos_sin_cache.shape[1] != rot
        or not cos_sin_cache.is_contiguous()
        or eps < 2.0**-126  # v_rsq_f32 without ocml's denormal scaling
    ):
        raise ValueError("mxfp8_qkv_norm_rope_nvfp4_insert: outputs / rope / eps")
    tile_m, ks = cfg[0], cfg[2]
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(x.device, ks * mpad * n)
    write_qkv = qkv_out is not None
    eps_bits = struct.unpack("<i", struct.pack("<f", float(eps)))[0]
    stream = torch.cuda.current_stream(x.device)
    spec = (nq, nkv, niq, hd, rot, kdim) + cfg + (write_qkv,)
    _run_compiled(
        _launcher(),
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(slab),
        ptr_arg(q_out), ptr_arg(index_q_out), ptr_arg(qkv_out if write_qkv else q_out),
        ptr_arg(q_norm_weight), ptr_arg(k_norm_weight), ptr_arg(index_q_norm_weight),
        ptr_arg(index_k_norm_weight), ptr_arg(cos_sin_cache), ptr_arg(positions),
        ptr_arg(slot_mapping), ptr_arg(index_slot_mapping), ptr_arg(k_cache),
        ptr_arg(v_cache), ptr_arg(k_scale), ptr_arg(v_scale), ptr_arg(index_cache),
        ptr_arg(index_scale),
        m, x.stride(0), mpad, cos_sin_cache.shape[0],
        k_cache.stride(0), k_cache.stride(1), k_scale.stride(0), k_scale.stride(1),
        index_cache.stride(0), index_scale.stride(0), eps_bits,
        fx.Stream(stream),
        *spec,
        specialization_key=("mxfp8_qkv_chain", x.device.index) + spec,
    )  # fmt: skip
