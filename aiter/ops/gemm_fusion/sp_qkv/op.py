# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER-owned `sp_qkv_fused` v1 contract, route and reference native.

The tuned implementation lives in an out-of-tree AITER kernel provider (the
plugin_poc `sp-qkv-provider`).  The native body below is a deliberately slow,
pure-PyTorch reference of the same v1 semantics: it is the fallback when no
provider serves the call, and it is what keeps the op usable on stock torch
without the provider installed.  It is **not** a performance path.

The reference is a single rank's view of the fused exchange: it computes the
local GEMM and emits all four source token blocks with the local rank's head
shard (q heads rank*16..rank*16+15, k/v/iq head rank, replicated ik).  With
byte-identical inputs across ranks the blocks are value-identical.
Multi-rank/capture behaviour belongs to the provider.
"""

from __future__ import annotations

from typing import Any

from aiter.plugin import route

__all__ = ["sp_qkv_fused"]

MODES = ("dense", "indexer", "reuse")
_Q_HEADS = 16
_KV_HEADS = 1
_HEAD_DIM = 128
_HIDDEN_K = 6144
_N = {"dense": 9216, "indexer": 9856, "reuse": 9856}
_PAGE = {"dense": 128, "indexer": 16, "reuse": 16}
_Q_OFF, _K_OFF, _V_OFF, _IQ_OFF, _IK_OFF = 0, 8192, 8704, 9216, 9728


class _TorchError(ValueError):
    """A tensor-contract violation in the native reference."""


def _torch() -> Any:
    import torch

    return torch


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise _TorchError(message)


def _tensor(name: str, value: Any, torch: Any) -> Any:
    _need(isinstance(value, torch.Tensor), f"{name} must be a torch.Tensor")
    return value


def _shape(name: str, value: Any, expected: tuple[int, ...], torch: Any) -> Any:
    _tensor(name, value, torch)
    _need(
        tuple(value.shape) == expected,
        f"{name} must have shape {expected}, got {tuple(value.shape)}",
    )
    return value


def _dtype(name: str, value: Any, dtype: Any, torch: Any) -> Any:
    _tensor(name, value, torch)
    _need(value.dtype == dtype, f"{name} must be {dtype}, got {value.dtype}")
    return value


def _validate_mode_tensors(
    mode: str,
    *,
    k: Any,
    v: Any,
    index_q: Any,
    index_q_norm_w: Any,
    index_k_norm_w: Any,
    index_cache: Any,
) -> None:
    if mode == "indexer":
        missing = [
            name
            for name, value in (
                ("index_q", index_q),
                ("index_q_norm_w", index_q_norm_w),
                ("index_k_norm_w", index_k_norm_w),
                ("index_cache", index_cache),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"mode='indexer' requires {', '.join(missing)} (the indexer tensors)"
            )
        return
    missing = [name for name, value in (("k", k), ("v", v)) if value is None]
    if missing:
        raise ValueError(
            f"mode={mode!r} requires {', '.join(missing)} (the dense/reuse KV outputs)"
        )


def _preshuffle_to_linear(w: Any, torch: Any) -> Any:
    """Undo ``shuffle_weight(layout=(16,16))`` on an [N, K] fp8 weight."""

    rows, k = int(w.shape[0]), int(w.shape[1])
    _need(rows % 16 == 0 and k >= 32 and k % 32 == 0, "w must be 16x32-block aligned")
    # Physical order after shuffle_weight: (n0, k0, k1, n1, k2).
    physical = w.float().view(rows // 16, k // 32, 2, 16, 16)
    return physical.permute(0, 3, 1, 2, 4).reshape(rows, k)


def _local_rank(group: Any) -> int:
    """The rank whose head shard this reference process models."""

    try:
        import torch.distributed as dist
    except ImportError:  # single-process reference without torch.distributed
        return 0
    if group is None:
        try:
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:  # noqa: BLE001, S110
            pass
        return 0
    rank_fn = getattr(group, "rank", None)
    if callable(rank_fn):
        try:
            return int(rank_fn())
        except Exception:  # noqa: BLE001, S110
            pass
    try:
        return int(dist.get_rank(group))
    except Exception:  # noqa: BLE001
        return 0


def _rms_rope(
    values: Any,
    weight: Any,
    cos_cache: Any,
    sin_cache: Any,
    positions: Any,
    torch: Any,
    eps: float,
) -> Any:
    """Gemma RMSNorm (``1 + weight``) followed by partial NeoX RoPE (dims 0..63)."""

    var = values.float().pow(2).mean(dim=-1, keepdim=True)
    nrm = values.float() * torch.rsqrt(var + float(eps)) * (1.0 + weight.float())
    out = nrm.clone()
    if nrm.shape[0]:
        idx = torch.arange(32, device=nrm.device)
        pos = positions.long()
        cos = cos_cache[pos[:, None], idx[None, :]].float()
        sin = sin_cache[pos[:, None], idx[None, :]].float()
        first, second = nrm[:, :32], nrm[:, 32:64]
        out[:, :32] = first * cos - second * sin
        out[:, 32:64] = second * cos + first * sin
    return out


def _qv_rows(
    gemm: Any,
    mode: str,
    m: int,
    torch: Any,
    rank: int = 0,
) -> tuple[Any, Any, Any, Any, Any]:
    """The local rank's view: every source token block carries this rank's shard.

    The reference cannot perform the SP exchange.  A single rank sees T rows made
    of four source token blocks; every block carries the destination (local)
    rank's q/k/v/iq head shard.  With identical inputs across ranks this makes
    every block value-identical, which is the property the GPU e2e compares.
    """

    if int(rank) not in range(4):
        raise ValueError(f"rank must be in 0..3, got {rank!r}")
    t = 4 * m
    rows = torch.arange(t, device=gemm.device)
    local = rows % m
    q_all = gemm[:, _Q_OFF : _Q_OFF + 64 * 128].view(m, 64, 128)[local]
    q = q_all[:, int(rank) * _Q_HEADS : (int(rank) + 1) * _Q_HEADS, :].reshape(
        t * 16, 128
    )
    k_all = gemm[:, _K_OFF : _K_OFF + 4 * 128].view(m, 4, 128)[local]
    k = k_all[:, int(rank), :].reshape(t, 1, 128)
    v_all = gemm[:, _V_OFF : _V_OFF + 4 * 128].view(m, 4, 128)[local]
    v = v_all[:, int(rank), :].reshape(t, 1, 128)
    iq = ik = None
    if mode in ("indexer", "reuse"):
        iq_all = gemm[:, _IQ_OFF : _IQ_OFF + 4 * 128].view(m, 4, 128)[local]
        iq = iq_all[:, int(rank), :].reshape(t, 1, 128)
        ik = gemm[:, _IK_OFF : _IK_OFF + 128].view(m, 1, 128)[local, 0]
    return q, k, v, iq, ik


def _write_kv(
    *,
    slot: int,
    mode: str,
    k_row: Any,
    v_row: Any,
    k_cache: Any,
    v_cache: Any,
    k_scale: Any,
    v_scale: Any,
    torch: Any,
) -> None:
    page = _PAGE[mode]
    blk, sib = divmod(slot, page)
    _need(k_cache.ndim == 5 and tuple(k_cache.shape[2::2]) == (8, 16), "k_cache layout")
    _need(
        v_cache.ndim == 5 and v_cache.shape[3] == 128 and v_cache.shape[4] == 16,
        "v_cache layout",
    )
    _need(
        k_cache.is_contiguous() and v_cache.is_contiguous(), "caches must be contiguous"
    )
    _need(
        k_scale.is_contiguous() and v_scale.is_contiguous(), "scales must be contiguous"
    )

    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    k_fp32 = k_row.float()
    amax = float(k_fp32.abs().max())
    ks = (amax / fp8_max) if amax else 1.0
    v_fp32 = v_row.float()
    amax = float(v_fp32.abs().max())
    vs = (amax / fp8_max) if amax else 1.0

    k_flat = k_cache.reshape(-1)
    k_sb, _, k_sdx, k_ss, k_sx = k_cache.stride()
    d = torch.arange(128, device=k_flat.device)
    base = blk * k_sb + sib * k_ss
    k_idx = base + (d // 16) * k_sdx + (d % 16) * k_sx
    _need(int(k_idx.max()) < k_flat.numel(), "k_cache slot out of range")
    k_flat[k_idx] = (k_fp32 / ks).to(torch.float8_e4m3fn)

    v_flat = v_cache.reshape(-1)
    v_sb, _, v_ssc, v_sd, v_sx = v_cache.stride()
    base = blk * v_sb + (sib // 16) * v_ssc + (sib % 16) * v_sx
    v_idx = base + d * v_sd
    _need(int(v_idx.max()) < v_flat.numel(), "v_cache slot out of range")
    v_flat[v_idx] = (v_fp32 / vs).to(torch.float8_e4m3fn)

    k_scale.reshape(-1)[slot] = ks
    v_scale.reshape(-1)[slot] = vs


def _native(
    q,
    x,
    x_scale,
    w,
    w_scale,
    q_norm_w,
    k_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    *,
    mode,
    k,
    v,
    index_q,
    index_q_norm_w,
    index_k_norm_w,
    index_cache,
    eps,
    group,
    stream,
) -> None:
    torch = _torch()
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    _validate_mode_tensors(
        mode,
        k=k,
        v=v,
        index_q=index_q,
        index_q_norm_w=index_q_norm_w,
        index_k_norm_w=index_k_norm_w,
        index_cache=index_cache,
    )
    _need(x.dim() == 2, "x must be 2-D [M, 6144]")
    _need(x.shape[1] == _HIDDEN_K, f"x K must be {_HIDDEN_K}, got {x.shape[1]}")
    _dtype("x", x, torch.float8_e4m3fn, torch)
    _need(x_scale.numel() >= int(x.shape[0]), "x_scale has fewer than M elements")
    _dtype("x_scale", x_scale, torch.float32, torch)
    _need(x_scale.is_contiguous(), "x_scale must be contiguous")
    m = int(x.shape[0])
    t = 4 * m
    _need(m >= 1, "x must have at least one row")
    n = _N[mode]
    _need(w.dim() == 2 and w.shape[1] == _HIDDEN_K, "w must be [rows >= N, 6144]")
    _need(w.shape[0] >= n, f"w has fewer than N={n} rows")
    _dtype("w", w, torch.float8_e4m3fn, torch)
    _need(w.is_contiguous(), "w must be contiguous")
    _need(w_scale.numel() >= n, f"w_scale has fewer than N={n} elements")
    _dtype("w_scale", w_scale, torch.float32, torch)
    _need(w_scale.is_contiguous(), "w_scale must be contiguous")
    _shape("q", q, (t, 16, 128), torch)
    _dtype("q", q, torch.bfloat16, torch)
    for name, value in (("q_norm_w", q_norm_w), ("k_norm_w", k_norm_w)):
        _shape(name, value, (128,), torch)
        _dtype(name, value, torch.float32, torch)
    for name, value in (("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        _tensor(name, value, torch)
        _dtype(name, value, torch.bfloat16, torch)
        _need(value.dim() == 2 and value.shape[1] >= 32, f"{name} must be [>=1, >=32]")
    for name, value in (("positions", positions), ("slot_mapping", slot_mapping)):
        _tensor(name, value, torch)
        _dtype(name, value, torch.int64, torch)
        _need(value.numel() >= t, f"{name} has fewer than T={t} entries")
    for name, value in (
        ("k_cache", k_cache),
        ("v_cache", v_cache),
        ("k_scale", k_scale),
        ("v_scale", v_scale),
    ):
        _tensor(name, value, torch)
    _dtype("k_cache", k_cache, torch.float8_e4m3fn, torch)
    _dtype("v_cache", v_cache, torch.float8_e4m3fn, torch)
    _dtype("k_scale", k_scale, torch.float32, torch)
    _dtype("v_scale", v_scale, torch.float32, torch)
    if mode == "indexer":
        _shape("index_q", index_q, (t, 1, 128), torch)
        _dtype("index_q", index_q, torch.bfloat16, torch)
        _shape("index_q_norm_w", index_q_norm_w, (128,), torch)
        _shape("index_k_norm_w", index_k_norm_w, (128,), torch)
        _dtype("index_q_norm_w", index_q_norm_w, torch.float32, torch)
        _dtype("index_k_norm_w", index_k_norm_w, torch.float32, torch)
        _tensor("index_cache", index_cache, torch)
        _dtype("index_cache", index_cache, torch.float8_e4m3fn, torch)
        _need(index_cache.is_contiguous(), "index_cache must be contiguous")
    else:
        _shape("k", k, (t, 1, 128), torch)
        _shape("v", v, (t, 1, 128), torch)
        _dtype("k", k, torch.bfloat16, torch)
        _dtype("v", v, torch.bfloat16, torch)
    if group is not None:
        size = getattr(group, "size", None)
        if callable(size) and int(size()) != 4:
            raise NotImplementedError(
                f"native sp_qkv_fused reference supports SP world 4, got {int(size())}"
            )

    x_deq = x.float() * x_scale.float().reshape(-1)[:m, None]
    w_lin = _preshuffle_to_linear(w[:n], torch)
    gemm = (x_deq @ w_lin.T) * w_scale.float().reshape(-1)[:n][None, :]
    gemm = gemm.to(torch.bfloat16).float()

    local_rank = _local_rank(group)
    q_rows, k_rows, v_rows, iq_rows, ik_rows = _qv_rows(
        gemm, mode, m, torch, rank=local_rank
    )
    pos_q = positions[:t].repeat_interleave(16)
    q_out = _rms_rope(q_rows, q_norm_w, cos_cache, sin_cache, pos_q, torch, float(eps))
    q.copy_(q_out.view(t, 16, 128).to(torch.bfloat16))
    k_out = _rms_rope(
        k_rows[:, 0], k_norm_w, cos_cache, sin_cache, positions[:t], torch, float(eps)
    )
    v_out = v_rows[:, 0].float()
    if mode != "indexer":
        k.copy_(k_out[:, None, :].to(torch.bfloat16))
        v.copy_(v_out[:, None, :].to(torch.bfloat16))
    if mode == "indexer":
        iq_out = _rms_rope(
            iq_rows[:, 0],
            index_q_norm_w,
            cos_cache,
            sin_cache,
            positions[:t],
            torch,
            float(eps),
        )
        ik_out = _rms_rope(
            ik_rows,
            index_k_norm_w,
            cos_cache,
            sin_cache,
            positions[:t],
            torch,
            float(eps),
        )
        index_q.copy_(iq_out[:, None, :].to(torch.bfloat16))
        flat_idx = index_cache.reshape(-1)
        idx_rows = torch.arange(128, device=flat_idx.device)

    for row in range(t):
        slot = int(slot_mapping[row])
        if slot < 0:
            continue
        _write_kv(
            slot=slot,
            mode=mode,
            k_row=k_out[row],
            v_row=v_out[row],
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            torch=torch,
        )
        if mode == "indexer":
            start = slot * 128
            _need(start + 128 <= flat_idx.numel(), "index_cache slot out of range")
            flat_idx[start + idx_rows] = (
                ik_out[row].to(torch.bfloat16).to(torch.float8_e4m3fn)
            )


@route(op="sp_qkv_fused", version=1)
def sp_qkv_fused(
    q,
    x,
    x_scale,
    w,
    w_scale,
    q_norm_w,
    k_norm_w,
    cos_cache,
    sin_cache,
    positions,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    *,
    mode,
    k=None,
    v=None,
    index_q=None,
    index_q_norm_w=None,
    index_k_norm_w=None,
    index_cache=None,
    eps=1e-6,
    group=None,
    stream=None,
) -> None:
    """One MiniMax-M3 SP4 QKV step (contract v1), with a slow native reference.

    The routed wrapper asks enabled providers first.  If none serves the call,
    this body computes the same semantics with plain PyTorch: dequantise x/w,
    undo the 16x16 weight preshuffle, GEMM, Gemma RMSNorm, partial NeoX RoPE,
    and write q/outputs plus the golden-shuffle KV/index caches at
    ``slot_mapping``.  It is a correctness reference only, not a tuned kernel.

    Returns ``None``; outputs and caches are written in place.  Invalid
    mode/mode-tensor sets and tensor-contract violations raise ``ValueError``;
    a valid-but-unsupported native case raises ``NotImplementedError``.
    """

    _native(
        q,
        x,
        x_scale,
        w,
        w_scale,
        q_norm_w,
        k_norm_w,
        cos_cache,
        sin_cache,
        positions,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        mode=mode,
        k=k,
        v=v,
        index_q=index_q,
        index_q_norm_w=index_q_norm_w,
        index_k_norm_w=index_k_norm_w,
        index_cache=index_cache,
        eps=eps,
        group=group,
        stream=stream,
    )
