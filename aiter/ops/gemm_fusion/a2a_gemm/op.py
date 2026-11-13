# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AITER-owned `a2a_gemm_fused` v1 contract, route and reference native.

The tuned implementation lives in an out-of-tree AITER kernel provider (the
plugin_poc `a2a-gemm-provider`).  The native body below is a deliberately slow,
pure-PyTorch reference of the same v1 semantics: it is the fallback when no
provider serves the call, and it is what keeps the op usable on stock torch
without the provider installed.  It is **not** a performance path.

The reference is a non-exchanging single-rank stand-in for the SP4 all-to-all:
a lone process cannot perform the peer exchange, so it models the exchange the
same way the `sp_qkv_fused` reference does -- the four source ranks are assumed
to hold byte-identical send buffers.  It assembles the local destination's
`[M_real, 8192]` activation from those four simulated source shards, runs the
reference A8W8 GEMM, and writes only the real output rows.  With rank-distinct
inputs that is a layout/arithmetic reference, not a distributed oracle.
Multi-rank/capture behaviour belongs to the provider.
"""

from __future__ import annotations

from typing import Any

from aiter.plugin import route

__all__ = ["a2a_gemm_fused"]

#: The frozen interface string; ``FINGERPRINT`` is derived from it.
SIGNATURE = (
    "x:positional, x_scale:positional, w:positional, w_scale:positional, "
    "out:positional, total_tokens:keyword-only, bucket_m:keyword-only=None, "
    "registered:keyword-only=False, group:keyword-only=None, "
    "stream:keyword-only=None, -> None"
)
#: ``sha256(SIGNATURE.encode("utf-8")).hexdigest()[:16]`` -- the contract id.
FINGERPRINT = "388d9816496c260e"

_SP_WORLD = 4
_SHARD_K = 2048
_K = 8192
_N = 6144
_M_MAX = 16384


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


def _preshuffle_to_linear(w: Any, torch: Any) -> Any:
    """Undo ``shuffle_weight(layout=(16,16))`` on the [N, K] fp8 weight."""

    rows, k = int(w.shape[0]), int(w.shape[1])
    _need(rows % 16 == 0 and k >= 32 and k % 32 == 0, "w must be 16x32-block aligned")
    # Physical order after shuffle_weight: (n0, k0, k1, n1, k2).
    physical = w.float().view(rows // 16, k // 32, 2, 16, 16)
    return physical.permute(0, 3, 1, 2, 4).reshape(rows, k)


def _local_rank(group: Any) -> int:
    """The destination rank this reference process models."""

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


def _assemble_from_peers(peers: Any, destination: int, m_real: int, torch: Any) -> Any:
    """All-to-all gather: destination ``d`` takes block ``d`` of every source.

    ``peers[s]`` is source rank ``s``'s send buffer ``[4*M_real, 2048]``; its
    rows ``[d*M_real:(d+1)*M_real]`` are destination ``d``'s K quarter from
    source ``s``.  The four quarters are concatenated along K in source order,
    which is the ``[M_real, 8192]`` GEMM operand.
    """

    _need(
        len(peers) == _SP_WORLD,
        f"need {_SP_WORLD} source shards, got {len(peers)}",
    )
    _need(
        0 <= int(destination) < _SP_WORLD,
        f"destination must be in 0..{_SP_WORLD - 1}, got {destination!r}",
    )
    _need(int(m_real) >= 1, "m_real must be at least 1")
    lo = int(destination) * int(m_real)
    hi = lo + int(m_real)
    blocks = []
    for source, peer in enumerate(peers):
        _tensor(f"peers[{source}]", peer, torch)
        _need(peer.dim() == 2, f"peers[{source}] must be 2-D [4*M_real, 2048]")
        _need(
            int(peer.shape[1]) == _SHARD_K,
            f"peers[{source}] K must be {_SHARD_K}, got {int(peer.shape[1])}",
        )
        _need(
            int(peer.shape[0]) == _SP_WORLD * int(m_real),
            f"peers[{source}] must have {_SP_WORLD}*M_real="
            f"{_SP_WORLD * int(m_real)} rows, got {int(peer.shape[0])}",
        )
        blocks.append(peer[lo:hi])
    return torch.cat(blocks, dim=1)


def _native(
    x,
    x_scale,
    w,
    w_scale,
    out,
    *,
    total_tokens,
    bucket_m,
    registered,
    group,
    stream,
) -> None:
    torch = _torch()
    _need(x.dim() == 2, "x must be 2-D [4*M_real, 2048]")
    _need(
        int(x.shape[1]) == _SHARD_K,
        f"x K must be {_SHARD_K}, got {int(x.shape[1])}",
    )
    _dtype("x", x, torch.float8_e4m3fn, torch)
    _need(x.is_contiguous(), "x must be contiguous")
    rows = int(x.shape[0])
    _need(
        rows % _SP_WORLD == 0,
        f"x rows must be {_SP_WORLD}*M_real, got {rows}",
    )
    m_real = rows // _SP_WORLD
    _need(1 <= m_real <= _M_MAX, f"M_real must be in 1..{_M_MAX}, got {m_real}")

    _need(
        isinstance(total_tokens, int) and not isinstance(total_tokens, bool),
        f"total_tokens must be an int, got {type(total_tokens).__name__}",
    )
    _need(
        int(total_tokens) == rows,
        f"total_tokens must equal 4*M_real={rows} (x.shape[0]), "
        f"got {total_tokens!r}",
    )

    if bucket_m is None:
        bucket_m = m_real
    else:
        _need(
            isinstance(bucket_m, int) and not isinstance(bucket_m, bool),
            f"bucket_m must be an int or None, got {type(bucket_m).__name__}",
        )
        _need(
            m_real <= int(bucket_m) <= _M_MAX,
            f"bucket_m must be in M_real..{_M_MAX} ({m_real}..{_M_MAX}), "
            f"got {bucket_m!r}",
        )
    bucket_m = int(bucket_m)

    _need(
        isinstance(registered, bool),
        f"registered must be a bool, got {type(registered).__name__}",
    )
    _shape("x_scale", x_scale, (m_real,), torch)
    _dtype("x_scale", x_scale, torch.float32, torch)
    _need(x_scale.is_contiguous(), "x_scale must be contiguous")
    _shape("w", w, (_N, _K), torch)
    _dtype("w", w, torch.float8_e4m3fn, torch)
    _need(w.is_contiguous(), "w must be contiguous")
    _shape("w_scale", w_scale, (_N,), torch)
    _dtype("w_scale", w_scale, torch.float32, torch)
    _need(w_scale.is_contiguous(), "w_scale must be contiguous")
    _shape("out", out, (bucket_m, _N), torch)
    _dtype("out", out, torch.bfloat16, torch)

    if group is not None:
        size = getattr(group, "size", None)
        if callable(size) and int(size()) != _SP_WORLD:
            raise NotImplementedError(
                f"native a2a_gemm_fused reference supports SP world {_SP_WORLD}, "
                f"got {int(size())}"
            )
    destination = _local_rank(group)
    _need(
        0 <= destination < _SP_WORLD,
        f"group rank must be in 0..{_SP_WORLD - 1}, got {destination}",
    )

    # Non-exchanging single-rank stand-in: the four source send buffers are
    # assumed byte-identical to the local one (see the module docstring).
    activation = _assemble_from_peers((x, x, x, x), destination, m_real, torch)
    weight = _preshuffle_to_linear(w, torch)
    acc = activation.float() @ weight.t()
    scaled = acc * x_scale.float().reshape(m_real, 1)
    scaled = scaled * w_scale.float().reshape(1, _N)
    # Only the real local rows are written; out[M_real:bucket_m] is the caller's.
    out[:m_real].copy_(scaled.to(torch.bfloat16))


@route(op="a2a_gemm_fused", version=1)
def a2a_gemm_fused(
    x,
    x_scale,
    w,
    w_scale,
    out,
    *,
    total_tokens,
    bucket_m=None,
    registered=False,
    group=None,
    stream=None,
) -> None:
    """One SP4 A2A->PTPC FP8 A8W8 o_proj step (contract v1), with a slow native.

    The routed wrapper asks enabled providers first.  If none serves the call,
    this body computes the same semantics with plain PyTorch: assemble the local
    destination's ``[M_real, 8192]`` activation from the four source K quarters,
    undo the 16x16 weight preshuffle, run the reference A8W8 GEMM in FP32, apply
    the per-token and per-channel scales, and write the real local rows of
    ``out`` in place.  It is a correctness fallback for a decline, not a
    performance path; multi-rank exchange and graph capture belong to the
    provider.

    ``x`` is fp8_e4m3fn ``[4*M_real, 2048]`` per rank (rank ``s`` holds
    destination blocks ``d=0..3``; rows ``[d*M_real:(d+1)*M_real]`` are
    destination ``d``'s K quarter from source ``s``); ``x_scale`` is fp32
    ``[M_real]``, one scale per token shared across all 8192 channels; ``w`` is
    the fp8_e4m3fn ``[6144, 8192]`` 16x16 preshuffled weight with fp32
    ``[6144]`` ``w_scale``; ``out`` is the caller-owned bf16 ``[bucket_m, 6144]``
    output.  ``total_tokens`` is ``4*M_real``; ``bucket_m`` defaults to
    ``M_real`` and rows ``[M_real:bucket_m]`` are left untouched.  ``registered``
    is a producer-owned-pointer hint with no meaning in the native reference;
    ``group``/``stream`` are execution context only.

    Returns ``None``; ``out`` is written in place.  Invalid tensor contracts,
    a ``total_tokens`` that is not ``4*M_real``, or a ``bucket_m`` outside
    ``M_real..16384`` raise ``ValueError`` before anything is written; a valid
    but unsupported native case (non-4 SP world) raises ``NotImplementedError``.
    """

    _native(
        x,
        x_scale,
        w,
        w_scale,
        out,
        total_tokens=total_tokens,
        bucket_m=bucket_m,
        registered=registered,
        group=group,
        stream=stream,
    )
