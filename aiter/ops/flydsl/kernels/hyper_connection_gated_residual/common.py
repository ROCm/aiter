# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared building blocks for the two-stage Gated-Residual (K1/K2) kernels.

Holds what both stages depend on: (1) the per-architecture bf16 MFMA shape +
K-permutation layout (:class:`MfmaConfig`, :func:`mfma_bf16`, :func:`ab_k_perm`)
that keep the GEMM bodies architecture-agnostic, and (2) the small weight/epilogue
helpers -- the merged down+inject weight pack/slice (:func:`merge_down_inject`,
:func:`split_down_inject`) and the split-K cross-reduction + SiLU epilogue
(:func:`_build_reduce_silu`).

The down and up GEMMs contract between the hidden dimension and a much narrower
low-rank dimension. Matrix-core shape and wave tiling therefore differ by ASIC.
gfx950 (CDNA4) uses the bf16 ``16x16x32`` matrix core; gfx942 (CDNA3) exposes
``16x16x16``. Both accumulate in float32. The ``k_group`` field is ``mma_k // 4``,
the per-instruction K packing expected by the tiled-MMA layout.
"""

import contextlib
from dataclasses import dataclass
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.tensor_shim import GTensor


@dataclass(frozen=True)
class MfmaConfig:
    mma_m: int
    mma_n: int
    mma_k: int

    @property
    def k_group(self) -> int:
        return self.mma_k // 4


# bf16 matrix-core shape per architecture (float32 accumulate).
_MFMA_BF16 = {
    "gfx950": MfmaConfig(16, 16, 32),
    "gfx942": MfmaConfig(16, 16, 16),
}


def arch_name(device=None) -> str:
    """Normalized GCN arch string (e.g. ``"gfx942"``) used as the routing key.

    Runtime dispatch follows the shared chip-info policy. ``device`` is accepted
    for call-site compatibility; supported ROCm nodes are homogeneous.
    """
    return get_gfx_runtime()


def require_supported_arch(device=None) -> str:
    """Live arch, or ``NotImplementedError`` if no kernel path was validated on it.

    Every public entry point calls this before dispatch: the skinny decode relies
    on the gfx942/gfx950 cache policy and the tiled paths on their MFMA shapes.
    """
    arch = arch_name(device)
    if arch not in _MFMA_BF16:
        raise NotImplementedError(
            f"two-stage HC gated-residual supports {sorted(_MFMA_BF16)}, got {arch!r}"
        )
    return arch


def mfma_bf16(arch: str | None = None) -> MfmaConfig:
    """bf16 MFMA shape for ``arch`` (defaults to the live device)."""
    if arch is None:
        arch = arch_name()
    try:
        return _MFMA_BF16[arch]
    except KeyError as exc:
        raise NotImplementedError(
            f"no bf16 MFMA shape registered for arch {arch!r}; "
            f"known: {sorted(_MFMA_BF16)}"
        ) from exc


def ab_k_perm(mma_k: int, *, dtype_width: int = 16, copy_bits: int = 128):
    """K-axis permutation pairing a ``copy_bits`` A/B load with the MFMA K fragment.

    A single ``copy_bits`` load holds ``copy_bits // dtype_width`` elements per
    lane; one MFMA instruction consumes ``k_group = mma_k // 4`` of them. On
    gfx950 (bf16 ``mma_k=32``) a 128-bit load is exactly one MFMA K-group
    (``k_group=8``), so the layout is the flat ``(k_group, 4)`` form. On gfx942
    (bf16 ``mma_k=16``) ``k_group=4``, so the same 128-bit load spans
    ``num_frgv = 2`` MFMA K-groups; the extra axis interleaves them so the
    tiled-copy partition derived from this tiled-MMA matches the wide load
    instead of emitting a mismatched-width fragment cast.

    Uses the flat ``(k_group, 4):(1, k_group)`` layout when one load contains one
    K-group (``num_frgv <= 1``).
    """
    k_group = mma_k // 4
    num_frgv = copy_bits // (k_group * dtype_width)
    if num_frgv <= 1:
        return fx.make_layout((k_group, 4), (1, k_group))
    num_elems = copy_bits // dtype_width
    return fx.make_layout((k_group, 4, num_frgv), (1, num_elems, k_group))


_NO_SCOPE = contextlib.nullcontext()


def stream_scope(stream: torch.cuda.Stream | int | None, device: torch.device):
    """Return ``(context, raw_stream)`` for launching on ``stream``.

    Tensors created inside ``context`` are allocated on, and ordered against,
    ``stream``. Without this, the caching allocator associates temporaries with
    the ambient stream and may recycle them while the launch stream still uses
    them. ``None`` selects ``device``'s current stream; an ``int`` is a raw
    handle already resolved by an enclosing scope. The raw handle avoids
    constructing a ``torch.cuda.Stream`` per call on the latency-bound path.

    A ``torch.cuda.Stream`` must belong to ``device``. When ``device`` is not the
    current device the context also makes it current, so launches and
    temporaries land on the tensors' device.
    """
    if isinstance(stream, int):
        return _NO_SCOPE, stream
    if stream is not None and stream.device != device:
        raise ValueError(
            f"stream is on {stream.device}, but the tensors are on {device}"
        )
    current = torch._C._cuda_getCurrentRawStream(device.index)
    raw = current if stream is None else stream.cuda_stream
    if device.index != torch.cuda.current_device():
        ctx = contextlib.ExitStack()
        ctx.enter_context(torch.cuda.device(device))
        if raw != current:
            ctx.enter_context(torch.cuda.stream(stream))
        return ctx, raw
    if raw == current:
        return _NO_SCOPE, raw
    return torch.cuda.stream(stream), raw


# The combine/RMS prologue covers a stream with one wave of 8-wide bf16 loads.
STREAM_DIM_ALIGN = 64 * 8


def check_streams(hidden: int, hc_count: int) -> int:
    """Return ``stream_dim``; ``hidden`` must split into ``hc_count`` streams."""
    if hc_count <= 0 or hidden % hc_count:
        raise ValueError(f"hidden={hidden} must be a multiple of hc_count={hc_count}")
    return hidden // hc_count


def check_hidden(hidden: int, hc_count: int) -> int:
    """Return ``stream_dim``; reject widths the K1 combine/RMS prologue cannot tile."""
    stream_dim = check_streams(hidden, hc_count)
    if stream_dim % STREAM_DIM_ALIGN:
        raise ValueError(
            f"stream_dim=hidden/hc_count={stream_dim} must be a multiple of "
            f"{STREAM_DIM_ALIGN}"
        )
    return stream_dim


def check_tensor(
    name: str,
    t: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    contiguous: bool = True,
) -> None:
    """Reject tensors the kernels' fixed row-major views cannot represent."""
    if t.device != device:
        raise ValueError(f"{name} must be on {device}, got {t.device}")
    if t.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {t.dtype}")
    if t.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(t.shape)}")
    if contiguous and not t.is_contiguous():
        raise ValueError(f"{name} must be row-major contiguous")


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """Half-open address range covering every element of ``t``."""
    start = t.data_ptr()
    if t.is_contiguous():
        return start, start + t.numel() * t.element_size()
    last = sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
    return start, start + (last + 1) * t.element_size()


def check_no_overlap(
    outputs: dict[str, torch.Tensor | None], inputs: dict[str, torch.Tensor | None]
) -> None:
    """Reject caller outputs whose memory overlaps an input or another output.

    Kernels write outputs while later stages still read inputs and earlier
    outputs, so no alias is safe. The check compares address ranges, so a
    strided view that interleaves another tensor without sharing an element is
    rejected too.
    """
    outs = [
        (k, _byte_span(t)) for k, t in outputs.items() if t is not None and t.numel()
    ]
    if not outs:
        return
    ins = [(k, _byte_span(t)) for k, t in inputs.items() if t is not None and t.numel()]
    for i, (name, (lo, hi)) in enumerate(outs):
        for other, (olo, ohi) in outs[i + 1 :] + ins:
            if lo < ohi and olo < hi:
                raise ValueError(f"{name} must not overlap {other} in memory")


def check_norm_weight(
    norm_weight: torch.Tensor, stream_dim: int, hidden: int, device: torch.device
) -> None:
    """``norm_weight`` is either one shared stream or one weight per channel."""
    if norm_weight.device != device:
        raise ValueError(f"norm_weight must be on {device}, got {norm_weight.device}")
    if norm_weight.numel() not in (stream_dim, hidden):
        raise ValueError(
            f"norm_weight must have {stream_dim} (shared) or {hidden} elements, "
            f"got {norm_weight.numel()}"
        )


def norm_weight_f32(norm_weight: torch.Tensor) -> torch.Tensor:
    """``norm_weight`` as a flat contiguous f32 buffer for the norm kernels.

    The kernels read the affine weight as contiguous f32. A caller already holding
    it in that form passes through untouched; a bf16 or non-contiguous weight is
    cast and packed here.
    """
    if norm_weight.dtype == torch.float32 and norm_weight.is_contiguous():
        return norm_weight.reshape(-1)
    return norm_weight.reshape(-1).float().contiguous()


def merge_down_inject(
    w_down: torch.Tensor, w_inject: torch.Tensor, n_pad: int
) -> torch.Tensor:
    """Pack ``w_down`` [lowrank, H] and ``w_inject`` [hc, H] into one [n_pad, H].

    Rows ``[0, lowrank)`` are ``w_down``, ``[lowrank, lowrank+hc)`` are
    ``w_inject``, the remainder is zero padding so ``n_pad`` is MFMA-friendly.
    """
    if w_down.ndim != 2 or w_inject.ndim != 2:
        raise ValueError("w_down and w_inject must be 2-D")
    lowrank, hidden = w_down.shape
    hc = w_inject.shape[0]
    check_tensor(
        "w_inject",
        w_inject,
        (hc, hidden),
        w_down.device,
        dtype=w_down.dtype,
        contiguous=False,
    )
    if n_pad < lowrank + hc:
        raise ValueError(f"n_pad={n_pad} must be >= lowrank + hc = {lowrank + hc}")
    merged = torch.zeros(n_pad, hidden, dtype=w_down.dtype, device=w_down.device)
    merged[:lowrank] = w_down
    merged[lowrank : lowrank + hc] = w_inject
    return merged


def split_down_inject(
    out: torch.Tensor, lowrank: int, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the packed down+inject output into ``(lora, inj_next)`` views."""
    return out[:, :lowrank], out[:, lowrank : lowrank + hc_count]


@lru_cache(maxsize=32)
def _build_reduce_silu(
    total: int,
    split_k: int,
    silu_cols: int,
    row_width: int,
    hc_count: int,
    block_threads: int,
    vec: int,
):
    inv_hc = 1.0 / hc_count
    all_silu = silu_cols >= row_width
    # One workgroup per element group exposes the full reduction grid to the CUs.
    n_iters = 1
    assert (
        total % (block_threads * vec) == 0
    ), f"total={total} must be a multiple of block_threads*vec={block_threads * vec}"

    @flyc.kernel(
        name=f"gr_down_reduce_t{total}_sk{split_k}_s{silu_cols}_w{row_width}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        partial: fx.Tensor,  # [split_k*M, row_width] f32
        lora: fx.Tensor,  # [M, row_width] bf16
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        part_g = GTensor(partial, T.f32, (1, total))
        lora_g = GTensor(lora, T.bf16, (1, total))
        for it in range_constexpr(n_iters):
            base = (
                (bid * fx.Int32(n_iters) + it) * fx.Int32(block_threads) * fx.Int32(vec)
            )
            off = base + tid * fx.Int32(vec)
            acc = fx.Vector(part_g.load(off, vec_size=vec))
            for k in range_constexpr(1, split_k):
                acc = acc + fx.Vector(
                    part_g.load(off + k * fx.Int32(total), vec_size=vec)
                )
            out = []
            for e in range_constexpr(vec):
                v = acc[e] * fx.Float32(inv_hc)
                if all_silu:
                    out.append(v * sigmoid_f32(v))
                else:
                    col = fx.get_scalar((off + e) % fx.Int32(row_width))
                    if isinstance(col, int):
                        out.append((v * sigmoid_f32(v)) if col < silu_cols else acc[e])
                    else:
                        is_lora = col < fx.Int32(silu_cols)
                        out.append(is_lora.select(v * sigmoid_f32(v), acc[e]))
            lora_g.store(
                off,
                fx.Vector.from_elements(out, dtype=fx.Float32).to(fx.BFloat16),
                vec_size=vec,
            )

    @flyc.jit
    def launch(
        partial: fx.Tensor,
        lora: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        grid = total // (block_threads * vec * n_iters)
        kernel(partial, lora).launch(
            grid=(fx.Int64(grid), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch
