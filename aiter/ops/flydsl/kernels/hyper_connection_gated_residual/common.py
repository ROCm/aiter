# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared building blocks for the two-stage Gated-Residual (K1/K2) kernels.

Holds what both stages depend on: (1) the per-architecture bf16 MFMA shape +
K-permutation layout (:class:`MfmaConfig`, :func:`mfma_bf16`, :func:`ab_k_perm`)
that keep the GEMM bodies architecture-agnostic, and (2) the small weight/epilogue
helpers -- the merged down+inject weight pack/slice (:func:`merge_down_inject`,
:func:`split_down_inject`) and the split-K cross-reduction + SiLU epilogue
(:func:`_build_reduce_silu`).

The GEMMs are tall-and-skinny -- down contracts the full hidden (10240) into the
~320-wide low-rank bottleneck; up contracts the 320 bottleneck back into 10240 --
so the matrix-core shape and wave tiling matter and differ by ASIC. gfx950 (CDNA4)
uses the bf16 ``16x16x32`` matrix core; gfx942 (CDNA3) exposes ``16x16x16``. Both
accumulate in float32. The ``k_group`` field is ``mma_k // 4`` -- the
per-instruction K packing the tiled-MMA layout expects.
"""

from dataclasses import dataclass

import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_gfx
from functools import lru_cache
import flydsl.compiler as flyc
import torch
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.act import _sigmoid_f32
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, _run_compiled


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

    Prefers the tensor's own ``device`` when given (correct under multi-GPU or a
    non-current device); falls back to the live device otherwise. Centralizes the
    ``gcnArchName`` parsing so every kernel keys off the same value.
    """
    if device is not None:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return get_gfx()


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

    Collapses to the original ``(k_group, 4):(1, k_group)`` layout whenever a
    load is one K-group (``num_frgv <= 1``), so gfx950 codegen is unchanged.
    """
    k_group = mma_k // 4
    num_frgv = copy_bits // (k_group * dtype_width)
    if num_frgv <= 1:
        return fx.make_layout((k_group, 4), (1, k_group))
    num_elems = copy_bits // dtype_width
    return fx.make_layout((k_group, 4, num_frgv), (1, num_elems, k_group))


def merge_down_inject(
    w_down: torch.Tensor, w_inject: torch.Tensor, n_pad: int
) -> torch.Tensor:
    """Pack ``w_down`` [lowrank, H] and ``w_inject`` [hc, H] into one [n_pad, H].

    Rows ``[0, lowrank)`` are ``w_down``, ``[lowrank, lowrank+hc)`` are
    ``w_inject``, the remainder is zero padding so ``n_pad`` is MFMA-friendly.
    """
    lowrank, hidden = w_down.shape
    hc = w_inject.shape[0]
    assert w_inject.shape == (hc, hidden)
    assert n_pad >= lowrank + hc
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
    # One workgroup per element-group so the reduction is fully parallel across
    # CUs; a single grid-strided workgroup (the naive grid) serializes the whole
    # output and dominates wall time at large M.
    n_iters = 1
    assert total % (block_threads * vec) == 0, (
        f"total={total} must be a multiple of block_threads*vec={block_threads * vec}"
    )

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
        part_g = GTensor(partial, fx.Float32, (1, total))
        lora_g = GTensor(lora, fx.BFloat16, (1, total))
        for it in range_constexpr(n_iters):
            base = (bid * fx.Int32(n_iters) + it) * fx.Int32(block_threads) * fx.Int32(vec)
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
                    out.append(v * _sigmoid_f32(v))
                else:
                    col = fx.get_scalar((off + e) % fx.Int32(row_width))
                    if isinstance(col, int):
                        out.append((v * _sigmoid_f32(v)) if col < silu_cols else acc[e])
                    else:
                        is_lora = col < fx.Int32(silu_cols)
                        out.append(is_lora.select(v * _sigmoid_f32(v), acc[e]))
            lora_g.store(
                off, fx.Vector.from_elements(out, dtype=fx.Float32).to(fx.BFloat16),
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
