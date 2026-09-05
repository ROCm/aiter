# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA v4 recipe selection, validation, and launch APIs.

Raw BF16 BSHD operands are delegated to producers in :mod:`mha_v4_quant`.
Format and scale-mode IDs are part of the launcher ABI. Optional block-sparse
execution uses a boolean tile mask on the raw API and a ragged LUT triple on
the packed API; the work table is built inside the sparse custom op.
"""

import csv
import functools
import os
from enum import IntEnum
from typing import Optional

import torch
from torch import Tensor

from aiter.jit.core import AITER_ROOT_DIR, compile_ops
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.mha_v4_quant import (
    MHA_V4_LOG2E,
    mha_v4_q_multiplier,
    mxfp4_k_view,
    mxfp4_v_view,
    mxfp6_k_view,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_int8,
    quantize_mxfp4_k,
    quantize_mxfp4_q,
    quantize_mxfp6_k,
    quantize_mxfp6_q,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    quantize_v_fp8,
    quantize_v_mxfp4,
    quantize_v_mxfp4_fp6_p,
    quantize_v_mxfp6,
    quantize_v_mxfp6_fp6_p,
    rotate_activation_hd128,
    rotate_activation_mxfp6_quant,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut

__all__ = (
    "MHA_V4_LOG2E",
    "AttentionFormat",
    "AttentionPack",
    "AttentionScaleMode",
    "mha_v4",
    "mha_v4_kv_tile",
    "mha_v4_packed",
    "mha_v4_q_multiplier",
    "mha_v4_sparse_work_table",
    "mxfp4_k_view",
    "mxfp4_v_view",
    "mxfp6_k_view",
    "native_fp8_format",
    "quantize_fp8",
    "quantize_fp8_rotated",
    "quantize_int8",
    "quantize_mxfp4_k",
    "quantize_mxfp4_q",
    "quantize_mxfp6_k",
    "quantize_mxfp6_q",
    "quantize_mxfp8_k",
    "quantize_mxfp8_q",
    "quantize_v_fp8",
    "quantize_v_mxfp4",
    "quantize_v_mxfp4_fp6_p",
    "quantize_v_mxfp6",
    "quantize_v_mxfp6_fp6_p",
    "rotate_activation_hd128",
    "rotate_activation_mxfp6_quant",
    "scale_modes_for_formats",
)


def _mha_v4_sparse_work_table_fake(
    lut_count: Tensor,
    batch: int,
    nhead: int,
    q_tiles: int,
) -> Tensor:
    del lut_count
    return torch.empty(batch * nhead * q_tiles, dtype=torch.int32, device="cuda")


@compile_ops("module_fmha_v4_fwd", gen_fake=_mha_v4_sparse_work_table_fake)
def mha_v4_sparse_work_table(
    lut_count: Tensor,
    batch: int,
    nhead: int,
    q_tiles: int,
) -> Tensor:
    """Return the tile visit order the sparse kernel would use for these LUT lengths.

    Exposed for testing. The sparse launcher builds this itself, and a wrong order only unbalances
    the waves rather than changing the result, so no test of the attention output can see it.

    Each entry packs one tile as ``q_tile | head << 16 | batch << 24``, ordered by LUT length
    descending with ties left in raster order.
    """


class AttentionFormat(IntEnum):
    """Stable operand-encoding IDs used by the Python/C++ dispatch ABI.

    MX names are aliases for their underlying element encoding; scale
    granularity is represented independently by :class:`AttentionScaleMode`.
    """

    FP32 = 0
    FP16 = 1
    BF16 = 2
    FP8_E4M3 = 3
    FP8_E4M3_FNUZ = 4
    FP8_E5M2 = 5
    FP8_E5M2_FNUZ = 6
    FP6_E2M3 = 7
    FP6_E3M2 = 8
    FP4_E2M1 = 9
    INT8 = 10
    UINT8 = 11
    INT4 = 12
    UINT4 = 13
    # Aliases
    FP8 = FP8_E4M3
    MXFP6_E2M3 = FP6_E2M3
    MXFP6 = FP6_E2M3
    MXFP6_E3M2 = FP6_E3M2
    MXBF6 = FP6_E3M2
    MXFP4 = FP4_E2M1


class AttentionPack(IntEnum):
    """Stable IDs describing operand layouts within a numeric format."""

    DEFAULT = 0
    V_FOR_FP6_P = 1


class AttentionScaleMode(IntEnum):
    """Stable IDs describing how each operand's descale tensor is indexed."""

    NONE = 0
    F32_PER_TENSOR = 1
    F32_PER_HEAD = 2
    F32_PER_TOKEN = 3
    F32_PER_CHANNEL = 4
    E8M0_PER_1X32 = 5


_FP8_FORMATS = (AttentionFormat.FP8_E4M3, AttentionFormat.FP8_E4M3_FNUZ)
_MX_FORMATS = (AttentionFormat.FP6_E2M3, AttentionFormat.FP4_E2M1)
_MXFP8_SCALE_MODES = (
    AttentionScaleMode.E8M0_PER_1X32,
    AttentionScaleMode.E8M0_PER_1X32,
    AttentionScaleMode.F32_PER_TENSOR,
)
_PACKED_QK_WIDTH = {
    AttentionFormat.BF16: 128,
    AttentionFormat.INT8: 128,
    AttentionFormat.FP8_E4M3: 128,
    AttentionFormat.FP8_E4M3_FNUZ: 128,
    AttentionFormat.FP6_E2M3: 96,
    AttentionFormat.FP4_E2M1: 64,
}

_MHA_V4_Q_TILE = 256
# mode=1 selects the sorted-sparse manifest rows; the launcher dispatches the same rows through
# find_config(..., mode=1).
_MHA_V4_SPARSE_MODE = 1


def native_fp8_format() -> AttentionFormat:
    """Return the FP8 E4M3 encoding native to the active GPU architecture."""
    return (
        AttentionFormat.FP8_E4M3_FNUZ
        if get_gfx() == "gfx942"
        else AttentionFormat.FP8_E4M3
    )


@functools.cache
def mha_v4_kv_tile() -> int:
    """Return the KV tile of sorted-sparse MHA v4 rows on the active GPU.

    Read from the same manifest the launcher dispatches on rather than restated here, so adding a
    sparse row with a different tile cannot leave the two disagreeing. 256x128 on gfx950, 256x64 on
    gfx942.
    """
    return _mha_v4_kv_tile_from_manifest()


# The read has to sit behind the guard, not just behind the cache above: Dynamo traces the body of a
# cached function regardless of whether the cache is warm, and `open` is not traceable, so an
# unguarded read costs two graph breaks on every mha_v4(block_mask=...) trace and fails outright
# under fullgraph=True. get_gfx() keeps its own rocminfo probe opaque the same way.
@torch_compile_guard()
def _mha_v4_kv_tile_from_manifest() -> int:
    gfx = get_gfx()
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    manifest = os.path.join(asm_dir, gfx, "fmha_v4_fwd", "fmha_v4_fwd.csv")
    tiles = set()
    try:
        with open(manifest, newline="") as handle:
            for row in csv.DictReader(
                filter(lambda line: not line.startswith("#"), handle)
            ):
                if int(row["mode"]) == _MHA_V4_SPARSE_MODE:
                    tiles.add(int(row["ts_kv"]))
    except FileNotFoundError as error:
        raise ValueError(
            f"no MHA v4 manifest for {gfx} at {manifest}; sorted-sparse MHA v4 is "
            "unavailable on this GPU"
        ) from error
    if not tiles:
        raise ValueError(f"{gfx} has no sorted-sparse MHA v4 manifest row")
    if len(tiles) > 1:
        raise ValueError(
            f"{gfx} sorted-sparse manifest rows disagree on ts_kv ({sorted(tiles)}); the "
            "mask geometry a caller builds is only well defined when they agree"
        )
    return tiles.pop()


def _is_fp8_format(format: AttentionFormat) -> bool:
    return format in _FP8_FORMATS


def _validate_format_contract(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> None:
    if q_format == AttentionFormat.FP6_E3M2:
        raise NotImplementedError(
            "FP6 E3M2 has a reserved format ID but no kernel row yet"
        )
    if q_format != k_format:
        raise ValueError("MHA v4 currently requires matching Q and K formats")
    if q_format == AttentionFormat.BF16:
        if v_format != AttentionFormat.BF16 and not _is_fp8_format(v_format):
            raise ValueError("BF16 Q/K currently requires BF16 or FP8 V")
        return
    if q_format not in _PACKED_QK_WIDTH:
        raise ValueError(f"unsupported Q/K format: {q_format!r}")
    if v_format not in (
        *_FP8_FORMATS,
        AttentionFormat.FP6_E2M3,
        AttentionFormat.FP4_E2M1,
    ):
        raise ValueError(f"unsupported V format: {v_format!r}")
    if q_format == AttentionFormat.INT8 and v_format not in _FP8_FORMATS:
        raise ValueError("INT8 Q/K currently requires FP8 V")
    if q_format in _FP8_FORMATS and v_format not in (
        q_format,
        AttentionFormat.MXFP6,
    ):
        raise ValueError("FP8 Q/K requires matching FP8 or MXFP6 V")


def _validate_pack_contract(
    v_format: AttentionFormat,
    v_pack: AttentionPack,
) -> None:
    if v_pack == AttentionPack.DEFAULT:
        return
    if v_pack == AttentionPack.V_FOR_FP6_P and v_format in (
        AttentionFormat.FP6_E2M3,
        AttentionFormat.FP4_E2M1,
    ):
        return
    raise ValueError(
        f"unsupported V pack {v_pack.name} for format {v_format.name}"
    )


def scale_modes_for_formats(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
) -> tuple[AttentionScaleMode, AttentionScaleMode, AttentionScaleMode]:
    """Return the canonical Q, K, and V scale modes for a format recipe."""
    _validate_format_contract(q_format, k_format, v_format)
    if q_format == AttentionFormat.BF16:
        return (
            AttentionScaleMode.NONE,
            AttentionScaleMode.NONE,
            (
                AttentionScaleMode.NONE
                if v_format == AttentionFormat.BF16
                else AttentionScaleMode.F32_PER_TENSOR
            ),
        )
    if q_format == AttentionFormat.INT8 or q_format in _FP8_FORMATS:
        v_scale_mode = (
            AttentionScaleMode.F32_PER_TENSOR
            if _is_fp8_format(v_format)
            else AttentionScaleMode.E8M0_PER_1X32
        )
        return (
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            v_scale_mode,
        )
    if q_format in _MX_FORMATS:
        v_scale_mode = (
            AttentionScaleMode.F32_PER_CHANNEL
            if _is_fp8_format(v_format)
            else AttentionScaleMode.E8M0_PER_1X32
        )
        return (
            AttentionScaleMode.E8M0_PER_1X32,
            AttentionScaleMode.E8M0_PER_1X32,
            v_scale_mode,
        )
    raise NotImplementedError(
        f"raw preprocessing is not implemented for Q/K format {q_format.name}"
    )


def _validate_scale_recipe(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    scale_modes: tuple[AttentionScaleMode, AttentionScaleMode, AttentionScaleMode],
) -> None:
    canonical_scale_modes = scale_modes_for_formats(q_format, k_format, v_format)
    is_mxfp8_recipe = (
        q_format in _FP8_FORMATS
        and k_format == q_format
        and v_format == q_format
        and scale_modes == _MXFP8_SCALE_MODES
    )
    if scale_modes != canonical_scale_modes and not is_mxfp8_recipe:
        raise ValueError(
            "unsupported scale recipe for formats: "
            f"got {tuple(mode.name for mode in scale_modes)}, "
            f"expected {tuple(mode.name for mode in canonical_scale_modes)}"
        )


def _raw_scale_recipe(
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    q_scale_mode: Optional[AttentionScaleMode],  # noqa: UP045
    k_scale_mode: Optional[AttentionScaleMode],  # noqa: UP045
    v_scale_mode: Optional[AttentionScaleMode],  # noqa: UP045
) -> tuple[AttentionScaleMode, AttentionScaleMode, AttentionScaleMode]:
    provided = (
        q_scale_mode is not None,
        k_scale_mode is not None,
        v_scale_mode is not None,
    )
    if not any(provided):
        return scale_modes_for_formats(q_format, k_format, v_format)
    if not all(provided):
        raise ValueError(
            "q_scale_mode, k_scale_mode, and v_scale_mode must all be set or all omitted"
        )
    scale_modes = (q_scale_mode, k_scale_mode, v_scale_mode)
    _validate_scale_recipe(q_format, k_format, v_format, scale_modes)
    return scale_modes


def _packed_lut_triple(
    kv_block_indices: Optional[Tensor],  # noqa: UP045
    lut_start: Optional[Tensor],  # noqa: UP045
    lut_count: Optional[Tensor],  # noqa: UP045
) -> Optional[tuple[Tensor, Tensor, Tensor]]:  # noqa: UP045
    present = (
        kv_block_indices is not None,
        lut_start is not None,
        lut_count is not None,
    )
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "kv_block_indices, lut_start, and lut_count must all be set or all omitted"
        )
    return kv_block_indices, lut_start, lut_count


def _block_mask_to_lut(
    block_mask: Tensor, query: Tensor, key: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    batch, query_length, query_heads, _ = query.shape
    key_length = key.shape[1]
    q_tiles = (query_length + _MHA_V4_Q_TILE - 1) // _MHA_V4_Q_TILE
    kv_tile = mha_v4_kv_tile()
    kv_tiles = (key_length + kv_tile - 1) // kv_tile
    # The conversion below counts selected blocks with an arithmetic sum but fills the index list
    # from truthiness, so anything other than bool can make lut_count disagree with the entries
    # actually written.
    if block_mask.dtype != torch.bool:
        raise ValueError(f"block_mask must be a bool tensor, got {block_mask.dtype}")
    if block_mask.device != query.device:
        raise ValueError(
            f"block_mask must be on the same device as Q; got {block_mask.device} "
            f"and {query.device}"
        )
    if block_mask.dim() == 4:
        expected = (batch, query_heads, q_tiles, kv_tiles)
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                "block_mask must have shape [batch, heads, "
                f"ceil(Sq/{_MHA_V4_Q_TILE}), ceil(Sk/{kv_tile})]; "
                f"got {tuple(block_mask.shape)}, expected {expected}"
            )
    elif block_mask.dim() == 3:
        expected = (batch, q_tiles, kv_tiles)
        if tuple(block_mask.shape) != expected:
            raise ValueError(
                "block_mask must have shape [batch, "
                f"ceil(Sq/{_MHA_V4_Q_TILE}), ceil(Sk/{kv_tile})] "
                f"or the 4-D per-head form; got {tuple(block_mask.shape)}, "
                f"expected {expected}"
            )
    else:
        raise ValueError(
            "block_mask must be 3-D [batch, Qtiles, KVtiles] or "
            "4-D [batch, heads, Qtiles, KVtiles]"
        )
    lut = block_attn_mask_to_ragged_lut(
        block_mask,
        num_heads=query_heads,
        return_none_if_dense=False,
    )
    if lut is None:
        raise RuntimeError("block_attn_mask_to_ragged_lut returned None")
    return lut


def _fmha_v4_fwd_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format, v_pack
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd",
    gen_fake=_fmha_v4_fwd_fake,
)
def _fmha_v4_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_launch", mutates_args=("out",))
def _mha_v4_fwd_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    _fmha_v4_fwd(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        v_pack,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
    )


@_mha_v4_fwd_launch.register_fake
def _mha_v4_fwd_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format, v_pack
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale


def _fmha_v4_fwd_sparse_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale
    del q_format, k_format, v_format, v_pack
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count
    del out


@compile_ops(
    "module_fmha_v4_fwd",
    fc_name="fmha_v4_fwd_sparse",
    gen_fake=_fmha_v4_fwd_sparse_fake,
)
def _fmha_v4_fwd_sparse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
) -> None: ...


@torch.library.custom_op("aiter::mha_v4_fwd_sparse_launch", mutates_args=("out",))
def _mha_v4_fwd_sparse_launch(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
) -> None:
    _fmha_v4_fwd_sparse(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        q_format,
        k_format,
        v_format,
        v_pack,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale,
        kv_block_indices,
        lut_start,
        lut_count,
    )


@_mha_v4_fwd_sparse_launch.register_fake
def _mha_v4_fwd_sparse_launch_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    out: Tensor,
    q_format: int,
    k_format: int,
    v_format: int,
    v_pack: int,
    q_scale_mode: int,
    k_scale_mode: int,
    v_scale_mode: int,
    softmax_scale: float,
    kv_block_indices: Tensor,
    lut_start: Tensor,
    lut_count: Tensor,
) -> None:
    del q, k, v, q_descale, k_descale, v_descale, out
    del q_format, k_format, v_format, v_pack
    del q_scale_mode, k_scale_mode, v_scale_mode, softmax_scale
    del kv_block_indices, lut_start, lut_count


def mha_v4_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    q_scale_mode: AttentionScaleMode,
    k_scale_mode: AttentionScaleMode,
    v_scale_mode: AttentionScaleMode,
    *,
    v_pack: AttentionPack = AttentionPack.DEFAULT,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    kv_block_indices: Optional[Tensor] = None,  # noqa: UP045
    lut_start: Optional[Tensor] = None,  # noqa: UP045
    lut_count: Optional[Tensor] = None,  # noqa: UP045
) -> Tensor:
    """Launch non-causal MHA v4 over pre-quantized BSHD operands.

    Formats and scale modes select an explicit ASM row. Packed widths and
    nonstandard K layouts are validated before launch; output is BF16 BSHD.
    Pass the ragged LUT triple to select the sorted-sparse row; omit all three
    tensors for dense. The work table is built inside the sparse custom op.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    lut = _packed_lut_triple(kv_block_indices, lut_start, lut_count)
    _validate_pack_contract(v_format, v_pack)
    scale_modes = (q_scale_mode, k_scale_mode, v_scale_mode)
    _validate_scale_recipe(q_format, k_format, v_format, scale_modes)

    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("MHA v4 expects BSHD Q, K, and V tensors")
    batch, query_length, query_heads, _ = q.shape
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("Q, K, and V must have the same batch size")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("K and V must have matching sequence and head dimensions")
    kv_heads = k.shape[2]
    if kv_heads == 0:
        raise ValueError("MHA v4 requires non-empty KV heads")
    if query_heads % kv_heads != 0:
        raise ValueError("MHA v4 requires query heads to be divisible by KV heads")
    gqa_ratio = query_heads // kv_heads
    if gqa_ratio > 16 or gqa_ratio & (gqa_ratio - 1):
        raise ValueError("MHA v4 supports power-of-two GQA ratios up to 16")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("MHA v4 expects GPU tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Q, K, and V must have contiguous last dimensions")

    logical_head_dim = 128
    expected_q_width = _PACKED_QK_WIDTH[q_format]
    if q.shape[-1] != expected_q_width or k.shape[-1] != expected_q_width:
        raise ValueError(
            f"{q_format.name} Q/K must have packed width {expected_q_width}"
        )
    if v.shape[-1] != logical_head_dim:
        raise ValueError("MHA v4 currently requires logical V head dimension 128")
    if q_format == AttentionFormat.MXFP4:
        tiles = (k.shape[1] + 127) // 128
        head_stride = tiles * 8192
        expected_k_stride = (k.shape[2] * head_stride, 64, head_stride, 1)
        if k.stride() != expected_k_stride:
            raise ValueError("MXFP4 K must use the coalesced MHA v4 tile layout")

    if softmax_scale is None:
        softmax_scale = logical_head_dim**-0.5
    if out is None:
        out = torch.empty(
            (batch, query_length, query_heads, logical_head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
    elif out.shape != (batch, query_length, query_heads, logical_head_dim):
        raise ValueError("out has the wrong shape for MHA v4")
    elif out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must be a BF16 tensor on the same device as Q")

    launch_args = (
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        out,
        int(q_format),
        int(k_format),
        int(v_format),
        int(v_pack),
        int(q_scale_mode),
        int(k_scale_mode),
        int(v_scale_mode),
        softmax_scale,
    )
    if lut is None:
        _mha_v4_fwd_launch(*launch_args)
    else:
        if q_format == AttentionFormat.BF16:
            raise NotImplementedError(
                "sorted-sparse MHA v4 does not have a BF16 manifest row yet"
            )
        kv_tile = mha_v4_kv_tile()
        if k.shape[1] % kv_tile != 0:
            raise ValueError(
                "sorted-sparse MHA v4 requires key length padded to a "
                f"multiple of {kv_tile}"
            )
        _mha_v4_fwd_sparse_launch(*launch_args, *lut)
    return out


@torch.library.custom_op(
    "aiter::mha_v4_launch_mxfp4_coalesced_v3", mutates_args=("out",)
)
def _launch_mxfp4_coalesced(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    v_pack: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    resolved_v_pack = AttentionPack(v_pack)
    k = mxfp4_k_view(k_data, k_descale)
    v = (
        v_data
        if _is_fp8_format(resolved_v_format)
        else mxfp4_v_view(v_data, v_descale, k.shape[1])
    )
    scale_modes = scale_modes_for_formats(
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        resolved_v_format,
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP4,
        AttentionFormat.MXFP4,
        resolved_v_format,
        *scale_modes,
        softmax_scale=softmax_scale,
        out=out,
        v_pack=resolved_v_pack,
    )


@_launch_mxfp4_coalesced.register_fake
def _launch_mxfp4_coalesced_fake(
    q: Tensor,
    q_descale: Tensor,
    k_data: Tensor,
    k_descale: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    v_format: int,
    v_pack: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_data, k_descale, v_data, v_descale, v_format, v_pack, softmax_scale
    del out


@torch.library.custom_op("aiter::mha_v4_launch_mxfp6_v3", mutates_args=("out",))
def _launch_mxfp6(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    v_pack: int,
    softmax_scale: float,
) -> None:
    resolved_v_format = AttentionFormat(v_format)
    resolved_v_pack = AttentionPack(v_pack)
    k, k_descale = mxfp6_k_view(k_raw, k_descale_raw, q.shape[0], sequence_k, heads)
    v = (
        mxfp4_v_view(v_data, v_descale, sequence_k)
        if resolved_v_format == AttentionFormat.MXFP4
        else v_data
    )
    scale_modes = scale_modes_for_formats(
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        resolved_v_format,
    )
    mha_v4_packed(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        resolved_v_format,
        *scale_modes,
        softmax_scale=softmax_scale,
        out=out,
        v_pack=resolved_v_pack,
    )


@_launch_mxfp6.register_fake
def _launch_mxfp6_fake(
    q: Tensor,
    q_descale: Tensor,
    k_raw: Tensor,
    k_descale_raw: Tensor,
    v_data: Tensor,
    v_descale: Tensor,
    out: Tensor,
    sequence_k: int,
    heads: int,
    v_format: int,
    v_pack: int,
    softmax_scale: float,
) -> None:
    del q, q_descale, k_raw, k_descale_raw, v_data, v_descale
    del sequence_k, heads, v_format, v_pack, softmax_scale
    del out


def _validate_mha_v4_raw_inputs(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Optional[Tensor],  # noqa: UP045
    operation: str,
) -> Tensor:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"{operation} expects BSHD Q, K, and V tensors")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError(f"{operation} currently expects BF16 Q, K, and V inputs")
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError(f"{operation} currently supports head dimension 128 only")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError(f"{operation} currently requires contiguous BSHD inputs")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{operation} requires Q, K, and V with the same batch size")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError(
            f"{operation} requires K and V with matching sequence and head dimensions"
        )
    kv_heads = k.shape[2]
    if kv_heads == 0:
        raise ValueError(f"{operation} requires non-empty KV heads")
    if q.shape[2] % kv_heads != 0:
        raise ValueError(f"{operation} requires query heads to be divisible by KV heads")
    gqa_ratio = q.shape[2] // kv_heads
    if gqa_ratio > 16 or gqa_ratio & (gqa_ratio - 1):
        raise ValueError(f"{operation} supports power-of-two GQA ratios up to 16")
    if out is None:
        return torch.empty_like(q, dtype=torch.bfloat16)
    if out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device:
        raise ValueError("out must match Q's shape/device and have BF16 dtype")
    return out


def mha_v4(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_format: AttentionFormat,
    k_format: AttentionFormat,
    v_format: AttentionFormat,
    softmax_scale: Optional[float] = None,  # noqa: UP045
    out: Optional[Tensor] = None,  # noqa: UP045
    return_lse: bool = False,
    block_mask: Optional[Tensor] = None,  # noqa: UP045
    q_scale_mode: Optional[AttentionScaleMode] = None,  # noqa: UP045
    k_scale_mode: Optional[AttentionScaleMode] = None,  # noqa: UP045
    v_scale_mode: Optional[AttentionScaleMode] = None,  # noqa: UP045
) -> Tensor:
    """Quantize BF16 BSHD operands and run non-causal MHA v4.

    Q and K formats must match. Formats select the canonical quantizers and
    scale modes unless all three scale-mode arguments select another supported
    recipe.
    K and V may have fewer heads than Q for GQA. The Q-to-KV head ratio must
    be a power of two no greater than 16; output retains Q's head count.
    ``block_mask`` is optional boolean tile metadata: ``[B, H, Qtiles, KVtiles]``
    or ``[B, Qtiles, KVtiles]`` (broadcast heads). Geometry is 256x128 on gfx950
    and 256x64 on gfx942. Sparse LUT rows are one per query head; K/V addressing
    uses the GQA ratio. A row may select nothing: an all-False row is a no-op
    that writes a zero output tile.
    """
    if return_lse:
        raise NotImplementedError("MHA v4 kernels do not produce LSE yet")
    out = _validate_mha_v4_raw_inputs(q, k, v, out, "mha_v4")
    scale_modes = _raw_scale_recipe(
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
    )
    q_scale_mode, k_scale_mode, v_scale_mode = scale_modes

    lut_indices: Optional[Tensor] = None  # noqa: UP045
    lut_start: Optional[Tensor] = None  # noqa: UP045
    lut_count: Optional[Tensor] = None  # noqa: UP045
    if block_mask is not None:
        lut_indices, lut_start, lut_count = _block_mask_to_lut(block_mask, q, k)
    packed_lut = {
        "kv_block_indices": lut_indices,
        "lut_start": lut_start,
        "lut_count": lut_count,
    }
    v_pack = AttentionPack.DEFAULT
    if q_format == AttentionFormat.BF16 and v_format == AttentionFormat.BF16:
        return mha_v4_packed(
            q,
            k,
            v,
            q,
            k,
            v,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            **packed_lut,
        )
    if q_format == AttentionFormat.BF16 and _is_fp8_format(v_format):
        v_quantized, v_descale = quantize_fp8(v)
        return mha_v4_packed(
            q,
            k,
            v_quantized,
            q,
            k,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            **packed_lut,
        )
    if scale_modes == _MXFP8_SCALE_MODES:
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = quantize_mxfp8_q(
            q, mha_v4_q_multiplier(softmax_scale)
        )
        k_quantized, k_descale = quantize_mxfp8_k(k)
        v_quantized, v_descale = quantize_fp8(v)
    elif q_format == AttentionFormat.INT8 and _is_fp8_format(v_format):
        q_quantized, q_descale = quantize_int8(q)
        k_quantized, k_descale = quantize_int8(k)
        v_quantized, v_descale = quantize_fp8(v)
    elif q_format in _FP8_FORMATS and v_format in (
        q_format,
        AttentionFormat.MXFP6,
    ):
        q_quantized, q_descale = quantize_fp8_rotated(q)
        k_quantized, k_descale = quantize_fp8_rotated(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_fp8(v)
        elif lut_indices is None:
            v_quantized, v_descale = quantize_v_mxfp6_fp6_p(v)
            v_pack = AttentionPack.V_FOR_FP6_P
        else:
            v_quantized, v_descale = quantize_v_mxfp6(v)
    elif q_format == AttentionFormat.MXFP4 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = quantize_mxfp4_q(q, mha_v4_q_multiplier(softmax_scale))
        k_quantized, k_descale = quantize_mxfp4_k(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_v_fp8(v)
        elif lut_indices is None:
            v_quantized, v_descale = quantize_v_mxfp4_fp6_p(v)
            v_pack = AttentionPack.V_FOR_FP6_P
        else:
            v_quantized, v_descale = quantize_v_mxfp4(v)
        if lut_indices is None:
            _launch_mxfp4_coalesced(
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                out,
                int(v_format),
                int(v_pack),
                softmax_scale,
            )
            return out
        k_view = mxfp4_k_view(k_quantized, k_descale)
        v_view = (
            v_quantized
            if _is_fp8_format(v_format)
            else mxfp4_v_view(v_quantized, v_descale, k.shape[1])
        )
        return mha_v4_packed(
            q_quantized,
            k_view,
            v_view,
            q_descale,
            k_descale,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            v_pack=v_pack,
            **packed_lut,
        )
    elif q_format == AttentionFormat.MXFP6 and v_format in (
        *_FP8_FORMATS,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP4,
    ):
        if softmax_scale is None:
            softmax_scale = 128**-0.5
        q_quantized, q_descale = quantize_mxfp6_q(q, mha_v4_q_multiplier(softmax_scale))
        k_quantized, k_descale = quantize_mxfp6_k(k)
        if _is_fp8_format(v_format):
            v_quantized, v_descale = quantize_v_fp8(v)
        elif v_format == AttentionFormat.MXFP6:
            if lut_indices is not None:
                raise NotImplementedError(
                    "sorted-sparse MXFP6 Q/K/V does not have a kernel row yet"
                )
            v_quantized, v_descale = quantize_v_mxfp6_fp6_p(v)
            v_pack = AttentionPack.V_FOR_FP6_P
        elif lut_indices is None:
            v_quantized, v_descale = quantize_v_mxfp4_fp6_p(v)
            v_pack = AttentionPack.V_FOR_FP6_P
        else:
            v_quantized, v_descale = quantize_v_mxfp4(v)
        if lut_indices is None:
            _launch_mxfp6(
                q_quantized,
                q_descale,
                k_quantized,
                k_descale,
                v_quantized,
                v_descale,
                out,
                k.shape[1],
                k.shape[2],
                int(v_format),
                int(v_pack),
                softmax_scale,
            )
            return out
        k_view, k_descale_view = mxfp6_k_view(
            k_quantized, k_descale, q.shape[0], k.shape[1], k.shape[2]
        )
        v_view = (
            v_quantized
            if v_format != AttentionFormat.MXFP4
            else mxfp4_v_view(v_quantized, v_descale, k.shape[1])
        )
        return mha_v4_packed(
            q_quantized,
            k_view,
            v_view,
            q_descale,
            k_descale_view,
            v_descale,
            q_format,
            k_format,
            v_format,
            q_scale_mode,
            k_scale_mode,
            v_scale_mode,
            softmax_scale=softmax_scale,
            out=out,
            return_lse=return_lse,
            v_pack=v_pack,
            **packed_lut,
        )
    else:
        raise NotImplementedError(
            "raw preprocessing is not implemented yet for "
            f"Q={q_format.name}, K={k_format.name}, V={v_format.name}"
        )

    return mha_v4_packed(
        q_quantized,
        k_quantized,
        v_quantized,
        q_descale,
        k_descale,
        v_descale,
        q_format,
        k_format,
        v_format,
        q_scale_mode,
        k_scale_mode,
        v_scale_mode,
        softmax_scale=softmax_scale,
        out=out,
        return_lse=return_lse,
        v_pack=v_pack,
        **packed_lut,
    )
