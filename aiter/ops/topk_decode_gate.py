# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which decode top-k kernel a call goes to, and the measured tables behind it.

Imports nothing from the kernel modules, so both entry points can import it at
load time; a band re-fit edits this file and no dispatch code.
"""

import functools
import os

import torch

from ..jit.utils.chip_info import get_cu_num_device, get_gfx_runtime

_FLYDSL_TOPK_DECODE_DISABLED = os.environ.get(
    "AITER_DISABLE_FLYDSL_TOPK_DECODE", "0"
) in ("1", "true", "True", "yes", "YES")

# Dispatch uses physical width because seq_lens is device-resident; padded rows
# with short effective lengths may therefore enter a long-row gate.
# (minimum width, maximum width, maximum rows), with inclusive bounds.
_FLYDSL_TOPK_DECODE_GATES = {
    "gfx942": {
        True: (
            (0, 20_000, 64),
            (131_072, None, 16),
        ),
        False: ((524_288, None, 32),),
    },
    "gfx950": {
        True: (
            (0, 20_000, 128),
            (32_768, 65_535, 16),
            (65_536, 131072, 32),
            (131072, 200_000, 48),
            (200_000, None, 128),
        ),
        False: ((131_072, None, 32),),
    },
}
_FLYDSL_TOPK_DECODE_KS = (512, 1024, 2048, 4096)

# What the gate answers. `default` names no kernel: the call keeps its entry
# point's own dispatch, which runs the FlyDSL one-block kernel where that is
# enabled and the C++ one-block kernel elsewhere.
BACKEND_DEFAULT = "default"
BACKEND_SINGLE = "single"
BACKEND_ADAPTIVE = "adaptive"

# Which shapes the adaptive kernel is fastest for. One entry, with names in
# place of the numbers:
#
#   (arch, cu_count): {
#       stable: {
#           (k, ...): (
#               (min_width, max_width, min_rows, max_rows),
#               ...
#           ),
#       },
#   },
#
# Every k in a group shares its bands. Bounds are inclusive, and a max_width of
# None means no upper bound: (4_096, 4_096, 128, 512) admits width 4096 at 128
# to 512 rows. The row minimum is not decoration: one row of a narrow buffer
# goes to a one-block kernel, which has no grid to fill while the adaptive
# kernel pays for having one.
#
# Keyed by CU count as well as arch: one arch name spans several, and the grid
# these were measured against is built from that count. A pair the table does
# not name is routed by the single-launch gate below alone.
#
# A cell is admitted only when this kernel is the fastest of the four decode
# kernels (adaptive, single-launch, FlyDSL one-block, C++ one-block) and at least
# 1.05x faster than the kernel the gate would pick without it, on a full buffer
# and on a padded one bounded at its live length; with the bound at the buffer
# width, as under graph capture, it must not be slower than that kernel. The
# gate cannot tell these apart. The gfx942 304 CU entry was fitted without the
# buffer-width case. Every entry assumes the caller declares `max_row_len`.
_ADAPTIVE_BANDS_BY_K_GROUP = {
    ("gfx942", 80): {
        True: {
            (256,): (
                (4_096, 4_096, 128, 512),
                (8_192, 20_000, 1, 512),
                (32_768, 32_768, 256, 512),
                (65_536, 65_536, 1, 8),
                (131_072, 131_072, 1, 16),
                (262_144, 262_144, 1, 8),
                (524_288, 524_288, 1, 4),
                (1_048_576, 1_048_576, 1, 2),
            ),
            (512, 1024, 2048): (
                (4_096, 4_096, 128, 512),
                (8_192, 20_000, 1, 512),
                (32_768, 32_768, 1, 4),
                (65_536, 65_536, 1, 16),
                (524_288, 524_288, 1, 4),
            ),
            (4096,): (
                (8_192, 20_000, 1, 512),
                (32_768, 65_536, 128, 512),
                (131_072, 262_144, 1, 16),
                (524_288, 524_288, 1, 512),
                (1_048_576, 1_048_576, 1, 16),
            ),
        },
        False: {
            (256,): (
                (4_096, 8_192, 1, 512),
                (16_384, 16_384, 128, 512),
                (20_000, 20_000, 1, 4),
                (32_768, 32_768, 128, 512),
                (65_536, 65_536, 1, 16),
                (131_072, 131_072, 1, 8),
                (262_144, 262_144, 1, 4),
                (524_288, 1_048_576, 1, 2),
            ),
            (512, 1024, 2048): (
                (4_096, 8_192, 1, 512),
                (16_384, 20_000, 128, 512),
                (32_768, 32_768, 1, 8),
                (65_536, 65_536, 1, 16),
                (131_072, 262_144, 1, 8),
                (524_288, 524_288, 1, 4),
                (1_048_576, 1_048_576, 1, 2),
            ),
            (4096,): (
                (8_192, 8_192, 1, 512),
                (16_384, 16_384, 128, 512),
                (20_000, 20_000, 1, 4),
                (32_768, 32_768, 128, 512),
                (524_288, 524_288, 1, 32),
                (1_048_576, 1_048_576, 1, 16),
            ),
        },
    },
    ("gfx942", 228): {
        True: {
            (256,): (
                (4_096, 4_096, 256, 512),
                (8_192, 20_000, 1, 512),
                (65_536, 65_536, 1, 8),
                (131_072, 131_072, 1, 16),
                (262_144, 262_144, 1, 8),
                (524_288, 524_288, 1, 4),
                (1_048_576, 1_048_576, 1, 2),
            ),
            (512, 1024, 2048): (
                (4_096, 4_096, 256, 512),
                (8_192, 20_000, 1, 512),
                (32_768, 32_768, 1, 4),
                (65_536, 65_536, 1, 16),
                (524_288, 524_288, 1, 8),
                (1_048_576, 1_048_576, 2, 4),
            ),
            (4096,): (
                (8_192, 20_000, 1, 512),
                (32_768, 65_536, 256, 512),
                (131_072, 262_144, 1, 16),
                (524_288, 524_288, 1, 512),
                (1_048_576, 1_048_576, 1, 16),
            ),
        },
        False: {
            (256,): (
                (4_096, 8_192, 1, 512),
                (16_384, 16_384, 256, 512),
                (20_000, 20_000, 1, 4),
                (65_536, 65_536, 1, 32),
                (131_072, 131_072, 1, 16),
                (262_144, 1_048_576, 1, 4),
            ),
            (512, 1024, 2048): (
                (4_096, 8_192, 1, 512),
                (16_384, 16_384, 256, 512),
                (32_768, 32_768, 1, 4),
                (65_536, 65_536, 1, 32),
                (131_072, 131_072, 1, 16),
                (262_144, 524_288, 1, 8),
                (1_048_576, 1_048_576, 1, 4),
            ),
            (4096,): (
                (8_192, 8_192, 1, 512),
                (16_384, 20_000, 64, 512),
                (524_288, 524_288, 1, 64),
                (1_048_576, 1_048_576, 1, 16),
            ),
        },
    },
    ("gfx942", 304): {
        True: {
            (256,): (
                (8_192, 8_192, 1, 2),
                (16_384, 20_000, 1, 128),
                (65_536, 65_536, 1, 16),
                (131_072, 131_072, 1, 32),
                (262_144, 262_144, 1, 16),
                (524_288, 524_288, 1, 8),
                (1_048_576, 1_048_576, 1, 4),
            ),
            (512, 1024, 2048): (
                (16_384, 20_000, 1, 128),
                (32_768, 32_768, 1, 2),
                (65_536, 65_536, 1, 16),
                (131_072, 131_072, 1, 32),
                (262_144, 262_144, 1, 16),
                (524_288, 524_288, 1, 8),
                (1_048_576, 1_048_576, 1, 4),
            ),
            (4096,): (
                (16_384, 32_768, 1, 128),
                (65_536, 1_048_576, 1, 512),
            ),
        },
        False: {
            (256,): (
                (65_536, 65_536, 1, 32),
                (131_072, 131_072, 1, 16),
                (262_144, 524_288, 1, 8),
                (1_048_576, 1_048_576, 1, 4),
            ),
            (512, 1024, 2048): (
                (32_768, 32_768, 1, 4),
                (65_536, 131_072, 1, 32),
                (262_144, 524_288, 1, 16),
                (1_048_576, 1_048_576, 1, 4),
            ),
            (4096,): (
                (32_768, 32_768, 1, 16),
                (65_536, 131_072, 1, 64),
                (262_144, 524_288, 1, 128),
                (1_048_576, 1_048_576, 1, 64),
            ),
        },
    },
    ("gfx950", 256): {
        True: {
            (256,): (
                (8_192, 20_000, 1, 512),
                (524_288, 524_288, 1, 8),
            ),
            (512, 1024, 2048): (
                (8_192, 20_000, 1, 512),
                (524_288, 524_288, 1, 8),
            ),
            (4096,): (
                (8_192, 8_192, 1, 128),
                (16_384, 20_000, 1, 512),
                (131_072, 262_144, 1, 32),
                (524_288, 524_288, 1, 512),
                (1_048_576, 1_048_576, 1, 32),
            ),
        },
        False: {
            (256,): (
                (4_096, 4_096, 1, 512),
                (8_192, 8_192, 1, 2),
                (524_288, 524_288, 1, 4),
                (1_048_576, 1_048_576, 1, 2),
            ),
            (512, 1024, 2048): (
                (4_096, 4_096, 1, 512),
                (8_192, 8_192, 1, 2),
                (131_072, 131_072, 1, 8),
                (262_144, 262_144, 1, 4),
                (524_288, 1_048_576, 1, 8),
            ),
            (4096,): (
                (8_192, 8_192, 1, 64),
                (20_000, 20_000, 1, 2),
                (131_072, 131_072, 1, 8),
                (262_144, 262_144, 1, 4),
                (524_288, 524_288, 1, 64),
                (1_048_576, 1_048_576, 1, 16),
            ),
        },
    },
}

_ADAPTIVE_BANDS = {
    device: {
        stable: {k: bands for ks, bands in per_group.items() for k in ks}
        for stable, per_group in per_emit.items()
    }
    for device, per_emit in _ADAPTIVE_BANDS_BY_K_GROUP.items()
}


def _in_bands(bands, width: int, num_rows: int) -> bool:
    return any(
        min_width <= width
        and (max_width is None or width <= max_width)
        and min_rows <= num_rows <= max_rows
        for min_width, max_width, min_rows, max_rows in bands
    )


@functools.lru_cache(maxsize=256)
def _decode_backend(
    arch: str,
    cu_count: int,
    stable: bool,
    width: int,
    num_rows: int,
    k: int,
    indices_only: bool,
    adaptive_width: int | None,
) -> str:
    """Which of the four decode kernels is fastest for this shape.

    The two tables read different lengths and must keep doing so. The adaptive
    bands read `adaptive_width`, the caller's bound, because that kernel is
    configured from it; `None` means no bound was stated and declines them. The
    single-launch bands read the physical `width` they were fitted on, so a
    bound never moves a call between the single-launch kernel and the one-block
    path.

    Both entry points ask this one function rather than deciding again further
    down, which is what keeps them from disagreeing.
    """
    if indices_only and adaptive_width is not None:  # adaptive emits no values
        per_k = _ADAPTIVE_BANDS.get((arch, cu_count), {}).get(stable, {})
        if _in_bands(per_k.get(k, ()), adaptive_width, num_rows):
            return BACKEND_ADAPTIVE
    if (
        arch in _FLYDSL_TOPK_DECODE_GATES
        and k in _FLYDSL_TOPK_DECODE_KS
        and _in_bands(
            (
                (lo, hi, 1, rows)
                for lo, hi, rows in _FLYDSL_TOPK_DECODE_GATES[arch][stable]
            ),
            width,
            num_rows,
        )
    ):
        return BACKEND_SINGLE
    return BACKEND_DEFAULT


def decode_adaptive_width(width: int, max_row_len: int) -> int:
    """The row length the adaptive path should be gated and configured on.

    A decode buffer is allocated once at the model's maximum context, so its
    physical width says what a row *could* hold and `seq_lens` says what it
    does. `seq_lens` is device-resident and this runs on the host, so the only
    way the host learns the live length is for the caller to state a bound.

    Both the band lookup and `decode_adaptive_config` take this, and they must
    take the same value: gating on one length and configuring for another is
    how a band admits a shape that then runs the wrong kernel for it.

    `top_k_per_row_decode` holds the `max_row_len` contract. It is required here:
    the gate declines the adaptive path when the caller states no bound (see
    `decode_backend_for_shape`), so `None` reaching this function is a bug. The
    physical width is not a fallback, because the bands were admitted assuming
    a configuration sized to the bound.
    """
    if max_row_len is None:
        raise ValueError(
            "max_row_len is required to configure the adaptive path; the gate "
            "declines when it is None, so this path should be unreachable"
        )
    if max_row_len < 1:
        raise ValueError(f"max_row_len must be positive, got {max_row_len}")
    return min(width, max_row_len)


def decode_backend_for_shape(
    logits: torch.Tensor,
    seq_lens: torch.Tensor | None,
    num_rows: int,
    k: int,
    stable: bool,
    values: torch.Tensor | None = None,
    max_row_len: int | None = None,
    *,
    width: int | None = None,
) -> str:
    """The backend for this call's shape, before the FlyDSL tensor contract.

    `width` is the plane width packed rows stand for, and defaults to the
    logits' own. `topk.decode_backend_for_call` adds the tensor contract.
    """
    if (
        _FLYDSL_TOPK_DECODE_DISABLED
        or not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
    ):
        return BACKEND_DEFAULT

    width = logits.shape[1] if width is None else width
    return _decode_backend(
        get_gfx_runtime(),
        get_cu_num_device(logits.device.index),
        stable,
        width,
        num_rows,
        k,
        values is None,
        # No bound means the host cannot size the adaptive config, and no
        # `seq_lens` leaves the adaptive kernel nothing to read row lengths from,
        # so either declines it; the single-launch bands still see the width.
        # A bound below 1 is no bound: it holds only for a batch of empty rows.
        (
            None
            if max_row_len is None or max_row_len < 1 or seq_lens is None
            else decode_adaptive_width(width, max_row_len)
        ),
    )
