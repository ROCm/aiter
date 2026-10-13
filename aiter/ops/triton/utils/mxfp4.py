# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.utils._triton.mxfp4 import (
    MXFP4_SCALE_KCHUNK,
    MXFP4_SCALE_STRIPE,
    MXFP4_SHUFFLE_GROUP_BYTES,
    MXFP4_SHUFFLE_TILE_ROWS,
)

_SIGN_VALIDATED_VERSION = "_aiter_h16_sign_validated_version"


def mxfp4_scale_swizzle_supported(rows: int, cols: int) -> bool:
    """Return whether a scale matrix exactly tiles the gfx950 layout."""
    return rows % MXFP4_SCALE_STRIPE == 0 and cols % MXFP4_SCALE_KCHUNK == 0


def mxfp4_data_shuffle_supported(rows: int, packed_cols: int) -> bool:
    """Return whether packed FP4 data exactly tiles the gfx950 B layout."""
    return (
        rows % MXFP4_SHUFFLE_TILE_ROWS == 0
        and packed_cols % MXFP4_SHUFFLE_GROUP_BYTES == 0
    )


def prepare_h16_sign(
    sign_vector: torch.Tensor,
    reference: torch.Tensor,
    size: int = 16,
) -> torch.Tensor:
    """Validate and flatten a same-device vector of H16 signs."""
    if not isinstance(sign_vector, torch.Tensor):
        raise TypeError("sign_vector must be a torch.Tensor")
    if sign_vector.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            "sign_vector must have dtype torch.bfloat16 or torch.float32, "
            f"got {sign_vector.dtype}"
        )
    if sign_vector.device != reference.device:
        raise ValueError(
            "sign_vector must be on the same device as the input, "
            f"got {sign_vector.device} and {reference.device}"
        )
    if sign_vector.numel() != size:
        raise ValueError(f"sign_vector must contain {size} elements")

    try:
        version = sign_vector._version
    except (AttributeError, RuntimeError):
        version = None
    if (
        version is None
        or getattr(sign_vector, _SIGN_VALIDATED_VERSION, None) != version
    ):
        valid = torch.all(
            torch.isfinite(sign_vector) & ((sign_vector == 1) | (sign_vector == -1))
        )
        message = "sign_vector entries must be finite and equal to +1 or -1"
        capturing = False
        if reference.device.type == "cuda":
            with torch.cuda.device(reference.device):
                capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            assert_async = getattr(torch, "_assert_async", None)
            if assert_async is None:
                raise RuntimeError("sign_vector must be validated before graph capture")
            assert_async(valid, message)
        elif not valid.item():
            raise ValueError(message)
        if version is not None:
            setattr(sign_vector, _SIGN_VALIDATED_VERSION, version)

    return sign_vector.reshape(-1).contiguous()
