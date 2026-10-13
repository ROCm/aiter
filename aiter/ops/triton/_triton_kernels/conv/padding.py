"""Standalone input padding and its inline coordinate-mapping helper."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.conv_config_utils import get_conv_config


def _get_config(M=None):
    return get_conv_config("CONV-PADDING", M=M)


@triton.jit
def _map_padding_coordinate(
    coordinates,
    size: tl.constexpr,
    padding,
    PADDING_MODE: tl.constexpr = "zeros",
):
    """Return input coordinates and a validity mask for one spatial axis.

    Requires positive size and host-validated padding. Nonzero modes extend
    only the finite domain [-padding, size + padding). Invalid coordinates
    need not map in bounds; callers must retain this mask and combine it
    with their batch, channel, and tile masks when loading.
    """
    tl.static_assert(size > 0, "input size must be positive")

    if PADDING_MODE == "zeros":
        return coordinates, (coordinates >= 0) & (coordinates < size)
    else:
        valid = (coordinates >= -padding) & (coordinates < size + padding)
        if PADDING_MODE == "replicate":
            mapped = tl.minimum(tl.maximum(coordinates, 0), size - 1)
        elif PADDING_MODE == "reflect":
            folded = tl.abs(coordinates)
            mapped = tl.where(folded < size, folded, 2 * (size - 1) - folded)
        else:
            tl.static_assert(PADDING_MODE == "circular", "unsupported padding mode")
            mapped = tl.where(
                coordinates < 0,
                coordinates + size,
                tl.where(coordinates >= size, coordinates - size, coordinates),
            )
        return mapped, valid


_pad_conv2d_kernel_repr = make_kernel_repr(
    "_pad_conv2d_kernel",
    ["BLOCK_SIZE", "PADDING_MODE", "CHANNELS_LAST"],
)


@triton.jit(repr=_pad_conv2d_kernel_repr)
def _pad_conv2d_kernel(
    source,
    output,
    output_numel,
    channels: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    pad_height: tl.constexpr,
    pad_width: tl.constexpr,
    source_batch_stride: tl.constexpr,
    source_channel_stride: tl.constexpr,
    source_row_stride: tl.constexpr,
    source_column_stride: tl.constexpr,
    PADDING_MODE: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Gather the full padded tensor in contiguous physical output order."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_height = height + 2 * pad_height
    output_width = width + 2 * pad_width
    if CHANNELS_LAST:
        channel_indices = offsets % channels
        column_indices = offsets // channels % output_width
        row_indices = offsets // (channels * output_width) % output_height
    else:
        column_indices = offsets % output_width
        row_indices = offsets // output_width % output_height
        channel_indices = offsets // (output_height * output_width) % channels
    batch_indices = offsets // (channels * output_height * output_width)

    source_rows, rows_valid = _map_padding_coordinate(
        row_indices - pad_height, height, pad_height, PADDING_MODE
    )
    source_columns, columns_valid = _map_padding_coordinate(
        column_indices - pad_width, width, pad_width, PADDING_MODE
    )
    source_offsets = (
        batch_indices * source_batch_stride
        + channel_indices * source_channel_stride
        + source_rows * source_row_stride
        + source_columns * source_column_stride
    )
    output_valid = offsets < output_numel
    values = tl.load(
        source + source_offsets,
        mask=output_valid & rows_valid & columns_valid,
        other=0.0,
    )
    tl.store(output + offsets, values, mask=output_valid)
