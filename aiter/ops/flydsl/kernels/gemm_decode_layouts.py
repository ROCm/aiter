# SPDX-License-Identifier: MIT

"""FlyDSL layouts and tensor views shared by BF16 decode GEMM policies."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx

from aiter.ops.flydsl.kernels import buffer_ops

from .gemm_decode_config import BF16_BYTES, WAVE_SIZE


@dataclass(frozen=True)
class DecodeLayoutGeometry:
    """Compile-time dimensions used to construct decode-kernel layouts."""

    m: int
    n: int
    k: int
    waves: int
    columns_per_wave: int
    staged_k: int = 0

    @property
    def block_threads(self) -> int:
        return self.waves * WAVE_SIZE

    @property
    def columns_per_block(self) -> int:
        return self.waves * self.columns_per_wave


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def make_buffer_matrix(
    tensor,
    rows: int,
    columns: int,
):
    """Give a packed BF16 global operand an exact two-dimensional view."""
    buffer = fx.rocdl.make_buffer_tensor(
        tensor,
        max_size=False,
        num_records_bytes=rows * columns * BF16_BYTES,
    )
    return fx.make_view(
        fx.get_iter(buffer),
        fx.make_layout((rows, columns), (columns, 1)),
    )


def make_shared_matrix(
    storage,
    rows: int,
    columns: int,
    row_stride: int,
):
    """View one shared allocation as a padded row-major matrix."""
    return fx.make_view(
        storage.ptr,
        fx.make_layout((rows, columns), (row_stride, 1)),
    )


def make_vector_view(
    tensor,
    row,
    column,
    row_stride: int,
    width: int,
):
    """Create a contiguous register-width view at a matrix coordinate."""
    # FlyDSL 0.3.0 crd2idx normalizes dynamic coordinates. These coordinates
    # are already proven in bounds, so keep the physical offset explicit at
    # this address-generation boundary.
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    pointer = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element))
    return fx.make_view(pointer, fx.make_layout(width, 1))


def load_vector(
    tensor,
    row,
    column,
    row_stride: int,
    width: int,
    cache_modifier: int = 0,
):
    """Load one logical row vector while preserving explicit cache controls."""
    if cache_modifier:
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))
        element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
        return buffer_ops.buffer_load(
            resource,
            element,
            vec_width=width,
            dtype=tensor.dtype,
            cache_modifier=cache_modifier,
        )
    return make_vector_view(tensor, row, column, row_stride, width).load()


def load_scalar(
    tensor,
    row,
    column,
    row_stride: int,
    cache_modifier: int = 0,
):
    """Load one logical matrix element with optional cache controls."""
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    if cache_modifier:
        resource = fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))
        return buffer_ops.buffer_load(
            resource,
            element,
            vec_width=1,
            dtype=tensor.dtype,
            cache_modifier=cache_modifier,
        )
    pointer = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element))
    return fx.make_view(pointer, fx.make_layout(1, 1))[0]


def store_scalar(
    tensor,
    row,
    column,
    row_stride: int,
    value,
):
    """Store one element at a proven in-bounds matrix coordinate."""
    element = fx.Int32(row) * fx.Int32(row_stride) + fx.Int32(column)
    pointer = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(element))
    fx.make_view(pointer, fx.make_layout(1, 1))[0] = value


def make_wave_lane_layout(waves: int):
    """Map a workgroup thread id to ``(wave, lane)``."""
    return fx.make_layout((waves, WAVE_SIZE), (WAVE_SIZE, 1))


def wave_lane_coordinates(thread_id, waves: int):
    return fx.idx2crd(thread_id, make_wave_lane_layout(waves)).unpack()


def make_n_owner_layout(
    waves: int,
    columns_per_wave: int,
):
    """Map ``(wave, local column)`` to a block-local output column."""
    return fx.make_layout(
        (waves, columns_per_wave),
        (columns_per_wave, 1),
    )


def n_owner_offset(
    wave,
    column,
    waves: int,
    columns_per_wave: int,
):
    return fx.Int32(
        fx.get_scalar(
            fx.crd2idx(
            (wave, column),
                make_n_owner_layout(waves, columns_per_wave),
            )
        )
    )


def make_a_vector_slot_layout(
    rows: int,
    vectors_per_row: int,
):
    """Map a cooperative vector slot to ``(row, row vector)``.

    The physical row mode is power-of-two padded so FlyDSL's coordinate
    normalization lowers to a mask instead of integer remainder. The staging
    predicate limits slots to the logical ``rows`` and leaves padded rows
    unreachable.
    """
    return fx.make_layout(
        (_next_power_of_two(rows), vectors_per_row),
        (vectors_per_row, 1),
    )


def a_vector_slot_coordinates(
    slot,
    rows: int,
    vectors_per_row: int,
):
    return fx.idx2crd(
        slot,
        make_a_vector_slot_layout(rows, vectors_per_row),
    ).unpack()


def make_a_tail_layout(
    rows: int,
    tail_per_row: int,
):
    """Map a cooperative scalar tail slot to a padded ``(row, row tail)``."""
    return fx.make_layout(
        (_next_power_of_two(rows), tail_per_row),
        (tail_per_row, 1),
    )


def a_tail_coordinates(
    slot,
    rows: int,
    tail_per_row: int,
):
    return fx.idx2crd(
        slot,
        make_a_tail_layout(rows, tail_per_row),
    ).unpack()


def make_k_chunk_layout(
    chunks: int,
    width: int,
):
    """Map ``(chunk, lane, value)`` to one logical K element."""
    return fx.make_layout(
        (chunks, WAVE_SIZE, width),
        (WAVE_SIZE * width, width, 1),
    )


def vector_value_offset(
    vector,
    value,
    vectors: int,
    width: int,
):
    """Map a row-local ``(vector, value)`` coordinate to its element."""
    return fx.Int32(
        fx.get_scalar(
            fx.crd2idx(
                (vector, value),
                fx.make_layout((vectors, width), (width, 1)),
            )
        )
    )


def k_element(
    chunk,
    lane,
    value,
    chunks: int,
    width: int,
):
    return fx.Int32(
        fx.get_scalar(
            fx.crd2idx(
            (chunk, lane, value),
                make_k_chunk_layout(chunks, width),
            )
        )
    )
