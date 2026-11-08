# SPDX-License-Identifier: MIT
"""Trace-time TDM and WMMA helpers for gfx1250 GEMM.

All shapes, pitches and offsets are in elements unless explicitly named bytes.
These helpers issue no barriers or waits: each kernel owns its pipeline timing.
"""

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T


def make_tdm_operand(
    pointer, shape, strides, tile_shape, tile_strides, *, num_warps, store=False
):
    """Build a tiled load/store atom and its tile-indexed global coordinates.

    Both schedules use the full input tensor shape for edge zero-fill.
    ``num_warps`` splits each tile into disjoint wave-sized copies.
    """
    view = fx.make_view(pointer, fx.make_layout(shape, strides))
    atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
        rocdl.TensorStore() if store else rocdl.TensorLoad(),
        view,
        fx.make_layout(tile_shape, tile_strides),
        tile_shape,
        num_warps=num_warps,
    )
    return atom, fx.zipped_divide(coords, tile_shape)


def copy_tdm_operand(
    atom,
    shared_pointer,
    coord,
    *,
    tile_shape,
    tile_strides,
    stage_size,
    stages,
    wave,
    num_warps,
    dst_stage,
    src_stage,
):
    """Copy this wave's tile slice into an LDS ring slot.

    Shared by both schedules. ``src_stage`` is an absolute K-tile index
    (including the split-K rank); ``dst_stage`` selects the LDS ring slot.
    The caller issues A/B copies and owns all barriers and wait counts.
    """
    shared_view = fx.make_view(
        shared_pointer,
        fx.make_layout((tile_shape, stages), (tile_strides, stage_size)),
    )
    dst, src = rocdl.cdna5.tdm_partition(
        atom,
        wave,
        fx.make_layout(num_warps, 1),
        shared_view,
        coord,
    )
    fx.copy(atom, src[None, src_stage], dst[None, dst_stage])


class WmmaFragments:
    """Wave32 fragment geometry plus typed and packed LDS load operations.

    Construct inside kernel tracing. BF16/FP16 16x16x32 uses two 128-bit loads (8 i32s per lane).
    Individual chunk loads let the kernel interleave them with WMMA instructions.
    """

    def __init__(self, dtype, mma_m, mma_k):
        self.dtype = dtype
        self.element_bytes = dtype.width // 8
        self.elements = mma_m * mma_k // 32
        self.chunk_elements = 128 // dtype.width
        self.chunk_count = self.elements // self.chunk_elements
        self.k_stride = 2 * self.chunk_elements
        self.packed_i32s = self.elements * dtype.width // 32

    def pointer(
        self,
        shared,
        stage_size,
        stage,
        row,
        column,
        pitch,
        lane,
        *,
        k_contiguous=True,
    ):
        """Pointer to this lane's first element in one WMMA operand fragment."""
        lane_k = lane // 16 * self.chunk_elements
        if k_contiguous:
            return (
                shared
                + stage * stage_size
                + (row + lane % 16) * pitch
                + column
                + lane_k
            )
        return shared + stage * stage_size + row + lane % 16 + (column + lane_k) * pitch

    def load(
        self,
        shared,
        stage_size,
        stage,
        row,
        column,
        pitch,
        lane,
        *,
        k_contiguous=True,
    ):
        """Load a typed fragment. NN B uses scalar strided loads along K."""
        pointer = self.pointer(
            shared,
            stage_size,
            stage,
            row,
            column,
            pitch,
            lane,
            k_contiguous=k_contiguous,
        )
        if k_contiguous:
            chunks = [
                fx.ptr_load(
                    pointer + group * self.k_stride,
                    result_type=T.vec(self.chunk_elements, self.dtype.ir_type),
                )
                for group in range(self.chunk_count)
            ]
            values = fx.Vector.from_elements(
                [
                    chunks[i // self.chunk_elements][i % self.chunk_elements]
                    for i in range(self.elements)
                ]
            )
        else:
            values = fx.Vector.from_elements(
                [
                    fx.ptr_load(
                        pointer
                        + (
                            (i // self.chunk_elements) * self.k_stride
                            + i % self.chunk_elements
                        )
                        * pitch
                    )
                    for i in range(self.elements)
                ]
            )
        fragment = fx.make_rmem_tensor(self.elements, self.dtype)
        fragment.store(values)
        return fragment

    def load_chunk(self, pointer, element_offset):
        """Load one aligned b128 chunk as four i32s; offset remains in elements."""
        layout = fx.make_layout(4, 1)
        atom = fx.make_copy_atom(fx.UniversalCopy(128), fx.Int32)
        pointer_type = fx.PointerType.get(
            elem_ty=fx.Int32.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        address = (
            fx.Int32(fx.ptrtoint(pointer))
            + fx.Int32(element_offset) * self.element_bytes
        )
        aligned = fx.inttoptr(pointer_type, address)
        values = fx.make_rmem_tensor(layout, fx.Int32)
        fx.copy(atom, fx.make_view(aligned, layout), values)
        return values.load()

    def join_chunks(self, parts):
        """Pack the two b128 chunks into the WMMA i32 representation."""
        return parts[0].shuffle(parts[1], list(range(8)))

    def load_packed(self, pointer, element_offset):
        """Load all chunks of an NT fragment (the first-half seed)."""
        return self.join_chunks(
            [
                self.load_chunk(pointer, element_offset + part * self.k_stride)
                for part in range(self.chunk_count)
            ]
        )

    def register_packed(self, values):
        fragment = fx.make_rmem_tensor(self.packed_i32s, fx.Int32)
        fragment.store(values)
        return fragment


def grouped_cluster_swizzle(cluster_id, rank, m_tiles, n_tiles, cm, cn, group_m):
    """Grouped cluster grid, then row-major ranks within each M/N cluster.

    Callers launch ceil(M/cm)*ceil(N/cn)*cm*cn programs and mask edge tiles.
    With split-K only, cm=cn=1; K-rank is not part of cluster_id.
    """
    gm = (m_tiles + cm - 1) // cm
    gn = (n_tiles + cn - 1) // cn
    group = cluster_id // (group_m * gn)
    first = group * group_m
    height = fx.min(group_m, gm - first)
    within = cluster_id % (group_m * gn)
    return (
        (first + within % height) * cm + rank // cn,
        (within // height) * cn + rank % cn,
    )
