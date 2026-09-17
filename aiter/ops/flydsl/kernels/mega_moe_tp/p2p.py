# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared P2P primitives for the fused TP MoE kernels.

The peers live in a :class:`~.symmetric_arena.SymmetricArena`, so a
region has the same byte offset on every rank and a peer address is just
``base_ptrs[peer] + region_offset``.  ``base_ptrs`` reaches the kernel through a
small int64 descriptor tensor rather than the fixed 4 GiB VA stride the mori
windows use, because HIP IPC hands back arbitrary addresses.
"""

import flydsl.expr as fx
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import global_typed_ptr
from ..tensor_shim import ptr_buf_tensor

__all__ = [
    "DESC_ARRIVE",
    "DESC_FLAGS",
    "DESC_PEER_BASE",
    "desc_peer_base",
    "desc_ptr",
    "desc_region_offset",
    "desc_region_src",
    "desc_size",
    "desc_slot",
    "flat_buffer",
    "load_words",
    "store_words",
]

# ---------------------------------------------------------------------------
# Descriptor layout (int64 entries), built once on the host.
#
#   0                      local arrive-counter address (i32)
#   1                      byte offset of the [world_size] i32 flag array
#   2 .. 2+TP-1            peer arena base addresses
#   2+TP + 2*r             source address of region r (rank-local)
#   2+TP + 2*r + 1         byte offset of region r inside the arena
# ---------------------------------------------------------------------------
DESC_ARRIVE = 0
DESC_FLAGS = 1
DESC_PEER_BASE = 2


def desc_size(tp_size: int, regions: int) -> int:
    return DESC_PEER_BASE + tp_size + 2 * regions


def desc_region_src(tp_size: int, region: int) -> int:
    return DESC_PEER_BASE + tp_size + 2 * region


def desc_region_offset(tp_size: int, region: int) -> int:
    return DESC_PEER_BASE + tp_size + 2 * region + 1


def desc_ptr(desc_addr):
    """Element-indexable int64 view of the descriptor."""
    return global_typed_ptr(desc_addr, T.i64, align=8)


def desc_slot(desc_addr, index):
    """Read one int64 slot; ``index`` may be a Python int or a runtime i32."""
    return fx.Int64(desc_ptr(desc_addr)[fx.Int32(index)])


def desc_peer_base(desc_addr, peer):
    """Base address of ``peer``'s arena."""
    return fx.Int64(desc_ptr(desc_addr)[fx.Int32(DESC_PEER_BASE) + fx.Int32(peer)])


def flat_buffer(addr, elem, num_records_bytes):
    """Bounded flat V# view over a raw global address.

    The hardware ``num_records`` bound is what keeps a grid-stride tail from
    running off the end of a region, so it is always passed explicitly.
    """
    return ptr_buf_tensor(
        global_typed_ptr(addr, elem.ir_type, align=max(1, elem.width // 8)),
        elem,
        num_records_bytes=num_records_bytes,
    )


def _copy_atom(elem, width, cache_modifier):
    return fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem.width * width, cache_modifier), elem
    )


def load_words(buffer, index, *, width, cache_modifier=0):
    """Load ``width`` int32 words starting at word ``index * width``."""
    fragment = fx.make_rmem_tensor(width, fx.Int32)
    fx.copy(
        _copy_atom(fx.Int32, width, cache_modifier),
        fx.slice(fx.logical_divide(buffer, fx.make_layout(width, 1)), (None, index)),
        fragment,
    )
    return fx.Vector(fx.memref_load_vec(fragment))


def store_words(buffer, index, value, *, width, cache_modifier=0):
    """Store ``width`` int32 words starting at word ``index * width``."""
    fragment = fx.make_rmem_tensor(width, fx.Int32)
    fx.memref_store_vec(value, fragment)
    fx.copy(
        _copy_atom(fx.Int32, width, cache_modifier),
        fragment,
        fx.slice(fx.logical_divide(buffer, fx.make_layout(width, 1)), (None, index)),
    )
