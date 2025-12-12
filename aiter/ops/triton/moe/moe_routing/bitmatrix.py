import torch
import triton
from typing import Type
from aiter.ops.triton._triton_kernels.moe.moe_routing.bitmatrix import (
    _sum_bitmatrix_memset,
    _sum_bitmatrix_rows,
)
from dataclasses import dataclass, fields


@dataclass
class Bitmatrix:
    """
    Represents a boolean matrix in a packed format where each element occupies
    a single bit of memory.

    _scratchpad is either None or an all-zero array of size >= shape[-1]; we pass it along
    with the actual bitmatrix to avoid having to launch a separate memset
    kernel when we call Bitmatrix::sum().
    """

    scratchpad: torch.Tensor = None

    def __init__(self, data, shape, scratchpad=None, scratchpad_partials=None):
        self.data = data
        self.shape = shape
        self.device = data.device
        self.scratchpad = scratchpad
        self.scratchpad_partials = scratchpad_partials

    def sum(self, partials_block_size):
        _, n_cols = self.shape
        dev = self.device
        if self.scratchpad is None:
            self.scratchpad = clear_sums(n_cols, dev)
        out_ret = self.scratchpad[:n_cols]
        self.scratchpad = None  # throw error if we try to sum again
        return sum_bitmatrix_rows(self, out_ret, partials_block_size)


def clear_sums(n_cols, device, MEMSET_BLOCK=512):
    cdiv = triton.cdiv
    blocks = cdiv(n_cols, MEMSET_BLOCK)
    out_ret = torch.empty((blocks * MEMSET_BLOCK,), device=device, dtype=torch.int32)
    _sum_bitmatrix_memset[(blocks,)](out_ret, MEMSET_BLOCK)
    return out_ret


def sum_bitmatrix_rows(x, out_ret, partials_block_size=None):
    assert partials_block_size is not None
    cdiv = triton.cdiv
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    assert out_ret.shape == (n_cols,)

    TILE_SIZE = 8
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    pids_x = cdiv(n_rows, BLOCK_MM)
    pids_y = cdiv(n_cols, 32)
    out_partials = x.scratchpad_partials

    # output tensors
    _sum_bitmatrix_rows[(pids_x, pids_y)](
        x.data,
        n_rows,
        x.data.stride(0),
        x.data.stride(1),  # input
        out_ret,  # output [final reduction]
        out_partials,
        out_partials.stride(0),
        out_partials.stride(1),
        out_partials.shape[1],
        pids_x,  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M,
        BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8,
    )

    out_partials = out_partials[: cdiv(n_rows, PARTIALS_BLOCK_M), :]

    return out_ret, out_partials
