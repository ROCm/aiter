import torch
from typing import Type
from .reduction_details.reduce_bitmatrix import clear_sums, sum_bitmatrix_rows
from dataclasses import dataclass, fields
from triton.tools.tensor_descriptor import TensorDescriptor


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
