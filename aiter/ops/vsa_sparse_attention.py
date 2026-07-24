# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops


def _vsa_sparse_attention_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
) -> Tensor:
    del k, v, block_lut, block_counts
    return torch.empty_like(q)


@compile_ops(
    "module_vsa_sparse_attention",
    fc_name="vsa_sparse_attention",
    gen_fake=_vsa_sparse_attention_fake,
)
def vsa_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_lut: Tensor,
    block_counts: Tensor,
) -> Tensor:
    """Run VSA block-sparse attention on contiguous BHSD tensors.

    ``block_lut`` stores one absolute K-block index followed by delta-encoded
    indices for each 128-token Q block. ``block_counts`` gives the number of
    active entries in each LUT row. The final LUT slot is reserved as a
    lookahead sentinel by the current CK pipeline.
    """
