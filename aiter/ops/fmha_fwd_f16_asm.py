# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python stub for the ASM FMHA-forward (BF16) op.

The real implementation lives in C++ (`csrc/py_itfs_cu/asm_fmha_fwd_f16.cu`)
and is exposed through the pybind module ``module_fmha_fwd_f16_asm``.
"""

from typing import List, Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops


def _shape_from_perm(perm: int, batch: int, heads: int, seqlen: int, dim: int):
    """Return the expected tensor shape for the given perm code."""
    if perm == 0:   # bshd
        return (batch, seqlen, heads, dim)
    elif perm == 1: # bhsd
        return (batch, heads, seqlen, dim)
    else:           # sbhd
        return (seqlen, batch, heads, dim)


def _dims_from_perm(t: Tensor, perm: int):
    """Extract (batch, heads, seqlen, dim) from tensor shape given perm."""
    if perm == 0:   # bshd [b,s,h,d]
        return t.size(0), t.size(2), t.size(1), t.size(3)
    elif perm == 1: # bhsd [b,h,s,d]
        return t.size(0), t.size(1), t.size(2), t.size(3)
    else:           # sbhd [s,b,h,d]
        return t.size(1), t.size(2), t.size(0), t.size(3)


def gen_fmha_fwd_f16_asm_fake_tensors(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
    i_perm: int = 2,
    o_perm: int = 0,
    sink: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> List[Tensor]:
    batch, q_head_num, q_seq_len, _ = _dims_from_perm(q, i_perm)
    _, _, _, d_v                    = _dims_from_perm(v, i_perm)

    fake_out_shape = _shape_from_perm(o_perm, batch, q_head_num, q_seq_len, d_v)
    fake_out = (
        out if out is not None
        else torch.empty(fake_out_shape, dtype=q.dtype, device=q.device)
    )
    if return_lse:
        fake_lse = torch.empty(
            (batch, q_head_num, q_seq_len), dtype=torch.float32, device=q.device
        )
        return [fake_out, fake_lse]
    return [fake_out]


@compile_ops(
    "module_fmha_fwd_f16_asm",
    fc_name="fmha_fwd_f16_asm",
    gen_fake=gen_fmha_fwd_f16_asm_fake_tensors,
)
def fmha_fwd_f16_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
    i_perm: int = 2,
    o_perm: int = 0,
    sink: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> List[Tensor]: ...
