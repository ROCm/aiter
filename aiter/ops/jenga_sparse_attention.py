# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
import functools
import pandas as pd
from ..jit.core import (
    compile_ops,
    AITER_ROOT_DIR,
)
from ..utility import dtypes
from ..jit.utils.chip_info import get_cu_num

def gen_jenga_sparse_attention_fake_tensors(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tblock_relation_onehot: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    seqstart_q: Optional[Tensor] = None,
    seqstart_k: Optional[Tensor] = None,
    bias_type: int = 0,
    batch: int = 0,
    nhead: int = 0,
    nhead_k: int = 0,
    seqlen_q: int = 0,
    seqlen_k: int = 0,
    hdim_q: int = 0,
    hdim_v: int = 0,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0
) -> Tensor:
    return out

@compile_ops(
    "module_jenga_sparse_attention",
    fc_name="jenga_sparse_attention",
    gen_fake=gen_jenga_sparse_attention_fake_tensors,
)
def jenga_sparse_attention(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tblock_relation_onehot: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    seqstart_q: Optional[Tensor] = None,
    seqstart_k: Optional[Tensor] = None,
    bias_type: int = 0,
    batch: int = 0,
    nhead: int = 0,
    nhead_k: int = 0,
    seqlen_q: int = 0,
    seqlen_k: int = 0,
    hdim_q: int = 0,
    hdim_v: int = 0,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0
) -> Tensor: ...

def gen_vsa_sparse_attention_fake_tensors(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tkv_block_idx: Tensor,
    Tkv_blocks: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    seqstart_q: Optional[Tensor] = None,
    seqstart_k: Optional[Tensor] = None,
    bias_type: int = 0,
    batch: int = 0,
    nhead: int = 0,
    nhead_k: int = 0,
    seqlen_q: int = 0,
    seqlen_k: int = 0,
    hdim_q: int = 0,
    hdim_v: int = 0,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0
) -> Tensor:
    return out

@compile_ops(
    "module_vsa_sparse_attention",
    fc_name="vsa_sparse_attention",
    gen_fake=gen_vsa_sparse_attention_fake_tensors,
)
def vsa_sparse_attention(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tkv_block_idx: Tensor,
    Tkv_blocks: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    seqstart_q: Optional[Tensor] = None,
    seqstart_k: Optional[Tensor] = None,
    bias_type: int = 0,
    batch: int = 0,
    nhead: int = 0,
    nhead_k: int = 0,
    seqlen_q: int = 0,
    seqlen_k: int = 0,
    hdim_q: int = 0,
    hdim_v: int = 0,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0
) -> Tensor: ...

def jenga_sparse_attention_CK(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tblock_relation_onehot: Tensor,
    Tkv_block_idx: Tensor,
    Tkv_blocks: Tensor,
    out: Tensor,
    bias: Optional[Tensor] = None,
    lse: Optional[Tensor] = None,
    seqstart_q: Optional[Tensor] = None,
    seqstart_k: Optional[Tensor] = None,
    bias_type: int = 0,
    batch: int = 0,
    nhead: int = 0,
    nhead_k: int = 0,
    seqlen_q: int = 0,
    seqlen_k: int = 0,
    hdim_q: int = 0,
    hdim_v: int = 0,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
):
    if Tkv_block_idx is None:
        return jenga_sparse_attention(TQ,TK,TV,Tblock_relation_onehot,out,bias,lse,seqstart_q,seqstart_k,bias_type, batch,nhead,nhead_k,seqlen_q,\
                                        seqlen_k, hdim_q, hdim_v, mod, i_perm, o_perm, max_seq_len_q, max_seq_len_k)
    else:
        return vsa_sparse_attention(TQ,TK,TV,Tkv_block_idx,Tkv_blocks, out,bias,lse,seqstart_q,seqstart_k,bias_type, batch,nhead,nhead_k,seqlen_q,\
                                        seqlen_k, hdim_q, hdim_v, mod, i_perm, o_perm, max_seq_len_q, max_seq_len_k)
