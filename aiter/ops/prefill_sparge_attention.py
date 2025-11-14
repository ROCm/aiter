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

def gen_prefill_sparge_attention_fake_tensors(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tlut: Tensor,
    Tvalid_block_num: Tensor,
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
    pv_threshold: float = 50,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
    is_causal: bool = True
) -> Tensor:
    return out

@compile_ops(
    "module_prefill_sparge_attention",
    fc_name="prefill_sparge_attention",
    gen_fake=gen_prefill_sparge_attention_fake_tensors,
)
def prefill_sparge_attention(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tlut: Tensor,
    Tvalid_block_num: Tensor,
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
    pv_threshold: float = 50,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
    is_causal: bool = True
) -> Tensor: ...

def prefill_sparge_attention_CK(
    TQ: Tensor,
    TK: Tensor,
    TV: Tensor,
    Tlut: Tensor,
    Tvalid_block_num: Tensor,
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
    pv_threshold: float = 50,
    mod: int = 0,
    i_perm: bool = True,
    o_perm: bool = True,
    max_seq_len_q: int = 0,
    max_seq_len_k: int = 0,
    is_causal: bool = True
):
    return prefill_sparge_attention(TQ,TK,TV,Tlut,Tvalid_block_num,out,bias,lse,seqstart_q,seqstart_k,bias_type, batch,nhead,nhead_k,seqlen_q,\
                                    seqlen_k, hdim_q, hdim_v, pv_threshold, mod, i_perm, o_perm, max_seq_len_q, max_seq_len_k, is_causal)
