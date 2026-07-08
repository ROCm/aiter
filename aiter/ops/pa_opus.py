# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch

from ..jit.core import compile_ops

MD_NAME = "module_pa_opus"


def _gen_pa_fwd_opus_fake(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: int = 1,
):
    if out_ is not None:
        return out_
    return torch.empty_like(Q)


@compile_ops("module_pa_opus", develop=True, fc_name="pa_fwd_opus", gen_fake=_gen_pa_fwd_opus_fake)
def _pa_fwd_opus(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int,
    K_QScale: Optional[torch.Tensor],
    V_QScale: Optional[torch.Tensor],
    out_: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    high_precision: int,
) -> None: ...


def pa_fwd_opus(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables_stride0: int,
    max_qlen: int = 1,
    K_QScale: Optional[torch.Tensor] = None,
    V_QScale: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    qo_indptr: Optional[torch.Tensor] = None,
    high_precision: int = 1,
) -> torch.Tensor:
    """
    Opus-based paged attention decode (sp3 MFMA path, self-contained in opus_pa/).

    API mirrors pa_fwd_asm. Initial support: bf16 Q, fp8 KV, head_dim=128,
    block_size=16, gqa=8|1.
    """
    output = out_ if out_ is not None else torch.empty_like(Q)
    _pa_fwd_opus(
        Q,
        K,
        V,
        block_tables,
        context_lens,
        block_tables_stride0,
        max_qlen,
        K_QScale,
        V_QScale,
        output,
        qo_indptr,
        high_precision,
    )
    return output
