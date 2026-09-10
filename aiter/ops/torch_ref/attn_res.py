# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-PyTorch reference for ``flydsl_attn_res``.

Mirrors Kimi-K3 Attention Residuals: an optional BF16-rounded prefix update, a
depth-axis softmax whose scores use RMS-normalized sources and whose values
remain raw, plus optional output RMSNorm.

Scoring, mixing, and output RMSNorm accumulate in ``dtype`` (default FP32).
Only the prefix-plus-delta update rounds through BF16. The output stays in
``dtype``; the caller decides whether to round it to BF16.

This is a correctness reference, not an inference path. It returns the
post-delta prefix without mutating inputs. Snapshot writes are checked
test-side by copying that returned prefix into the expected block slot, so this
module intentionally does not take ``block_write_idx`` or mutate ``blocks``.
"""

import torch
import torch.nn.functional as F


def attn_res(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    eps: float,
    output_norm_eps: float,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the AttnRes output (in ``dtype``) and post-delta prefix."""
    hidden_size = prefix.shape[-1]
    if delta is not None:
        prefix = (prefix.to(torch.float32) + delta.to(torch.float32)).to(torch.bfloat16)

    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1).to(dtype)
    keys = F.rms_norm(values, (hidden_size,), norm_weight.to(dtype), eps)
    probs = (keys @ qk_weight.to(dtype)).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)

    if output_norm_weight is not None:
        output = F.rms_norm(
            output,
            (hidden_size,),
            output_norm_weight.to(dtype),
            output_norm_eps,
        )

    return output, prefix
