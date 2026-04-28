# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise ``fused_moe`` with ``AITER_PYHIP_HSACO_MOE=1`` on gfx942 (``hsaco_tools.get_kernel``).

Requires: ``model_dim=4096``, ``inter_dim=128``, ``topk=10``, FP8 weights, ``doweight_stage1=False``.

``.co`` artifacts (trimmed basenames; copy under ``hsa/gfx942/fmoe_pyhip/`` — see
``pyhip/tests/contrib/moe/MOE_PYHIP_KERNELS.md`` and ``aiter/fused_moe_pyhip_hsaco.py``):

**B = 1:** if both ``moe_gemm_batch1`` ``.co`` (SiLU True + False) are present under
``hsa/gfx942/fmoe_pyhip/``, ``fused_moe`` uses the pyhip ``entry_b1`` / ``test_small_batch_perf``
path (no ``moe_sorting``). Otherwise it falls back to the two-kernel path below.

**B ≥ 1 (fallback for B=1, or all B>1):** pyhip ``16x32_2s_b`` — ``moe_sorting`` +:

1. ``moe_gemm_batch-1-weight_dtype=torch.float8_e4m3fnuz-with_silu=True.co`` (``moe_gemm_batch``).
2. ``moe_2stage_splitk-1-weight_dtype=torch.float8_e4m3fnuz-TOPK=10-K=128-N=4096-with_silu=False-BLOCK_TILE_SIZE_M=16-BLOCK_TILE_SIZE_N=64-fp8_ptpc=True.co`` (``moe_2stage_splitk``).
"""

import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes, get_gfx
from aiter import pertoken_quant
from aiter.fused_moe import fused_moe, fused_topk, torch_moe
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

# Batch sizes aligned with pyhip ``test_small_batch_perf`` / ``entry_common`` loop (plus 10).
TOKEN_SIZES = (1, 2, 4, 8, 10, 12, 16, 32)

MODEL_DIM = 4096
INTER_DIM = 128
E = 512
TOPK = 10


@pytest.mark.parametrize("token", TOKEN_SIZES)
@pytest.mark.skipif(get_gfx() != "gfx942", reason="pyhip bundle is built for gfx942")
def test_fused_moe_pyhip_hsaco_gfx942_tokens(token: int):
    """Same two ``.co`` files for all ``token`` (``moe_gemm_batch`` + ``moe_2stage_splitk``)."""
    os.environ["AITER_PYHIP_HSACO_MOE"] = "1"
    dtype = dtypes.bf16
    device = "cuda"

    hidden = torch.randn(token, MODEL_DIM, dtype=dtype, device=device) * 0.01
    w1_bf16 = torch.randn(E, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=device) / 10.0
    w2_bf16 = torch.randn(E, MODEL_DIM, INTER_DIM, dtype=dtype, device=device) / 10.0
    score = torch.randn(token, E, device=device, dtype=dtype)
    topk_w, topk_ids = fused_topk(hidden, score, TOPK, True)

    w1, fc1_scale = pertoken_quant(w1_bf16, quant_dtype=dtypes.fp8)
    w2, fc2_scale = pertoken_quant(w2_bf16, quant_dtype=dtypes.fp8)
    w1 = shuffle_weight(w1, layout=(16, 16))
    w2 = shuffle_weight(w2, layout=(16, 16))

    ref = torch_moe(
        hidden,
        w1,
        w2,
        topk_w,
        topk_ids,
        fc1_scale,
        fc2_scale,
        None,
        None,
        activation=ActivationType.Silu,
    )

    out = fused_moe(
        hidden,
        w1,
        w2,
        topk_w,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Token,
        doweight_stage1=False,
        w1_scale=fc1_scale,
        w2_scale=fc2_scale,
    )

    err_frac = checkAllclose(
        ref,
        out,
        rtol=0.05,
        atol=0.05,
        msg=f"pyhip hsaco fused_moe vs torch_moe (token={token})",
    )
    assert err_frac == 0, (
        f"fused_moe (HSACO) vs torch_moe token={token}: {err_frac:.2%} of elements not close "
        f"(NaN in out often means bad kernel args or scale layout)"
    )


if __name__ == "__main__":
    #test_fused_moe_pyhip_hsaco_gfx942_tokens(1)
    for token in TOKEN_SIZES:
        test_fused_moe_pyhip_hsaco_gfx942_tokens(token)
