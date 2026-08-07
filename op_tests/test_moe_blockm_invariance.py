# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe results must not depend on block_m.

block_m is a tile-size / scheduling knob chosen by get_block_size_M or pinned by
a tuned_fmoe.csv row. It must not change what the MoE computes, so the same
inputs run at different block_m have to agree.

This guards a regression where the CK 2-stage kernel silently wrote nothing at
block_m=32 and fused_moe returned an identically zero tensor. It was invisible
end to end because prefill uses larger tiles and stays correct while decode,
which is where the small tiles are selected, quietly lost the entire routed
expert contribution.

Note the failure mode is an all-zero output rather than an exception or NaN, so
a test that only checks for finiteness or for a successful launch will pass. The
assertions below check non-zero output explicitly and compare across tiles.
"""

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter import fused_moe as fm
from aiter.ops.quant import pertoken_quant
from aiter.test_common import checkAllclose

# Tiles the 2-stage heuristic can emit (get_block_size_M support_list).
BLOCK_SIZES = [32, 64, 128]
REFERENCE_BLOCK_M = 128


def _clear_dispatch_caches():
    """Drop memoized dispatch decisions.

    Kernel selection is lru_cached on shape, so without this every iteration
    after the first silently reuses the first block_m and the test passes
    vacuously.
    """
    for name in dir(fm):
        obj = getattr(fm, name, None)
        if hasattr(obj, "cache_clear"):
            obj.cache_clear()


def _run_at_block_m(block_m, hidden, w1, w2, topk_weights, topk_ids, quant_type,
                    w1_scale=None, w2_scale=None):
    orig = fm.get_block_size_M
    try:
        fm.get_block_size_M = lambda *_args, **_kw: block_m
        _clear_dispatch_caches()
        out = fm.fused_moe(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            quant_type=quant_type,
            activation=ActivationType.Silu,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
        return out.float().clone()
    finally:
        fm.get_block_size_M = orig
        _clear_dispatch_caches()


def _build(token, model_dim, inter_dim, E, topk, quantized):
    dev, dt = "cuda", dtypes.bf16
    torch.manual_seed(token)
    hidden = torch.randn(token, model_dim, dtype=dt, device=dev) / 10
    w1 = torch.randn(E, inter_dim * 2, model_dim, dtype=dt, device=dev) / 40
    w2 = torch.randn(E, model_dim, inter_dim, dtype=dt, device=dev) / 40
    topk_weights = torch.softmax(
        torch.randn(token, topk, dtype=dtypes.fp32, device=dev), -1
    )
    topk_ids = torch.randint(0, E, (token, topk), dtype=dtypes.i32, device=dev)

    if not quantized:
        return hidden, w1, w2, topk_weights, topk_ids, QuantType.No, None, None

    w1_q, w1_scale = pertoken_quant(w1, quant_dtype=dtypes.fp8)
    w2_q, w2_scale = pertoken_quant(w2, quant_dtype=dtypes.fp8)
    return (
        hidden, w1_q, w2_q, topk_weights, topk_ids,
        QuantType.per_Token, w1_scale, w2_scale,
    )


# token=1 is the decode regime the regression actually broke; token=64 separates
# "small tile" from "few tokens", since the two are otherwise confounded.
@pytest.mark.parametrize("token", [1, 64])
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "fp8_per_token"])
def test_fused_moe_is_block_m_invariant(token, quantized):
    model_dim, inter_dim, E, topk = 4096, 512, 128, 8
    hidden, w1, w2, tw, tid, qt, w1s, w2s = _build(
        token, model_dim, inter_dim, E, topk, quantized
    )

    ref = _run_at_block_m(REFERENCE_BLOCK_M, hidden, w1, w2, tw, tid, qt, w1s, w2s)
    assert ref.abs().max().item() > 0, (
        f"reference block_m={REFERENCE_BLOCK_M} produced an all-zero output; "
        "the test configuration itself is wrong"
    )

    for block_m in BLOCK_SIZES:
        out = _run_at_block_m(block_m, hidden, w1, w2, tw, tid, qt, w1s, w2s)

        assert out.abs().max().item() > 0, (
            f"fused_moe returned an identically zero tensor at block_m={block_m} "
            f"(token={token}, quantized={quantized}). The kernel silently wrote "
            "nothing; this is the regression this test exists for."
        )
        assert torch.isfinite(out).all(), f"non-finite output at block_m={block_m}"

        # bf16 accumulation order differs per tile, so compare loosely; the bug
        # this guards is a total loss of signal, not a rounding difference.
        checkAllclose(
            ref,
            out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"block_m={block_m} disagrees with block_m={REFERENCE_BLOCK_M}",
        )


if __name__ == "__main__":
    for quantized in (False, True):
        for token in (1, 64):
            test_fused_moe_is_block_m_invariant(token, quantized)
            print(f"PASS token={token} quantized={quantized}")
    aiter.logger.info("fused_moe block_m invariance: all passed")
