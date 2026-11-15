# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Atomic gemm2 must honour the sort's valid token count on a padded M.

A LocalToken moe_sorting (``num_local_tokens``) fills each expert block with the
sentinel ``MOE_SORTING_MOCK_ID(tokens, topk)`` whose token id is the REAL count.
When the caller's M is padded (cudagraph), that id is still < M, so an atomic
epilog bounded by ``token_id < M`` scatters every pad slot into row ``tokens``.
``use_valid_token_count`` bounds rows by ``min(cumsum[1], M)`` instead.

The weights and activations are random bytes: only which rows get written is
under test, and each valid row takes exactly one slot so on/off are bit-equal.

Run:
    pytest op_tests/flydsl_tests/test_gemm2_valid_token_count.py -q
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="gfx950 + FlyDSL required"
)

M_PAD, TOKENS, NE, BM, N, K = 128, 100, 4, 32, 512, 256


def _sorted_buffers():
    """Each token goes to expert t % NE (topk 1); blocks padded with the sentinel."""
    ids, eids = [], []
    for e in range(NE):
        toks = list(range(e, TOKENS, NE))
        for b in range(0, len(toks), BM):
            blk = toks[b : b + BM]
            ids += blk + [TOKENS | (1 << 24)] * (BM - len(blk))
            eids.append(e)
    dev = "cuda"
    sorted_ids = torch.tensor(ids, dtype=torch.int32, device=dev)
    cumsum = torch.tensor([len(ids), TOKENS], dtype=torch.int32, device=dev)
    return (
        sorted_ids,
        torch.tensor(eids, dtype=torch.int32, device=dev),
        cumsum,
        torch.ones(len(ids), dtype=torch.float32, device=dev),
    )


def _check(run):
    off, on = run(False), run(True)
    assert off[TOKENS].abs().sum() > 0, "the pad pileup did not reproduce"
    assert torch.equal(on[:TOKENS], off[:TOKENS])
    assert not on[TOKENS:].any(), "a pad slot was scattered into a pad row"


def test_a16w4_gemm2_honours_valid_token_count():
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm2

    sorted_ids, eids, cumsum, weights = _sorted_buffers()
    torch.manual_seed(0)
    a = torch.randn(sorted_ids.numel(), K, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (NE, N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.full((NE * N * K // 32,), 127, dtype=torch.uint8, device="cuda")

    def run(flag):
        out = torch.zeros(M_PAD, N, dtype=torch.bfloat16, device="cuda")
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=a,
            w2_u8=w,
            w2_scale_u8=scale,
            sorted_expert_ids=eids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_ids,
            sorted_weights=weights,
            flat_out=out.view(-1),
            M_logical=M_PAD,
            max_sorted=sorted_ids.numel(),
            NE=NE,
            D_HIDDEN=N,
            D_INTER=K,
            topk=1,
            tile_m=BM,
            use_valid_token_count=flag,
        )
        torch.cuda.synchronize()
        return out

    _check(run)


def test_mxfp4_gemm2_honours_valid_token_count():
    from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2

    sorted_ids, eids, cumsum, weights = _sorted_buffers()
    torch.manual_seed(0)
    rows = sorted_ids.numel()
    aq = torch.randint(0, 256, (rows, K // 2), dtype=torch.uint8, device="cuda")
    ascale = torch.full((rows * K // 32,), 127, dtype=torch.uint8, device="cuda")
    w = torch.randint(0, 256, (NE, N, K // 2), dtype=torch.uint8, device="cuda")
    wscale = torch.full((NE * N * K // 32,), 127, dtype=torch.uint8, device="cuda")

    def run(flag):
        out = torch.zeros(M_PAD, N, dtype=torch.bfloat16, device="cuda")
        flydsl_mxfp4_gemm2(
            inter_sorted_quant=aq,
            inter_sorted_shuffled_scale=ascale,
            w2_u8=w,
            w2_scale_u8=wscale,
            sorted_expert_ids=eids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_ids,
            sorted_weights=weights,
            flat_out=out,
            M_logical=M_PAD,
            max_sorted=rows,
            BM=BM,
            use_nt=False,
            atomic=True,
            mxfp4out=False,
            NE=NE,
            D_HIDDEN=N,
            D_INTER=K,
            topk=1,
            use_valid_token_count=flag,
        )
        torch.cuda.synchronize()
        return out

    _check(run)
