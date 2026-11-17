# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Atomic gemm2 must honour the sort's valid token count on a padded M.

A LocalToken moe_sorting (``num_local_tokens``) fills each expert block with the
sentinel ``MOE_SORTING_MOCK_ID(tokens, topk)`` whose token id is the REAL count.
When the caller's M is padded (cudagraph), that id is still < M, so an atomic
epilog bounded by ``token_id < M`` scatters every pad slot into row ``tokens``.
``use_valid_token_count`` bounds rows by ``min(cumsum[1], M)`` instead.

The reduce epilog has no row bound and must reject the sentinel by its slot:
with one pad row its ``token * topk + slot`` row is one past its buffer.

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


def test_a16w4_gemm2_reduce_skips_pad_slots_at_one_pad_row():
    """The reduce epilog writes unique ``[token * topk + slot, N]`` rows and
    must skip the sort's pad slots. A LocalToken sort's pad slot has token ==
    the real count and slot == topk, so with one pad row (real count M - 1)
    its row is ``M * topk``: one past the buffer. ``token_id < M`` does not
    reject it. At production shapes the buffer can end on an allocation
    boundary (Kimi-K3: ``M * 16 * 3584`` bf16 is a whole number of 2 MiB pages
    for every M that is a multiple of 128), so the stray row faults or lands
    in another tensor. A canary row past the buffer catches it here."""
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm2

    m_pad = TOKENS + 1  # one pad row
    sorted_ids, eids, cumsum, weights = _sorted_buffers()
    torch.manual_seed(0)
    a = torch.randn(sorted_ids.numel(), K, dtype=torch.bfloat16, device="cuda")
    w = torch.randint(0, 256, (NE, N, K // 2), dtype=torch.uint8, device="cuda")
    scale = torch.full((NE * N * K // 32,), 127, dtype=torch.uint8, device="cuda")

    def run(epilog):
        # The kernel owns the first m_pad * topk rows; the last is the canary.
        out = torch.zeros(m_pad + 1, N, dtype=torch.bfloat16, device="cuda")
        flydsl_a16w4_gemm2(
            inter_sorted_bf16=a,
            w2_u8=w,
            w2_scale_u8=scale,
            sorted_expert_ids=eids,
            cumsum_tensor=cumsum,
            sorted_token_ids=sorted_ids,
            sorted_weights=weights,
            flat_out=out.view(-1),
            M_logical=m_pad,
            max_sorted=sorted_ids.numel(),
            NE=NE,
            D_HIDDEN=N,
            D_INTER=K,
            topk=1,
            tile_m=BM,
            epilog=epilog,
            use_valid_token_count=epilog == "atomic",
        )
        torch.cuda.synchronize()
        return out

    reduce, atomic = run("reduce"), run("atomic")
    assert atomic[:TOKENS].abs().sum() > 0
    # topk 1: each real token's one slot is its reduce row, bit-equal to atomic.
    assert torch.equal(reduce[:TOKENS], atomic[:TOKENS])
    written = (reduce[TOKENS:].abs().sum(dim=1) > 0).nonzero().flatten() + TOKENS
    assert not written.numel(), (
        f"rows {written.tolist()} written past the real rows; row {m_pad} is "
        f"past the [M * topk = {m_pad}, N] buffer"
    )


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
