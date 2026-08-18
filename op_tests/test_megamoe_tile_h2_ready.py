# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import pytest
import torch


def _require_gpu_and_flydsl():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")


def _logits_diff(reference: torch.Tensor, actual: torch.Tensor) -> float:
    ref = reference.float().double()
    got = actual.float().double()
    denominator = ref.square().sum() + got.square().sum()
    if denominator.item() == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (ref * got).sum() / denominator)


@pytest.mark.parametrize(
    "tokens,hidden,inter,experts,topk,workers",
    [
        (8, 1024, 256, 8, 2, 32),
        (8, 3584, 384, 56, 16, 192),
    ],
    ids=("small_token8", "k3_target_token8"),
)
def test_ready_h2_matches_weighted_local_partial(
    tokens: int,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    workers: int,
):
    """Consume sc2 H1 tiles and match the existing weighted GMM2 partial."""

    _require_gpu_and_flydsl()
    # Keep the baseline's spatial policy deterministic. The ready kernel is
    # persistent, so BF16 atomic accumulation may still differ in low bits.
    os.environ["MXFP4_G2_SPART"] = "0"
    os.environ["MXFP4_G2_BF16_LDS"] = "1"
    os.environ["MXFP4_G2_KSTATIC"] = "1"

    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_hier_epoch_module,
        compile_hier_stage1_ready_a4w4,
        compile_hier_stage2_partial_a4w4,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    seed = 83 + hidden
    torch.manual_seed(seed)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    bm = 32

    x = (torch.randn(tokens, hidden, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (
        torch.randn(experts, 2 * inter, hidden, device=dev) * 0.03
    ).to(torch.bfloat16)
    w2 = (torch.randn(experts, hidden, inter, device=dev) * 0.03).to(
        torch.bfloat16
    )
    score = torch.rand(tokens, experts, device=dev)
    values, ids = torch.topk(score, topk, dim=1)
    route_weights = torch.softmax(values, dim=1).float()
    prepared = prepare_local_a4w4_weights(w1, w2)

    sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
        ids.to(torch.int32),
        route_weights,
        experts,
        hidden,
        torch.bfloat16,
        bm,
        accumulate=False,
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(
        a1s.view(tokens, 1, hidden // 32),
        sorted_ids,
        nvalid,
        tokens,
        bm,
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    max_sorted = int(sorted_ids.shape[0])
    max_m_tiles = (max_sorted + bm - 1) // bm
    active_m_tiles = int(nvalid[0].item()) // bm
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((inter // 32) + 7) // 8 * 8

    layout = HierCcoArenaLayout.create(
        max_m_tiles=max_m_tiles,
        max_source_tokens=tokens,
    )
    arena = layout.allocate_local(device=dev)
    generation = 51
    ptrs = layout.epoch_pointers(arena.data_ptr(), generation)
    reset, publish_plan, _ = build_hier_epoch_module(
        max_m_tiles=max_m_tiles,
        max_source_tokens=tokens,
    )
    reset(
        ptrs.h1_input_expected,
        ptrs.h1_input_ready,
        ptrs.h1_output_done,
        ptrs.h2_output_done,
        ptrs.rank_route_expected,
        ptrs.rank_route_ready,
        active_m_tiles,
        tokens,
        1,
        stream=stream,
    )
    input_expected = layout.view(
        arena, "h1_input_expected", parity=ptrs.parity
    )
    input_ready = layout.view(arena, "h1_input_ready", parity=ptrs.parity)
    input_expected[:active_m_tiles].fill_(1)
    input_ready[:active_m_tiles].fill_(1)
    publish_plan(ptrs.plan_ready, generation, stream=stream)

    h1_q = torch.zeros(
        (max_sorted, inter // 2), dtype=torch.uint8, device=dev
    )
    h1_s = torch.zeros(
        scale_rows * scale_cols, dtype=torch.uint8, device=dev
    )
    dummy_hidden = torch.zeros(
        (tokens, hidden), dtype=torch.bfloat16, device=dev
    )
    h1 = compile_hier_stage1_ready_a4w4(
        D_HIDDEN=hidden,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        activation="silu",
    )
    assert "_sc2_" in h1.kernel_name
    h1_work = active_m_tiles * h1.num_n_blocks
    h1(
        ptrs.plan_ready,
        ptrs.h1_input_ready,
        ptrs.h1_input_expected,
        ptrs.h1_output_done,
        ptrs.h1_output_ready,
        generation,
        a1q.data_ptr(),
        a1ss.data_ptr(),
        prepared.w1.data_ptr(),
        prepared.w1_scale.data_ptr(),
        sorted_eids.data_ptr(),
        nvalid.data_ptr(),
        m_indices.data_ptr(),
        tokens,
        min(workers, h1_work),
        h1_q.data_ptr(),
        h1_s.data_ptr(),
        dummy_hidden.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    h1_ready = layout.view(arena, "h1_output_ready", parity=ptrs.parity)
    assert torch.all(h1_ready[:active_m_tiles] == generation)

    reference = torch.zeros(
        (tokens, hidden), dtype=torch.bfloat16, device=dev
    )
    mxfp4_moe_gemm2(
        inter_sorted_quant=h1_q,
        inter_sorted_shuffled_scale=h1_s,
        w2_u8=prepared.w2.view(torch.uint8),
        w2_scale_u8=prepared.w2_scale.view(torch.uint8),
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=nvalid,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        out=reference,
        M_logical=tokens,
        max_sorted=max_sorted,
        NE=experts,
        D_HIDDEN=hidden,
        D_INTER=inter,
        topk=topk,
        BM=bm,
        BN=128,
        BK=128,
        a_dtype="fp4",
        epilog="atomic",
        SBM=bm,
        HIDDEN_MAX=hidden,
        INTER_MAX=inter,
    )

    actual = torch.zeros_like(reference)
    h2 = compile_hier_stage2_partial_a4w4(
        D_HIDDEN=hidden,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
    )
    h2_work = active_m_tiles * h2.num_n_blocks
    h2(
        ptrs.h1_output_ready,
        ptrs.h2_output_done,
        ptrs.h2_output_ready,
        generation,
        h1_q.data_ptr(),
        h1_s.data_ptr(),
        prepared.w2.data_ptr(),
        prepared.w2_scale.data_ptr(),
        sorted_eids.data_ptr(),
        nvalid.data_ptr(),
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        tokens,
        max_m_tiles,
        min(workers, h2_work),
        actual.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual.float(), reference.float(), rtol=2.0e-2, atol=2.0e-2
    )
    assert _logits_diff(reference, actual) < 1.0e-2

    h2_done = layout.view(arena, "h2_output_done", parity=ptrs.parity)
    h2_ready = layout.view(arena, "h2_output_ready", parity=ptrs.parity)
    assert torch.all(h2_done[:active_m_tiles] == h2.num_n_blocks)
    assert torch.all(h2_ready[:active_m_tiles] == generation)
    assert torch.count_nonzero(h2_done[active_m_tiles:]).item() == 0
    assert torch.count_nonzero(h2_ready[active_m_tiles:]).item() == 0
    if hidden == 3584:
        assert h2.num_n_blocks == 28
        assert torch.all(h2_done[:active_m_tiles] == 28)

