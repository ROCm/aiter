# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def _require_gpu_and_flydsl():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")


def test_ready_epoch_and_target_gmm1_gmm2_compile_run_smoke():
    """Run control atoms and JIT target-shape compute with an empty route set."""

    _require_gpu_and_flydsl()
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_hier_epoch_module,
        build_h1_ready_queue_publisher,
        compile_hier_stage1_queue_a4w4,
        compile_hier_stage1_ready_a4w4,
        compile_hier_stage2_partial_a4w4,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    generation = 11
    max_tiles = 8
    max_source_tokens = 32
    layout = HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=4,
        max_m_tiles=max_tiles,
        max_source_tokens=max_source_tokens,
    )
    arena = layout.allocate_local(device=dev)
    ptrs = layout.epoch_pointers(arena.data_ptr(), generation)

    # Prove the control-plane kernels execute, not merely compile.
    input_expected = layout.view(arena, "h1_input_expected", parity=ptrs.parity)
    input_ready = layout.view(arena, "h1_input_ready", parity=ptrs.parity)
    output_done = layout.view(arena, "h1_output_done", parity=ptrs.parity)
    h2_done = layout.view(arena, "h2_output_done", parity=ptrs.parity)
    route_expected = layout.view(
        arena, "rank_route_expected", parity=ptrs.parity
    )
    route_ready = layout.view(arena, "rank_route_ready", parity=ptrs.parity)
    for tensor in (
        input_expected,
        input_ready,
        output_done,
        h2_done,
        route_expected,
        route_ready,
    ):
        tensor.fill_(7)

    reset, publish_plan, mark_input = build_hier_epoch_module(
        max_m_tiles=max_tiles, max_source_tokens=max_source_tokens
    )
    reset(
        ptrs.h1_input_expected,
        ptrs.h1_input_ready,
        ptrs.h1_output_done,
        ptrs.h2_output_done,
        ptrs.rank_route_expected,
        ptrs.rank_route_ready,
        max_tiles,
        max_source_tokens,
        1,
        stream=stream,
    )
    publish_plan(ptrs.plan_ready, generation, stream=stream)
    mark_input(ptrs.h1_input_ready, 0, 3, stream=stream)
    torch.cuda.synchronize(dev)
    assert torch.count_nonzero(input_expected).item() == 0
    assert input_ready[0].item() == 3
    assert torch.count_nonzero(input_ready[1:]).item() == 0
    assert torch.count_nonzero(output_done).item() == 0
    assert torch.count_nonzero(h2_done).item() == 0
    assert torch.count_nonzero(route_expected).item() == 0
    assert torch.count_nonzero(route_ready).item() == 0
    assert layout.view(arena, "plan_ready")[ptrs.parity].item() == generation

    # cumsum=0 makes runtime work empty while JIT still compiles the complete
    # target H3584/I384/E56 kernels and activation/epilogues.
    dummy = torch.zeros(4096, dtype=torch.uint8, device=dev)
    cumsum = torch.zeros(2, dtype=torch.int32, device=dev)

    h1 = compile_hier_stage1_ready_a4w4(
        D_HIDDEN=3584,
        D_INTER=384,
        NE=56,
        TOPK=16,
        activation="silu",
    )
    h1(
        ptrs.plan_ready,
        ptrs.h1_input_ready,
        ptrs.h1_input_expected,
        ptrs.h1_output_done,
        ptrs.h1_output_ready,
        generation,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        cumsum.data_ptr(),
        dummy.data_ptr(),
        1,
        1,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        stream=stream,
    )

    queue_capacity = layout.view(
        arena, "h1_ready_queue", parity=ptrs.parity
    ).numel()
    init_queue, _, finish_queue = build_h1_ready_queue_publisher(
        num_n_blocks=3, max_work=queue_capacity
    )
    init_queue(
        ptrs.h1_queue_header, generation, 0, stream=stream
    )
    finish_queue(ptrs.h1_queue_header, generation, stream=stream)
    queue_h1 = compile_hier_stage1_queue_a4w4(
        D_HIDDEN=3584,
        D_INTER=384,
        NE=56,
        TOPK=16,
        max_work=queue_capacity,
        activation="silu",
    )
    queue_h1(
        ptrs.h1_queue_header,
        ptrs.h1_ready_queue,
        generation,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        cumsum.data_ptr(),
        dummy.data_ptr(),
        1,
        1,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        stream=stream,
    )

    h2 = compile_hier_stage2_partial_a4w4(
        D_HIDDEN=3584,
        D_INTER=384,
        NE=56,
        TOPK=16,
    )
    h2(
        ptrs.h1_output_ready,
        ptrs.h2_output_done,
        ptrs.h2_output_ready,
        generation,
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        cumsum.data_ptr(),
        dummy.data_ptr(),
        dummy.data_ptr(),
        1,
        1,
        1,
        dummy.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    assert "h3584_i384_e56_k16" in h1.kernel_name
    assert "_wpe2_" in h1.kernel_name
    assert "h3584_i384_e56_k16" in h2.kernel_name
    assert "h3584_i384_e56_k16" in queue_h1.kernel_name
    assert "_wpe2_" in queue_h1.kernel_name
    assert h1.num_n_blocks == 3
    assert queue_h1.num_n_blocks == 3
    assert h2.num_n_blocks == 28
    assert h1.lds_bytes > 0 and h2.lds_bytes > 0
    assert queue_h1.lds_bytes > 0
    assert h1.input_ready_kind == "route-count"
    assert h2.output_contract == "weighted-rank-local-source-partial"
    assert h2.requires_zeroed_output is True


def test_ready_h1_executes_valid_tiles_bitwise_across_two_epochs():
    """Compare ready-aware H1 with the existing persistent A4W4 port."""

    _require_gpu_and_flydsl()
    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        build_hier_epoch_module,
        build_h1_ready_queue_publisher,
        compile_hier_stage1_queue_a4w4,
        compile_hier_stage1_ready_a4w4,
    )
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    torch.manual_seed(29)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    m, h, inter, experts, topk, bm = 4, 1024, 256, 8, 2, 32
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.05).to(
        torch.bfloat16
    )
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.05).to(
        torch.bfloat16
    )
    ids = torch.tensor(
        [[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.int32, device=dev
    )
    route_weights = torch.tensor(
        [[0.6, 0.4]] * m, dtype=torch.float32, device=dev
    )
    prepared = prepare_local_a4w4_weights(w1, w2)
    sorted_ids, _, sorted_eids, nvalid, _ = moe_sorting(
        ids, route_weights, experts, h, torch.bfloat16, bm, accumulate=False
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(
        a1s.view(m, 1, h // 32), sorted_ids, nvalid, m, bm
    )
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()
    max_sorted = int(sorted_ids.shape[0])
    max_m_tiles = (max_sorted + bm - 1) // bm
    active_m_tiles = int(nvalid[0].item()) // bm
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((inter // 32) + 7) // 8 * 8
    dummy_hidden = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)

    reference_q = torch.zeros(
        (max_sorted, inter // 2), dtype=torch.uint8, device=dev
    )
    reference_s = torch.zeros(
        scale_rows * scale_cols, dtype=torch.uint8, device=dev
    )
    flydsl_mxfp4_gemm1(
        a_quant=a1q,
        a_scale_sorted_shuffled=a1ss,
        w1_u8=prepared.w1.view(torch.uint8),
        w1_scale_u8=prepared.w1_scale.view(torch.uint8),
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=nvalid,
        m_indices=m_indices,
        inter_sorted_quant=reference_q,
        inter_sorted_shuffled_scale=reference_s,
        hidden_states=dummy_hidden,
        n_tokens=m,
        BM=bm,
        use_nt=True,
        inline_quant=False,
        NE=experts,
        D_HIDDEN=h,
        D_INTER=inter,
        topk=topk,
        act="silu",
        persistent=True,
        persistent_blocks=4,
    )
    torch.cuda.synchronize(dev)

    layout = HierCcoArenaLayout.create(
        max_m_tiles=max_m_tiles,
        max_source_tokens=32,
    )
    arena = layout.allocate_local(device=dev)
    reset, publish_plan, mark_input = build_hier_epoch_module(
        max_m_tiles=max_m_tiles, max_source_tokens=32
    )
    ready_h1 = compile_hier_stage1_ready_a4w4(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        activation="silu",
    )
    assert ready_h1.num_n_blocks == 2

    for generation in (21, 22):
        ptrs = layout.epoch_pointers(arena.data_ptr(), generation)
        input_expected = layout.view(
            arena, "h1_input_expected", parity=ptrs.parity
        )
        output_done = layout.view(
            arena, "h1_output_done", parity=ptrs.parity
        )
        output_ready = layout.view(
            arena, "h1_output_ready", parity=ptrs.parity
        )
        reset(
            ptrs.h1_input_expected,
            ptrs.h1_input_ready,
            ptrs.h1_output_done,
            ptrs.h2_output_done,
            ptrs.rank_route_expected,
            ptrs.rank_route_ready,
            active_m_tiles,
            m,
            1,
            stream=stream,
        )
        input_expected[:active_m_tiles].fill_(1)
        for tile in range(active_m_tiles):
            mark_input(ptrs.h1_input_ready, tile, 1, stream=stream)
        publish_plan(ptrs.plan_ready, generation, stream=stream)

        result_q = torch.zeros_like(reference_q)
        result_s = torch.zeros_like(reference_s)
        ready_h1(
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
            m,
            4,
            result_q.data_ptr(),
            result_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=stream,
        )
        torch.cuda.synchronize(dev)

        assert torch.equal(result_q, reference_q), generation
        assert torch.equal(result_s, reference_s), generation
        assert torch.all(
            output_done[:active_m_tiles] == ready_h1.num_n_blocks
        )
        assert torch.all(output_ready[:active_m_tiles] == generation)

    # A/B scheduler: the sidecar publishes ready M tiles to a global queue;
    # compute CTAs own deterministic queue positions and carry no expected/output
    # counters through the MFMA body.
    queue_generation = 31
    queue_ptrs = layout.epoch_pointers(arena.data_ptr(), queue_generation)
    queue_header = layout.view(
        arena, "h1_queue_header", parity=queue_ptrs.parity
    )
    queue_entries = layout.view(
        arena, "h1_ready_queue", parity=queue_ptrs.parity
    )
    total_work = active_m_tiles * ready_h1.num_n_blocks
    queue_capacity = queue_entries.numel()
    init_queue, publish_tile, finish_queue = build_h1_ready_queue_publisher(
        num_n_blocks=ready_h1.num_n_blocks,
        max_work=queue_capacity,
    )
    queue_h1 = compile_hier_stage1_queue_a4w4(
        D_HIDDEN=h,
        D_INTER=inter,
        NE=experts,
        TOPK=topk,
        max_work=queue_capacity,
        activation="silu",
    )
    init_queue(
        queue_ptrs.h1_queue_header,
        queue_generation,
        total_work,
        stream=stream,
    )
    for m_tile in range(active_m_tiles):
        publish_tile(
            queue_ptrs.h1_queue_header,
            queue_ptrs.h1_ready_queue,
            m_tile,
            stream=stream,
        )
    finish_queue(
        queue_ptrs.h1_queue_header, queue_generation, stream=stream
    )
    queue_q = torch.zeros_like(reference_q)
    queue_s = torch.zeros_like(reference_s)
    queue_h1(
        queue_ptrs.h1_queue_header,
        queue_ptrs.h1_ready_queue,
        queue_generation,
        a1q.data_ptr(),
        a1ss.data_ptr(),
        prepared.w1.data_ptr(),
        prepared.w1_scale.data_ptr(),
        sorted_eids.data_ptr(),
        nvalid.data_ptr(),
        m_indices.data_ptr(),
        m,
        4,
        queue_q.data_ptr(),
        queue_s.data_ptr(),
        dummy_hidden.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    assert torch.equal(queue_q, reference_q)
    assert torch.equal(queue_s, reference_s)
    assert queue_header.tolist() == [
        queue_generation,
        total_work,
        total_work,
        queue_generation,
    ]
    assert torch.equal(
        queue_entries[:total_work].cpu(), torch.arange(total_work, dtype=torch.int32)
    )
