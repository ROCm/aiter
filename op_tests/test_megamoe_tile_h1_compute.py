# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import torch


def test_fused_h1_compute_matches_existing_a4w4_port_bitwise():
    from aiter.fused_moe import moe_sorting
    from aiter.ops.flydsl.kernels.megamoe_tile import PersistentH1Workspace, prepare_local_a4w4_weights
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import (
        compile_hier_stage1_a4w4,
        compile_hier_stage1_a4w4_persistent,
        compile_hier_stage2_a4w4,
    )
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import flydsl_mxfp4_gemm1
    from aiter.ops.flydsl.kernels.mxfp4_gemm1 import gemm1_grid
    from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    torch.manual_seed(19)
    dev = torch.device("cuda", 0)
    m, h, inter, experts, topk, bm = 4, 1024, 256, 8, 2, 32
    x = (torch.randn(m, h, device=dev) * 0.1).to(torch.bfloat16)
    w1 = (torch.randn(experts, 2 * inter, h, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(experts, h, inter, device=dev) * 0.05).to(torch.bfloat16)
    ids = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.int32, device=dev)
    weights = torch.tensor([[0.6, 0.4]] * m, dtype=torch.float32, device=dev)
    prepared = prepare_local_a4w4_weights(w1, w2)

    sorted_ids, sorted_weights, sorted_eids, nvalid, _ = moe_sorting(
        ids, weights, experts, h, torch.bfloat16, bm, accumulate=False
    )
    a1q, a1s = per_1x32_f4_quant(x, shuffle=False)
    a1ss = moe_mxfp4_sort(a1s.view(m, 1, h // 32), sorted_ids, nvalid, m, bm)
    m_indices = (sorted_ids & 0x00FFFFFF).to(torch.int32).contiguous()

    max_sorted = sorted_ids.shape[0]
    scale_rows = (max_sorted + 255) // 256 * 256
    scale_cols = ((inter // 32) + 7) // 8 * 8
    dummy_hidden = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)
    copy_src = torch.arange(4096, dtype=torch.int32, device=dev).remainder(251).to(torch.uint8)
    grid = gemm1_grid(m, bm, NE=experts, TOPK=topk, INTER=inter, BN=256)
    kernel_names = set()
    persistent_kernel_names = set()
    packed_by_activation = {}
    activation_cases = (
        ("silu", None, 4.0, 25.0),
        ("swiglu", 3.0, 4.0, 25.0),
        ("situv2", None, 2.0, 9.0),
    )
    for case_index, (activation, limit, beta, linear_beta) in enumerate(
        activation_cases
    ):
        ref_q = torch.zeros(
            (max_sorted, inter // 2), dtype=torch.uint8, device=dev
        )
        ref_s = torch.zeros(
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
            inter_sorted_quant=ref_q,
            inter_sorted_shuffled_scale=ref_s,
            hidden_states=dummy_hidden,
            n_tokens=m,
            BM=bm,
            use_nt=True,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            act=activation,
            swiglu_limit=limit,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        torch.cuda.synchronize()

        port_persistent_q = torch.zeros_like(ref_q)
        port_persistent_s = torch.zeros_like(ref_s)
        flydsl_mxfp4_gemm1(
            a_quant=a1q,
            a_scale_sorted_shuffled=a1ss,
            w1_u8=prepared.w1.view(torch.uint8),
            w1_scale_u8=prepared.w1_scale.view(torch.uint8),
            sorted_expert_ids=sorted_eids,
            cumsum_tensor=nvalid,
            m_indices=m_indices,
            inter_sorted_quant=port_persistent_q,
            inter_sorted_shuffled_scale=port_persistent_s,
            hidden_states=dummy_hidden,
            n_tokens=m,
            BM=bm,
            use_nt=True,
            inline_quant=False,
            NE=experts,
            D_HIDDEN=h,
            D_INTER=inter,
            topk=topk,
            act=activation,
            swiglu_limit=limit,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
            persistent=True,
            persistent_blocks=4,
        )
        torch.cuda.synchronize()
        assert torch.equal(port_persistent_q, ref_q), activation
        assert torch.equal(port_persistent_s, ref_s), activation

        out_q = torch.zeros_like(ref_q)
        out_s = torch.zeros_like(ref_s)
        copy_dst = torch.zeros_like(copy_src)
        signal = torch.zeros(1, dtype=torch.int64, device=dev)
        fused = compile_hier_stage1_a4w4(
            D_HIDDEN=h,
            D_INTER=inter,
            NE=experts,
            TOPK=topk,
            activation=activation,
            swiglu_limit=limit,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        kernel_names.add(fused.kernel_name)
        generation = 23 + case_index
        fused(
            copy_src.data_ptr(),
            copy_dst.data_ptr(),
            signal.data_ptr(),
            copy_src.numel(),
            generation,
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            grid,
            out_q.data_ptr(),
            out_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert torch.equal(copy_src, copy_dst)
        assert signal.item() == generation
        assert torch.equal(out_q, ref_q), activation
        assert torch.equal(out_s, ref_s), activation

        # Persistent H1: eight resident workers dynamically claim the same
        # flat GEMM tiles. This forces multiple queue iterations at this shape.
        persistent_q = torch.zeros_like(ref_q)
        persistent_s = torch.zeros_like(ref_s)
        persistent_dst = torch.zeros_like(copy_src)
        persistent_signal = torch.zeros_like(signal)
        persistent = compile_hier_stage1_a4w4_persistent(
            D_HIDDEN=h,
            D_INTER=inter,
            NE=experts,
            TOPK=topk,
            WORK_SHARDS=8,
            activation=activation,
            swiglu_limit=limit,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        persistent_kernel_names.add(persistent.kernel_name)
        persistent_workspace = PersistentH1Workspace.allocate(
            work_shards=persistent.work_shards,
            worker_blocks=8,
            device=dev,
        )
        persistent.checked(
            persistent_workspace,
            copy_src.data_ptr(),
            persistent_dst.data_ptr(),
            persistent_signal.data_ptr(),
            copy_src.numel(),
            generation + 100,
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            persistent_q.data_ptr(),
            persistent_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert torch.equal(copy_src, persistent_dst)
        assert persistent_signal.item() == generation + 100
        assert torch.equal(persistent_q, ref_q), activation
        assert torch.equal(persistent_s, ref_s), activation
        assert int(persistent_workspace.work_head[::16].sum().item()) > 0

        if case_index == 0:
            # The device owner advances the epoch and resets work-head state;
            # no host memset or workspace replacement is allowed here.
            persistent_q.zero_()
            persistent_s.zero_()
            persistent_signal.zero_()
            persistent.checked(
                persistent_workspace,
                copy_src.data_ptr(),
                persistent_dst.data_ptr(),
                persistent_signal.data_ptr(),
                copy_src.numel(),
                generation + 200,
                a1q.data_ptr(),
                a1ss.data_ptr(),
                prepared.w1.data_ptr(),
                prepared.w1_scale.data_ptr(),
                sorted_eids.data_ptr(),
                nvalid.data_ptr(),
                m_indices.data_ptr(),
                m,
                persistent_q.data_ptr(),
                persistent_s.data_ptr(),
                dummy_hidden.data_ptr(),
                stream=torch.cuda.current_stream(),
            )
            torch.cuda.synchronize()
            assert persistent_signal.item() == generation + 200
            assert torch.equal(persistent_q, ref_q)
            assert torch.equal(persistent_s, ref_s)
            assert persistent_workspace.entry_count.item() == 16
            assert persistent_workspace.epoch_gate.item() == 2

        strided_q = torch.zeros_like(ref_q)
        strided_s = torch.zeros_like(ref_s)
        strided_dst = torch.zeros_like(copy_src)
        strided_signal = torch.zeros_like(signal)
        strided = compile_hier_stage1_a4w4_persistent(
            D_HIDDEN=h,
            D_INTER=inter,
            NE=experts,
            TOPK=topk,
            WORK_SHARDS=8,
            scheduler="strided",
            activation=activation,
            swiglu_limit=limit,
            situ_beta=beta,
            situ_linear_beta=linear_beta,
        )
        strided_workspace = PersistentH1Workspace.allocate(
            work_shards=8, worker_blocks=8, device=dev
        )
        strided.checked(
            strided_workspace,
            copy_src.data_ptr(),
            strided_dst.data_ptr(),
            strided_signal.data_ptr(),
            copy_src.numel(),
            generation + 300,
            a1q.data_ptr(),
            a1ss.data_ptr(),
            prepared.w1.data_ptr(),
            prepared.w1_scale.data_ptr(),
            sorted_eids.data_ptr(),
            nvalid.data_ptr(),
            m_indices.data_ptr(),
            m,
            strided_q.data_ptr(),
            strided_s.data_ptr(),
            dummy_hidden.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        torch.cuda.synchronize()
        assert torch.equal(copy_src, strided_dst)
        assert strided_signal.item() == generation + 300
        assert torch.equal(strided_q, ref_q), activation
        assert torch.equal(strided_s, ref_s), activation
        packed_by_activation[activation] = (out_q.clone(), out_s.clone())
    assert len(kernel_names) == len(activation_cases)
    assert len(persistent_kernel_names) == len(activation_cases)
    assert not torch.equal(
        packed_by_activation["silu"][0], packed_by_activation["swiglu"][0]
    )
    assert not torch.equal(
        packed_by_activation["silu"][0], packed_by_activation["situv2"][0]
    )

    # H2: compare the existing linear GMM2 path with the new low-ticket
    # copy/return + GMM2 kernel. Disable spatial remap so both use the same tile
    # order for deterministic BF16 atomic accumulation.
    os.environ["MXFP4_G2_SPART"] = "0"
    os.environ["MXFP4_G2_BF16_LDS"] = "1"
    os.environ["MXFP4_G2_KSTATIC"] = "1"
    ref_out = torch.zeros((m, h), dtype=torch.bfloat16, device=dev)
    mxfp4_moe_gemm2(
        inter_sorted_quant=ref_q,
        inter_sorted_shuffled_scale=ref_s,
        w2_u8=prepared.w2.view(torch.uint8),
        w2_scale_u8=prepared.w2_scale.view(torch.uint8),
        sorted_expert_ids=sorted_eids,
        cumsum_tensor=nvalid,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        out=ref_out,
        M_logical=m,
        max_sorted=max_sorted,
        NE=experts,
        D_HIDDEN=h,
        D_INTER=inter,
        topk=topk,
        BM=bm,
        BN=128,
        BK=128,
        a_dtype="fp4",
        epilog="atomic",
        SBM=bm,
        HIDDEN_MAX=h,
        INTER_MAX=inter,
    )
    torch.cuda.synchronize()

    out2 = torch.zeros_like(ref_out)
    copy_dst2 = torch.zeros_like(copy_src)
    signal2 = torch.zeros_like(signal)
    actual_m_blocks = int(nvalid[0].item()) // bm
    max_m_blocks = (max_sorted + bm - 1) // bm
    g2_grid = actual_m_blocks * (h // 128)
    fused2 = compile_hier_stage2_a4w4(
        D_HIDDEN=h, D_INTER=inter, NE=experts, TOPK=topk
    )
    fused2(
        copy_src.data_ptr(),
        copy_dst2.data_ptr(),
        signal2.data_ptr(),
        copy_src.numel(),
        31,
        ref_q.data_ptr(),
        ref_s.data_ptr(),
        prepared.w2.data_ptr(),
        prepared.w2_scale.data_ptr(),
        sorted_eids.data_ptr(),
        nvalid.data_ptr(),
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        m,
        max_m_blocks,
        g2_grid,
        out2.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    assert torch.equal(copy_src, copy_dst2)
    assert signal2.item() == 31
    torch.testing.assert_close(out2.float(), ref_out.float(), rtol=2e-2, atol=2e-2)
