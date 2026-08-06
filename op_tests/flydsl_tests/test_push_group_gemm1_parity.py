# SPDX-License-Identifier: MIT
"""Task 4: push-group fixed-slot GEMM1 A-load == compact gather GEMM1 (byte-exact).

Drives the a8w4 TDM GEMM1 two ways on the SAME bf16 tokens / weights:

  * pull  -- flydsl_moe_topids_to_rows -> contiguous_psum_remap ->
             fused_quant_preshuffle(gather) -> gemm (compact rows).
  * push  -- tokens placed into a fixed-slot [E, CAP] grid ->
             fused_quant_preshuffle(masked, identity map) ->
             push_group_finalize -> gemm(push_group=1, tile_row_base/expert_ids).

Each token's post-GEMM1 logit row must be byte-identical between the two paths
(same fp8 quant of the same bf16 row, same weight; only the physical row differs).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs gfx1250"
)


def _is_gfx1250():
    try:
        from flydsl.runtime.device import get_rocm_arch

        return "gfx1250" in get_rocm_arch().lower()
    except Exception:
        return False


@pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250 hardware")
def test_push_group_gemm1_byte_exact():
    from aiter.ops.shuffle import moe_shuffle_weight, moe_shuffle_scale
    from aiter.ops.flydsl.grouped_moe_gfx1250 import (
        _grouped_weight_uint8,
        contiguous_psum_remap,
    )
    from aiter.ops.flydsl.moe_kernels import (
        flydsl_moe_fused_quant_preshuffle,
        flydsl_moe_topids_to_rows,
    )
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked
    from aiter.ops.flydsl.kernels.push_group_finalize_gfx1250 import (
        launch_push_group_finalize,
    )

    dev = torch.device("cuda")
    torch.manual_seed(7)
    E = 8
    token_num = 64
    topk = 2
    K = 512          # model_dim
    inter = 256
    two_inter = 2 * inter
    tile_m, tile_n, tile_k = 64, 256, 256
    wmma_rep = tile_m // 16  # 4
    SCALE_BLOCK = 32

    hidden = (torch.randn(token_num, K, device=dev) * 0.5).to(torch.bfloat16)
    # distinct experts per token
    ids = torch.empty(token_num, topk, dtype=torch.int32, device=dev)
    for t in range(token_num):
        ids[t] = torch.randperm(E, device=dev)[:topk].to(torch.int32)

    # ---- a8w4 gate+up weight (mxfp4) + n32k4 scale ----
    w1_packed = torch.randint(
        0, 256, (E, two_inter, K // 2), dtype=torch.uint8, device=dev
    )
    w1_scale_raw = torch.randint(
        118, 130, (E, two_inter, K // SCALE_BLOCK), dtype=torch.uint8, device=dev
    )
    w1_grouped = moe_shuffle_weight(w1_packed, experts_cnt=E, gate_up=True)
    w1_scale = moe_shuffle_scale(w1_scale_raw.contiguous(), experts_cnt=E)
    w1_u8 = _grouped_weight_uint8(w1_grouped)
    w1s_i32 = w1_scale.reshape(-1).view(torch.int32)

    align = tile_m
    max_m = max(align, ((token_num * topk + align - 1) // align) * align)
    contiguous_m = max(
        align,
        ((token_num * topk + E * align - topk + align - 1) // align) * align,
    )

    # ===================== PULL (reference) =====================
    _masked_m, topids_to_rows = flydsl_moe_topids_to_rows(ids, E, max_m)
    _starts, psum, _ = contiguous_psum_remap(_masked_m, topids_to_rows, E, max_m, tile_m)
    psum = psum.to(torch.int32).contiguous()
    a1p, a1s = flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, token_num, K), 1, contiguous_m,
        wmma_rep=wmma_rep, quant_mode="fp8",
        masked_m=None, topids_to_rows=topids_to_rows, source_topk=topk,
    )
    y_pull = torch.zeros((1, contiguous_m, inter), dtype=torch.bfloat16, device=dev)
    flydsl_grouped_gemm_a8w4_masked(
        y_pull, a1p, w1_u8, a1s, w1s_i32, psum,
        n_experts=E, contiguous_m=contiguous_m, N=two_inter, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        out_is_f16=0, a_is_fp4=0, stage1_act=1, bias=None, num_buffers=2,
    )
    torch.cuda.synchronize()

    # ===================== PUSH (fixed-slot) =====================
    CAP = ((token_num * topk + tile_m - 1) // tile_m) * tile_m  # >= max per-expert
    grouped = torch.zeros(E, CAP, K, dtype=torch.bfloat16, device=dev)
    counts = [0] * E
    push_row = {}  # (t,k) -> grouped flat row
    ids_cpu = ids.cpu()
    for t in range(token_num):
        for k in range(topk):
            e = int(ids_cpu[t, k])
            r = counts[e]
            grouped[e, r] = hidden[t]
            push_row[(t, k)] = e * CAP + r
            counts[e] += 1
    counts_t = torch.tensor(counts, dtype=torch.int32, device=dev)

    a1p_pg, a1s_pg = flydsl_moe_fused_quant_preshuffle(
        grouped.reshape(E, CAP, K), E, CAP,
        wmma_rep=wmma_rep, quant_mode="fp8",
        masked_m=counts_t, topids_to_rows=None,
    )
    cm_pg = E * CAP
    max_tiles = cm_pg // tile_m
    trb = torch.full((max_tiles,), -1, dtype=torch.int32, device=dev)
    eids = torch.full((max_tiles,), E, dtype=torch.int32, device=dev)  # E == skip sentinel
    tvd = torch.zeros((max_tiles,), dtype=torch.int32, device=dev)
    num_valid = torch.zeros(1, dtype=torch.int32, device=dev)
    # Request compact metadata layout: persistent GEMM enumerates exactly the
    # [0, num_valid / tile_m) prefix, while valid_rows is unused in this test.
    valid_rows = torch.empty(token_num * topk, dtype=torch.int32, device=dev)
    valid_routes = torch.zeros(1, dtype=torch.int32, device=dev)
    launch_push_group_finalize(
        pg_running_ptr=counts_t.data_ptr(),
        tile_row_base_ptr=trb.data_ptr(),
        expert_ids_ptr=eids.data_ptr(),
        num_valid_ptr=num_valid.data_ptr(),
        tile_valid_ptr=tvd.data_ptr(),
        num_local_experts=E, cap=CAP, tile_m=tile_m, rank=0, experts_per_rank=E,
        valid_rows_ptr=valid_rows.data_ptr(),
        valid_routes_ptr=valid_routes.data_ptr(),
    )
    torch.cuda.synchronize()

    psum_dummy = torch.zeros(E, dtype=torch.int32, device=dev)
    y_push = torch.zeros((1, cm_pg, inter), dtype=torch.bfloat16, device=dev)
    flydsl_grouped_gemm_a8w4_masked(
        y_push, a1p_pg, w1_u8, a1s_pg, w1s_i32, psum_dummy,
        n_experts=E, contiguous_m=cm_pg, N=two_inter, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        out_is_f16=0, a_is_fp4=0, stage1_act=1, bias=None, num_buffers=2,
        push_group=1, tile_row_base=trb, expert_ids=eids, tile_valid=tvd,
    )

    # The persistent scheduler receives the exact tile-aligned valid-row count
    # produced by finalize, never reads it on the host, and must write the same
    # fixed-slot output rows as the static compact scheduler above.
    y_push_persistent = torch.zeros_like(y_push)
    flydsl_grouped_gemm_a8w4_masked(
        y_push_persistent, a1p_pg, w1_u8, a1s_pg, w1s_i32, psum_dummy,
        n_experts=E, contiguous_m=cm_pg, N=two_inter, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        out_is_f16=0, a_is_fp4=0, stage1_act=1, bias=None, num_buffers=2,
        push_group=1, tile_row_base=trb, expert_ids=eids, tile_valid=tvd,
        ep_persistent_gemm1=1, num_valid_rows=num_valid, persistent_workers=256,
    )
    torch.cuda.synchronize()
    assert torch.equal(y_push, y_push_persistent)

    # ===================== compare per (token, route) =====================
    tir = topids_to_rows.reshape(token_num, topk).cpu()
    yp = y_pull[0].view(torch.int16).cpu()
    yg = y_push[0].view(torch.int16).cpu()
    mism = 0
    for t in range(token_num):
        for k in range(topk):
            pull_r = int(tir[t, k])
            push_r = push_row[(t, k)]
            if not torch.equal(yp[pull_r], yg[push_r]):
                mism += 1
                if mism <= 5:
                    d = (y_pull[0][pull_r].float() - y_push[0][push_r].float()).abs().max()
                    print(f"[MISM] t={t} k={k} pull_row={pull_r} push_row={push_r} maxabs={d}")
    assert mism == 0, f"{mism} token rows differ between pull and push GEMM1"
    print("[PASS] push-group GEMM1 byte-exact vs compact gather")
