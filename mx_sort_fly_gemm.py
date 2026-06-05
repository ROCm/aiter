"""Hybrid MoE runner: mxfp4 (mx_fn) sort kernels + swappable gemm1/gemm2 backends.

Goal (per request): build ``mx_sort_fly_gemm1_gemm2_fn`` =
  1. reuse mx_fn's moe-sort-related kernels (mxfp4 threestage sort + quant + sort_scales);
  2. run gemm1 / gemm2 via pluggable backends. The default backend is the mxfp4
     C++/HIP kernel (so the hybrid is immediately correct and gives the perf
     upper bound). FlyDSL gemm1/gemm2 backends (faithful ports of the mxfp4 HIP
     kernels) are plugged in via ``gemm1_backend`` / ``gemm2_backend`` and tuned
     to match the HIP perf.

Currently targets the M=256 config of the Kimi-K2.5 TP=4 bench:
  BM=32, threestage sort, gemm2 ATOMIC (no scatter_reduce).
This mirrors ``aiter.fused_moe._mxfp4_moe_run`` for that config exactly.

Per request: the large-M nonatomic path reuses mx_fn's HIP scatter_reduce
(``aiter.mxfp4_moe_scatter_reduce`` / ``_q``) verbatim; only gemm1/gemm2 are
ported to FlyDSL.
"""

import torch

import aiter
from aiter import dtypes


def _empty_bf16(device):
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


def _empty_u8(device):
    return torch.empty((0,), dtype=torch.uint8, device=device)


def _mxfp4_gemm1_hip(*, cumsum_tensor, a_quant, a_scale_sorted_shuffled, w1, w1_scale,
                     sorted_expert_ids, m_indices, inter_sorted_quant,
                     inter_sorted_shuffled_scale, hidden_states, kernelName1):
    """Default gemm1 backend = mxfp4 HIP kernel."""
    aiter.mxfp4_moe_gemm1_a4w4(
        cumsum_tensor=cumsum_tensor,
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        w12_shuffled_quant=w1,
        w12_shuffled_scale=w1_scale,
        sorted_expert_ids=sorted_expert_ids,
        m_indices=m_indices,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        hidden_states=hidden_states,
        kernelName=kernelName1,
    )


def _mxfp4_gemm2_hip(*, cumsum_tensor, inter_sorted_quant, inter_sorted_shuffled_scale,
                     w2, w2_scale, sorted_token_ids, sorted_expert_ids, sorted_weights,
                     out_buf, M, max_sorted, kernelName2):
    """Default gemm2 backend = mxfp4 HIP kernel (atomic)."""
    aiter.mxfp4_moe_gemm2_a4w4(
        cumsum_tensor=cumsum_tensor,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w3_shuffled_quant=w2,
        w3_shuffled_scale=w2_scale,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights,
        flat_out=out_buf,
        M_logical=M,
        max_sorted=max_sorted,
        kernelName=kernelName2,
    )


def mx_sort_fly_gemm1_gemm2(
    hidden_states,
    w1,            # [E, 2*D_INTER, D_HIDDEN] packed MXFP4 (mx_w / a16w4 layout)
    w2,            # [E, D_HIDDEN, D_INTER]   packed MXFP4 (a16w4 layout)
    topk_ids,
    topk_weight,
    topk,
    *,
    w1_scale=None,
    w2_scale=None,
    BM=32,
    gemm1_backend=None,
    gemm2_backend=None,
):
    """mxfp4 sort prologue + swappable gemm1/gemm2 (BM32 / threestage / atomic).

    Returns out_buf [M, D_HIDDEN] bf16 (== mx_fn output).
    """
    device = hidden_states.device
    if w1.element_size() == 1 and w1.dtype != torch.uint8:
        w1 = w1.view(torch.uint8)
    if w2.element_size() == 1 and w2.dtype != torch.uint8:
        w2 = w2.view(torch.uint8)

    NE = w1.shape[0]
    D_HIDDEN = hidden_states.shape[1]
    D_INTER = w1.shape[1] // 2
    M = hidden_states.shape[0]

    # codegen'd HIP kernel names for this shape (BM32, atomic, no scatter_reduce)
    kernelName1 = f"mxfp4_moe_g1_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_BM{BM}_NT"
    kernelName2 = (
        f"mxfp4_moe_g2_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_TOPK{topk}_BM{BM}_ATOMIC_NT"
    )

    gemm1_backend = gemm1_backend or _mxfp4_gemm1_hip
    gemm2_backend = gemm2_backend or _mxfp4_gemm2_hip

    # ── sort buffers (mirror _mxfp4_moe_run) ──────────────────────────────
    active = min(NE, M * topk)
    cumsum_max = M * topk + active * (BM - 1)
    max_sorted = ((cumsum_max + BM - 1) // BM) * BM

    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sorted_expert_ids = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
    cumsum_tensor = torch.empty((1,), device=device, dtype=dtypes.i32)
    reverse_sorted = torch.empty((M * topk,), device=device, dtype=dtypes.i32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    masked_m = torch.empty((NE,), device=device, dtype=dtypes.i32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)

    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)

    # atomic gemm2 output buffer (zero-init'd by quant kernel below)
    atomic_output_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    # ── threestage sort + quant + sort_scales (mx_fn's sort-related kernels) ──
    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids, topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights,
        masked_m=masked_m, m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device),
        bf16_zero_workspace=_empty_bf16(device),
        M_logical=M, NE=NE, TOPK=topk,
        D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
        prologue=1,
    )
    aiter.mxfp4_moe_quant(
        a_input=hidden_states, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=atomic_output_buf,
        NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
    )
    padded_rows = ((max_sorted + 31) // 32) * 32
    cols = D_HIDDEN // 32
    a_scale_sorted_shuffled = torch.empty(
        (padded_rows * cols * 2,), device=device, dtype=torch.uint8)
    aiter.mxfp4_moe_sort_scales(
        a_scale=a_scale,
        sorted_token_ids=sorted_token_ids,
        cumsum_tensor=cumsum_tensor,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER,
        MB=BM, max_sorted=max_sorted,
    )

    # ── gemm1 (swappable) ─────────────────────────────────────────────────
    inter_sorted_quant = torch.empty(
        (max_sorted, D_INTER // 2), device=device, dtype=torch.uint8)
    BM_MIN = 64
    inter_scale_cols = D_INTER // 32
    inter_scale_bytes = max_sorted * (1024 // BM_MIN) * 4
    inter_scale_rows = (inter_scale_bytes + inter_scale_cols - 1) // inter_scale_cols
    inter_scale_rows = (inter_scale_rows + 31) // 32 * 32
    inter_sorted_shuffled_scale = torch.empty(
        (inter_scale_rows, inter_scale_cols), device=device, dtype=torch.uint8)

    gemm1_backend(
        cumsum_tensor=cumsum_tensor,
        a_quant=a_quant,
        a_scale_sorted_shuffled=a_scale_sorted_shuffled,
        w1=w1, w1_scale=w1_scale,
        sorted_expert_ids=sorted_expert_ids,
        m_indices=m_indices,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        hidden_states=hidden_states,
        kernelName1=kernelName1,
    )

    # ── gemm2 (swappable, atomic) ─────────────────────────────────────────
    gemm2_backend(
        cumsum_tensor=cumsum_tensor,
        inter_sorted_quant=inter_sorted_quant,
        inter_sorted_shuffled_scale=inter_sorted_shuffled_scale,
        w2=w2, w2_scale=w2_scale,
        sorted_token_ids=sorted_token_ids,
        sorted_expert_ids=sorted_expert_ids,
        sorted_weights=sorted_weights,
        out_buf=atomic_output_buf,
        M=M, max_sorted=max_sorted,
        kernelName2=kernelName2,
    )
    return atomic_output_buf
