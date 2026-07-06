import torch
import triton

from aiter.ops.triton.moe.moe_routing.topk import topk
from aiter.ops.triton._triton_kernels.moe.moe_routing.minunique import _keepk_sort0
from aiter.ops.triton._triton_kernels.moe.moe_routing.routing import _combined_routing
from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    RoutingData,
    ExptData,
)
from aiter.ops.triton.utils._triton.arch_info import is_tdm_avail


def keepk_sort0(
    expt_scal,
    expt_indx,
    pop,
    hist,
    part,
    n_expts_tot,
    k,
    apply_softmax,
    HIST_BLOCK_M,
    apply_renorm=False,
    routed_scaling_factor=1.0,
):
    """Fusion-2 driver: (k+1) candidates + popularity -> kept-k (Vout/Iout) +
    post-drop histogram + cross-block prefix offsets. `hist`/`part` are PRE-ZEROED."""
    M, KP1 = expt_scal.shape
    dev = expt_scal.device
    KP1_PAD = max(2, triton.next_power_of_2(KP1))
    num_blocks = triton.cdiv(M, HIST_BLOCK_M)
    Vout = torch.empty((M, k), dtype=expt_scal.dtype, device=dev)
    Iout = torch.empty((M, k), dtype=torch.int16, device=dev)
    _keepk_sort0[(M,)](
        expt_scal,
        expt_indx,
        expt_scal.stride(0),
        pop,
        Vout,
        Iout,
        Vout.stride(0),
        hist,
        part,
        part.stride(0),
        part.stride(1),
        M,
        n_expts_tot,
        HIST_BLOCK_M=HIST_BLOCK_M,
        NUM_BLOCKS=num_blocks,
        KP1=KP1,
        K=k,
        KP1_PAD=KP1_PAD,
        APPLY_SOFTMAX=apply_softmax,
        APPLY_RENORM=apply_renorm,
        ROUTED_SCALING=routed_scaling_factor,
        num_warps=1,
    )
    return Vout, Iout


def _minunique_common(
    num_tokens,
    n_expts_tot,
    k,
    block_m,
    logits_device,
    logits_dtype,
    expt_scal,
    expt_indx,
    pop,
    hist,
    partials,
    HIST_BLOCK_M,
    apply_softmax,
    apply_renorm,
    routed_scaling_factor,
):
    """Shared Fusion-2 + Fusion-3 for both flat-topk and fused-scoring HERD."""
    expt_scal2, expt_indx2 = keepk_sort0(
        expt_scal,
        expt_indx,
        pop,
        hist,
        partials,
        n_expts_tot,
        k,
        apply_softmax=apply_softmax,
        HIST_BLOCK_M=HIST_BLOCK_M,
        apply_renorm=apply_renorm,
        routed_scaling_factor=routed_scaling_factor,
    )

    n_gates = num_tokens * k
    n_expts_act_pad = triton.next_power_of_2(k)
    dev = logits_device
    dtype = expt_scal2.dtype
    if n_gates <= 65536:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.uint16, device=dev)
    else:
        combined_indx = torch.empty(n_gates * 2, dtype=torch.int32, device=dev)
    topk_indx = combined_indx[:n_gates]
    gate_indx = combined_indx[n_gates:]
    gate_scal = torch.empty(n_gates, dtype=dtype, device=dev)
    token_offs_raw, token_offs_pad, block_pid_map, blocks1a, BLOCK_A, block_m_log2 = (
        _compute_expt_data_internal(n_expts_tot, n_gates, block_m, dev)
    )
    blocks1b = triton.cdiv(num_tokens, HIST_BLOCK_M)
    _combined_routing[(blocks1a + blocks1b,)](
        topk_indx,
        gate_indx,
        gate_scal,
        expt_scal2,
        expt_indx2,
        partials,
        partials.stride(0),
        partials.stride(1),
        n_gates,
        HIST_BLOCK_M,
        num_tokens % HIST_BLOCK_M == 0,
        k,
        n_expts_act_pad,
        hist,
        n_expts_tot,
        token_offs_raw,
        token_offs_pad,
        blocks1a,
        block_pid_map,
        block_pid_map.shape[0],
        block_m_log2,
        BLOCK_A=BLOCK_A,
        EQUAL_A=(hist.shape[0] == BLOCK_A),
        USE_TDM=is_tdm_avail(),
        num_warps=1,
    )
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    routing_data = RoutingData(block_m, gate_scal, hist, n_expts_tot, k, expt_data)
    return routing_data, topk_indx, gate_indx


def routing_minunique(logits, n_expts_act, *, sm_first=False):
    """Flat-topk min-unique routing. Mirror of `routing(..., score_mode=None)`."""
    num_tokens, n_expts_tot = logits.shape
    k = n_expts_act
    m = num_tokens * k
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    if sm_first:
        logits = torch.softmax(logits, dim=-1)
    HIST_BLOCK_M = 32
    num_blocks = triton.cdiv(num_tokens, HIST_BLOCK_M)

    z = torch.zeros(
        2 * n_expts_tot + num_blocks * n_expts_tot,
        dtype=torch.int32,
        device=logits.device,
    )
    pop = z[:n_expts_tot]
    hist = z[n_expts_tot : 2 * n_expts_tot]
    partials = z[2 * n_expts_tot :].view(num_blocks, n_expts_tot)

    expt_scal, expt_indx, _bitmatrix = topk(
        logits, k + 1, apply_softmax=False, HIST_BLOCK_M=HIST_BLOCK_M, pop_out=pop
    )

    return _minunique_common(
        num_tokens,
        n_expts_tot,
        k,
        block_m,
        logits.device,
        logits.dtype,
        expt_scal,
        expt_indx,
        pop,
        hist,
        partials,
        HIST_BLOCK_M,
        apply_softmax=(not sm_first),
        apply_renorm=False,
        routed_scaling_factor=1.0,
    )


def routing_minunique_fused(
    logits,
    n_expts_act,
    *,
    score_mode="sqrtsoftplus",
    bias=None,
    renorm=True,
    routed_scaling_factor=1.0,
):
    """Fused-scoring min-unique routing (sqrtsoftplus + bias + renorm)."""
    num_tokens, n_expts_tot = logits.shape
    k = n_expts_act
    m = num_tokens * k
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))
    HIST_BLOCK_M = 32
    num_blocks = triton.cdiv(num_tokens, HIST_BLOCK_M)

    z = torch.zeros(
        2 * n_expts_tot + num_blocks * n_expts_tot,
        dtype=torch.int32,
        device=logits.device,
    )
    pop = z[:n_expts_tot]
    hist = z[n_expts_tot : 2 * n_expts_tot]
    partials = z[2 * n_expts_tot :].view(num_blocks, n_expts_tot)

    expt_scal, expt_indx, _bitmatrix = topk(
        logits,
        k + 1,
        apply_softmax=False,
        score_mode=score_mode,
        bias=bias,
        renorm=False,
        routed_scaling_factor=1.0,
        HIST_BLOCK_M=HIST_BLOCK_M,
        pop_out=pop,
    )

    return _minunique_common(
        num_tokens,
        n_expts_tot,
        k,
        block_m,
        logits.device,
        logits.dtype,
        expt_scal,
        expt_indx,
        pop,
        hist,
        partials,
        HIST_BLOCK_M,
        apply_softmax=False,
        apply_renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
