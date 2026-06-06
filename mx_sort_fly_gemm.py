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


def _fly_ptr(t):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _fly_gemm2(*, cumsum_tensor, inter_sorted_quant, inter_sorted_shuffled_scale,
               w2, w2_scale, sorted_token_ids, sorted_expert_ids, sorted_weights,
               out_buf, M, max_sorted, kernelName2):
    """Fresh FlyDSL gemm2 backend (drop-in for mxfp4_moe_gemm2_a4w4, atomic).

    Consumes the *exact* HIP gemm2 inputs (a16w4 w2, make_preshuffle scales,
    sorted inter A_q, sorted ids/weights); validated cos 0.9999 vs HIP.
    """
    from aiter.ops.flydsl.kernels.mxfp4_a4w4_gemm import compile_mxfp4_gemm2_a4w4

    NE = w2.shape[0]
    D_HIDDEN = w2.shape[1]            # N_OUT
    D_INTER = w2.shape[2] * 2         # K (w2 [E, D_HIDDEN, D_INTER//2] uint8)
    topk = int(kernelName2.split("TOPK")[1].split("_")[0])
    BM = int(kernelName2.split("_BM")[1].split("_")[0])
    launcher = compile_mxfp4_gemm2_a4w4(
        experts=NE, model_dim=D_HIDDEN, inter_dim=D_INTER, topk=topk, BM=BM
    )
    total_m_blocks = max_sorted // BM
    launcher(
        _fly_ptr(out_buf), _fly_ptr(inter_sorted_quant), _fly_ptr(w2),
        _fly_ptr(inter_sorted_shuffled_scale), _fly_ptr(w2_scale),
        _fly_ptr(sorted_token_ids), _fly_ptr(sorted_expert_ids),
        _fly_ptr(sorted_weights), _fly_ptr(cumsum_tensor),
        int(M), int(max_sorted), int(total_m_blocks),
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

    # codegen'd HIP kernel names for this shape (atomic, no scatter_reduce).
    # BM=16 has no separate-quant gemm1 — only the inline-quant variant exists,
    # so the small-M path fuses quant into gemm1 (prologue=0, no sort_scales).
    inline_quant = (BM == 16)
    if inline_quant:
        kernelName1 = (
            f"mxfp4_moe_g1_a4w4_NE{NE}_H{D_HIDDEN}_E{D_INTER}_BM16_INLINEQUANT"
        )
    else:
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

    # atomic gemm2 output buffer (zero-init'd via bf16_zero_out in sort/quant)
    atomic_output_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    # ── sort (+ quant + sort_scales for threestage; gemm1 fuses quant for
    #    inline-quant). mirrors aiter.fused_moe._mxfp4_moe_run. ─────────────
    if inline_quant:
        aiter.mxfp4_moe_sort(
            topk_ids=topk_ids, topk_weight=topk_weight,
            sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
            cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights,
            masked_m=masked_m, m_indices=m_indices,
            bf16_zero_out=atomic_output_buf,
            bf16_zero_workspace=_empty_bf16(device),
            M_logical=M, NE=NE, TOPK=topk,
            D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
            prologue=0,
        )
        a_scale_sorted_shuffled = _empty_u8(device)
    else:
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


# ════════════════════════════════════════════════════════════════════════
# FlyDSL gemm path: mxfp4 sort kernels + FlyDSL a4w4 gemm1/gemm2 (reads mx_w).
# ════════════════════════════════════════════════════════════════════════
def mx_sort_fly_gemm1_gemm2_flydsl(
    hidden_states,
    w1,            # fly_w1 = shuffle_weight(w1_qt, (16,16))   (FlyDSL separated layout)
    w2,            # fly_w2 = shuffle_weight(w2_qt, (16,16))
    topk_ids,
    topk_weight,
    topk,
    *,
    w1_scale=None, # e8m0_shuffle(w1_scale)
    w2_scale=None, # e8m0_shuffle(w2_scale)
    large_m_threshold=2048,
):
    """mxfp4 routing sort + mxfp4 quant, then FlyDSL stage1/stage2 a4w4 gemm.

    sort-related kernels are mx_fn's (threestage routing sort + activation quant);
    the activation scale is re-sorted into the FlyDSL tile layout via
    ``moe_mxfp4_sort`` (matching test_flydsl_moe_a4w4.py). The FlyDSL gemm reads
    the FlyDSL preshuffle of the weights (``fly_w`` = shuffle_weight(16,16)) which
    is the only layout the tested FlyDSL a4w4 path supports for w1.

    Per-M regime (mirrors mx_fn's BM/mode policy):
      * small/mid M  -> BM=32 sort, gemm1 t32x128, gemm2 t32x256 ATOMIC.
      * large M      -> BM=128 sort, gemm1 t128x128, gemm2 t64x256 REDUCE (sbm128,
        persistent). FlyDSL reduce sums topk via torch internally; mxfp4's HIP
        scatter_reduce is not layout-compatible with FlyDSL stage2 output.
    """
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
    from aiter.utility.fp4_utils import moe_mxfp4_sort

    device = hidden_states.device
    w1u = w1.view(torch.uint8) if (w1.element_size() == 1 and w1.dtype != torch.uint8) else w1
    w2u = w2.view(torch.uint8) if (w2.element_size() == 1 and w2.dtype != torch.uint8) else w2
    NE = w1u.shape[0]
    D_HIDDEN = hidden_states.shape[1]
    D_INTER = w1u.shape[1] // 2
    M = hidden_states.shape[0]

    # ── per-M regime (BM must be in {32,128}: mxfp4 3stage sort support) ──
    if M >= large_m_threshold:
        # large M: BM=128 sort, gemm1 t128x128, gemm2 FLAT (per-sorted-row,
        # unweighted) -> mxfp4 HIP scatter_reduce does the topk weighted reduce.
        BM = 128
        g1_tm, g1_tn, g1_wpe, b_nt1 = 128, 128, 2, 2
        g2_tm, g2_tn, g2_sbm, g2_persist, b_nt2 = 64, 256, 128, True, 2
        g2_flat, g2_fp4 = True, True
    else:
        BM = 32
        g1_tm, g1_tn, g1_wpe, b_nt1 = 32, 128, 2, 2
        g2_tm, g2_tn, g2_mode, g2_sbm, g2_persist, b_nt2 = 32, 256, "atomic", 0, None, 2
        g2_flat, g2_fp4 = False, False

    # ── mxfp4 routing sort + activation quant (mx_fn's sort-related kernels) ──
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
    out_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids, topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights, masked_m=masked_m, m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device), bf16_zero_workspace=_empty_bf16(device),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
        prologue=1,
    )
    aiter.mxfp4_moe_quant(
        a_input=hidden_states, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=out_buf, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
    )

    # FlyDSL expects num_valid_ids[0] = padded sorted-row count (== cumsum_tensor[0]).
    num_valid_ids = cumsum_tensor.repeat(2)

    # Re-sort the per-token e8m0 activation scale into FlyDSL's tile layout.
    a1 = a_quant.view(dtypes.fp4x2)
    a1_scale = moe_mxfp4_sort(
        a_scale.view(dtypes.fp8_e8m0).view(M, 1, -1),
        sorted_ids=sorted_token_ids,
        num_valid_ids=num_valid_ids,
        token_num=M,
        block_size=BM,
    )

    # ── FlyDSL gemm1 (a4w4 gate/up + SiLU·mul + fused fp4 requant) ─────────
    inter = flydsl_moe_stage1(
        a=a1, w1=w1u.view(dtypes.fp4x2),
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, topk=topk,
        tile_m=g1_tm, tile_n=g1_tn, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="fp4",
        w1_scale=w1_scale.view(dtypes.fp8_e8m0) if w1_scale is not None else None,
        a1_scale=a1_scale, sorted_weights=None, use_async_copy=True,
        waves_per_eu=g1_wpe, b_nt=b_nt1,
        gate_mode="separated",
    )
    inter_q, inter_scale = (inter[0], inter[1]) if isinstance(inter, tuple) else (inter, None)

    w2v = w2u.view(dtypes.fp4x2)
    w2sv = w2_scale.view(dtypes.fp8_e8m0) if w2_scale is not None else None
    if g2_fp4:
        # ── FlyDSL gemm2 FLAT-MXFP4: per-sorted-row fp4 + e8m0 (unweighted) ──
        flat_out_q = torch.empty(
            (max_sorted, D_HIDDEN // 2), dtype=torch.uint8, device=device)
        flat_out_scale = torch.empty(
            (max_sorted, D_HIDDEN // 32), dtype=torch.uint8, device=device)
        flydsl_moe_stage2(
            inter_states=inter_q, w2=w2v,
            sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids, topk=topk,
            out=flat_out_q.view(dtypes.fp4x2), out_scale=flat_out_scale,
            tile_m=g2_tm, tile_n=g2_tn, tile_k=256,
            a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
            w2_scale=w2sv, a2_scale=inter_scale, sorted_weights=None, b_nt=b_nt2,
            sort_block_m=g2_sbm, persist=g2_persist, flat_mxfp4=True,
        )
        # ── mx_fn's HIP scatter_reduce_q (fp4 input) weighted topk reduce ──
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=flat_out_q, flat_out_scale=flat_out_scale,
            reverse_sorted=reverse_sorted, sorted_weights=sorted_weights,
            out=out_buf, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
        )
    elif g2_flat:
        # ── FlyDSL gemm2 FLAT bf16 (per-sorted-row, unweighted) ────────────
        flat_out = torch.empty((max_sorted, D_HIDDEN), dtype=dtypes.bf16, device=device)
        flydsl_moe_stage2(
            inter_states=inter_q, w2=w2v,
            sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids, topk=topk, out=flat_out,
            tile_m=g2_tm, tile_n=g2_tn, tile_k=256,
            a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
            w2_scale=w2sv, a2_scale=inter_scale, sorted_weights=None, b_nt=b_nt2,
            sort_block_m=g2_sbm, persist=g2_persist, flat_output=True,
        )
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=flat_out, reverse_sorted=reverse_sorted,
            sorted_weights=sorted_weights, out=out_buf,
            NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
        )
    else:
        # ── FlyDSL gemm2 (a4w4 down-proj + atomic topk reduce) ─────────────
        flydsl_moe_stage2(
            inter_states=inter_q, w2=w2v,
            sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids, topk=topk, out=out_buf,
            tile_m=g2_tm, tile_n=g2_tn, tile_k=256,
            a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode=g2_mode,
            w2_scale=w2sv, a2_scale=inter_scale, sorted_weights=sorted_weights,
            b_nt=b_nt2, sort_block_m=g2_sbm, persist=g2_persist,
        )
    return out_buf
