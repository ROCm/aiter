# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus MoE backward ops (gfx950). M1: naive-correct K-grouped BF16 wgrad.

Serves both stage1 (dW1 = d_act^T @ x) and stage2 (dW2 = dy^T @ h) weight
gradients; rows must be grouped contiguously by expert (offsets = cumulative
per-expert route counts). Correctness-first; validated against the triton
reference (aiter.ops.triton.moe_bwd_ref) and the autograd golden."""
import torch
import torch.nn.functional as F
from torch import Tensor

from ...jit.core import compile_ops


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_wgrad_bf16", develop=True)
def _opus_moe_wgrad_bf16_raw(
    dy: Tensor, a: Tensor, expert_offsets: Tensor, dW: Tensor
) -> None: ...


def opus_moe_wgrad_bf16(dy: Tensor, a: Tensor, offs: Tensor, E: int) -> Tensor:
    """dW[e] = sum_{m in expert e} dy[m,:]^T (x) a[m,:].
    dy [M,P] bf16, a [M,Q] bf16, offs [E+1] cumulative -> dW [E,P,Q] fp32."""
    P, Q = dy.shape[1], a.shape[1]
    dW = torch.empty(E, P, Q, device=dy.device, dtype=torch.float32)
    _opus_moe_wgrad_bf16_raw(
        dy.contiguous(),
        a.contiguous(),
        offs.to(torch.int32).contiguous(),
        dW,
    )
    return dW


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_dgrad_bf16", develop=True)
def _opus_moe_dgrad_bf16_raw(
    dy: Tensor, w: Tensor, row_expert: Tensor, dh: Tensor
) -> None: ...


def build_padded_route_blocks(offs: Tensor, B_M: int, tensors):
    """approach-A padding for the fused MFMA grouped GEMM: pad each expert's rows
    to a B_M multiple so every B_M-row tile belongs to exactly one expert (opus
    stage2 route-block layout). offs[E+1] compact cumulative; each tensor [M,*]
    row-grouped by expert. Returns (padded tensors [Mp,*] zero-filled,
    sorted_expert_ids[num_blocks] int32, padded_row[M] compact->padded map, Mp)."""
    dev = offs.device
    E = offs.numel() - 1
    M = int(offs[-1])
    lens = offs[1:] - offs[:-1]
    padded_lens = ((lens + B_M - 1) // B_M) * B_M
    pad_offs = torch.zeros(E + 1, dtype=torch.long, device=dev)
    pad_offs[1:] = padded_lens.cumsum(0)
    Mp = int(pad_offs[-1])
    blocks_per_e = (padded_lens // B_M).to(torch.long)
    sorted_expert_ids = torch.repeat_interleave(
        torch.arange(E, device=dev), blocks_per_e
    ).to(torch.int32)
    e_of_row = torch.repeat_interleave(torch.arange(E, device=dev), lens)
    padded_row = pad_offs[e_of_row] + (torch.arange(M, device=dev) - offs[:-1][e_of_row])
    out = []
    for t in tensors:
        tp = torch.zeros((Mp,) + tuple(t.shape[1:]), dtype=t.dtype, device=dev)
        tp[padded_row] = t
        out.append(tp)
    return out, sorted_expert_ids, padded_row, Mp


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_transpose_pad_bf16", develop=True)
def _opus_moe_transpose_pad_bf16_raw(
    src: Tensor, col_to_m: Tensor, dst: Tensor
) -> None: ...


def build_wgrad_transpose_meta(offs: Tensor, B_K: int):
    """Amortizable wgrad metadata: pad each expert's routes to a (2*B_K)-multiple
    (>=2*B_K so a16w16 loops are even and >=2). Returns (col_to_m [Mp] i32 =
    compact row of each padded column (-1=pad), pad_offs [E+1] i32 per-expert
    padded cumulative, Mp). Built once per moe-sort, shared across wgrad stages."""
    dev = offs.device
    offs = offs.to(torch.long)
    E = offs.numel() - 1
    M = int(offs[-1])
    lens = offs[1:] - offs[:-1]
    unit = 2 * B_K
    padded_lens = torch.clamp(((lens + unit - 1) // unit) * unit, min=unit)
    pad_offs = torch.zeros(E + 1, dtype=torch.long, device=dev)
    pad_offs[1:] = padded_lens.cumsum(0)
    Mp = int(pad_offs[-1])
    e_of_row = torch.repeat_interleave(torch.arange(E, device=dev), lens)
    block_col = pad_offs[e_of_row] + (torch.arange(M, device=dev) - offs[:-1][e_of_row])
    col_to_m = torch.full((Mp,), -1, dtype=torch.int32, device=dev)
    col_to_m[block_col] = torch.arange(M, dtype=torch.int32, device=dev)
    return col_to_m, pad_offs.to(torch.int32), Mp


def transpose_pad(src: Tensor, col_to_m: Tensor, Mp: int) -> Tensor:
    """Compact [M,F] -> feature-major route-padded [F,Mp] via the fused kernel
    (one pass, coalesced writes). col_to_m from build_wgrad_transpose_meta."""
    F = src.shape[1]
    dst = torch.empty(F, Mp, device=src.device, dtype=src.dtype)
    _opus_moe_transpose_pad_bf16_raw(src.contiguous(), col_to_m, dst)
    return dst


def build_padded_transposed(dy: Tensor, a: Tensor, offs: Tensor, B_K: int):
    """wgrad plumbing: route-pad + transpose to feature-major so the contraction
    dim (routes) becomes contiguous. Returns (dyT [P,Mp], aT [Q,Mp], pad_offs
    [E+1] int32, Mp). Uses the fused pad+transpose kernel (metadata amortizable
    via build_wgrad_transpose_meta). wgrad NT-GEMM: dW[e]=dyT[:,r]@aT[:,r]^T."""
    col_to_m, pad_offs, Mp = build_wgrad_transpose_meta(offs, B_K)
    dyT = transpose_pad(dy, col_to_m, Mp)  # [P, Mp]
    aT = transpose_pad(a, col_to_m, Mp)    # [Q, Mp]
    return dyT, aT, pad_offs, Mp


def _row_expert_from_offs(offs: Tensor, E: int, device) -> Tensor:
    lens = (offs[1:] - offs[:-1]).to(torch.long)
    return torch.repeat_interleave(
        torch.arange(E, device=device, dtype=torch.int32), lens
    ).to(torch.int32)


def opus_moe_dgrad_bf16(dy: Tensor, w: Tensor, offs: Tensor) -> Tensor:
    """dh[m,:] = dy[m,:] @ w[expert(m)]. dy [M,K] bf16, w [E,K,N] bf16,
    offs [E+1] cumulative -> dh [M,N] bf16."""
    M, N, E = dy.shape[0], w.shape[2], w.shape[0]
    row_expert = _row_expert_from_offs(offs, E, dy.device)
    dh = torch.empty(M, N, device=dy.device, dtype=dy.dtype)
    _opus_moe_dgrad_bf16_raw(dy.contiguous(), w.contiguous(), row_expert, dh)
    return dh


SONIC_SWIGLU = "SonicSwiglu"
_ACT_ID = {
    "No": 0,
    "Silu": 0,
    SONIC_SWIGLU: 0,
    "Gelu": 1,
    "Swiglu": 2,
    "Situv2": 3,
}


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_act_bwd_bf16", develop=True)
def _opus_moe_act_bwd_bf16_raw(
    dh: Tensor, act_input: Tensor, d_act_input: Tensor, act: int, swiglu_limit: float
) -> None: ...


def opus_moe_act_bwd_bf16(dh: Tensor, act_input: Tensor, act_type: str, swiglu_limit=None) -> Tensor:
    """d[gate;up] from dh [M,I] + pre-act act_input [M,2I] -> [M,2I] (g1u1)."""
    d_act = torch.empty_like(act_input)
    _opus_moe_act_bwd_bf16_raw(
        dh.contiguous(), act_input.contiguous(), d_act,
        _ACT_ID[act_type], -1.0 if swiglu_limit is None else float(swiglu_limit),
    )
    return d_act


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_dgrad_mfma_bf16", develop=True)
def _opus_moe_dgrad_mfma_bf16_raw(
    dy: Tensor, w: Tensor, sorted_expert_ids: Tensor,
    block_m_start: Tensor, block_m_end: Tensor, dh: Tensor
) -> None: ...


def _build_dgrad_block_meta(offs: Tensor, B_M: int):
    """Build grouped-dgrad metadata and detect an equal-routes fast path.

    The offsets are already copied to the host here to construct the compact
    ragged tile list.  Reuse that same copy to return ``uniform_m`` instead of
    adding another device synchronization in the MoE forward path.
    """
    dev = offs.device
    o = offs.to(torch.int64).cpu().tolist()
    E = len(o) - 1
    e_ids, bms, bme = [], [], []
    for e in range(E):
        start, end = o[e], o[e + 1]
        s = start
        while s < end:
            e_ids.append(e); bms.append(s); bme.append(end)
            s += B_M
    meta = torch.tensor(e_ids + bms + bme, dtype=torch.int32, device=dev)
    n = len(e_ids)
    lens = [o[e + 1] - o[e] for e in range(E)]
    uniform_m = lens[0] if lens and all(m == lens[0] for m in lens) else None
    return meta[:n], meta[n:2 * n], meta[2 * n:], uniform_m


def build_dgrad_block_meta(offs: Tensor, B_M: int):
    """Compact grouped-dgrad tiling metadata (no operand padding). Each B_M row
    block covers COMPACT rows [start, expert_end) of one expert. offs[E+1]
    cumulative -> (sorted_expert_ids, block_m_start, block_m_end) [num_blocks] i32.
    Computed on host (E is tiny; ~10 GPU-op launches cost >150us here) then
    uploaded once."""
    seid, bms, bme, _uniform_m = _build_dgrad_block_meta(offs, B_M)
    return seid, bms, bme


def opus_moe_dgrad_mfma_bf16(dy: Tensor, w: Tensor, offs: Tensor, B_M: int = 128) -> Tensor:
    """Fused MFMA grouped dgrad. dh = dy @ W[e] contracting over the forward
    output dim. dy [M, K=fwd_out] bf16 (compact, expert-grouped), w [E, fwd_out,
    fwd_in] bf16 (forward weight), offs [E+1] compact cumulative. Returns compact
    dh [M, fwd_in] bf16. Compact tiling (no operand padding)."""
    M, fwd_in = dy.shape[0], w.shape[2]
    seid, bms, bme = build_dgrad_block_meta(offs, B_M)
    w_bnk = w.transpose(1, 2).contiguous()  # [E, fwd_in(N), fwd_out(K)]
    dh = torch.empty(M, fwd_in, device=dy.device, dtype=torch.bfloat16)
    _opus_moe_dgrad_mfma_bf16_raw(dy.contiguous(), w_bnk, seid, bms, bme, dh)
    return dh


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_wgrad_mfma_bf16", develop=True)
def _opus_moe_wgrad_mfma_bf16_raw(
    dyT: Tensor, aT: Tensor, pad_offs: Tensor, dW: Tensor
) -> None: ...


def opus_moe_wgrad_mfma_bf16(dy: Tensor, a: Tensor, offs: Tensor, E: int, B_K: int = 32) -> Tensor:
    """Fused MFMA grouped wgrad. dW[e] = dy_e^T @ a_e (contract routes).
    dy [M,P] bf16, a [M,Q] bf16 (compact, expert-grouped), offs [E+1]. Returns
    dW [E,P,Q] fp32. Transposes+pads internally (build_padded_transposed)."""
    P, Q = dy.shape[1], a.shape[1]
    dyT, aT, pad_offs, _Mp = build_padded_transposed(dy, a, offs, B_K)
    dW = torch.empty(E, P, Q, device=dy.device, dtype=torch.float32)
    _opus_moe_wgrad_mfma_bf16_raw(dyT, aT, pad_offs, dW)
    return dW


# ---------------------------------------------------------------------------
# M4: prepared/kernel-only entry points. In production the padded route-block
# layout (dgrad) and per-expert routing metadata are produced ONCE by moe
# sorting and shared across fwd/dgrad/wgrad, so the padding/transpose torch
# prep in the convenience wrappers above is not a per-call cost. These take
# precomputed inputs and are the fast path (opus kernels beat triton here).
# ---------------------------------------------------------------------------
def opus_moe_dgrad_mfma_prepared(dy: Tensor, w_bnk: Tensor, sorted_expert_ids: Tensor,
                                 block_m_start: Tensor, block_m_end: Tensor,
                                 out: Tensor) -> Tensor:
    """dgrad kernel-only (compact): dy [M,K], w_bnk [E,N,K] (=forward weight
    transposed), sorted_expert_ids/block_m_start/block_m_end [num_blocks], out [M,N]."""
    _opus_moe_dgrad_mfma_bf16_raw(dy, w_bnk, sorted_expert_ids, block_m_start, block_m_end, out)
    return out


def _mono_dgrad_shape_ok(dy: Tensor, w_bnk: Tensor, uniform_m: int) -> bool:
    """Whether the gfx950 8-wave mono kid 1400 can serve this uniform GMM."""
    E, N, K = w_bnk.shape
    return (
        uniform_m > 0
        and dy.shape == (E * uniform_m, K)
        and K >= 128
        and K % 64 == 0
        and N % 256 == 0
    )


def opus_moe_dgrad_uniform_prepared(
    dy: Tensor, w_bnk: Tensor, uniform_m: int, out: Tensor
) -> Tensor:
    """Equal-routes dgrad through the mature gfx950 8-wave mono GEMM.

    Routing has already made every expert's rows contiguous.  When all experts
    own the same number of rows, the grouped problem is exactly a zero-copy
    strided-batched GEMM: ``[E,M,K] @ [E,N,K]^T -> [E,M,N]``.  Kid 1400 uses
    the 192x256x64, eight-wave pipeline; ragged routing must keep using
    :func:`opus_moe_dgrad_mfma_prepared`.
    """
    if not _mono_dgrad_shape_ok(dy, w_bnk, uniform_m):
        raise ValueError(
            "opus_moe_dgrad_uniform_prepared: shape is unsupported by mono "
            f"kid 1400 (dy={tuple(dy.shape)}, w={tuple(w_bnk.shape)}, "
            f"uniform_m={uniform_m})"
        )
    from .gemm_op_a16w16 import opus_gemm_a16w16_tune

    E, N, K = w_bnk.shape
    opus_gemm_a16w16_tune(
        dy.view(E, uniform_m, K),
        w_bnk,
        out.view(E, uniform_m, N),
        bias=None,
        kernelId=1400,
        splitK=0,
    )
    return out


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_wgrad_tn_bf16", develop=True)
def _opus_moe_wgrad_tn_bf16_raw(
    dy: Tensor, a: Tensor, offs: Tensor, dW: Tensor
) -> None: ...


def opus_moe_wgrad_tn_bf16(dy: Tensor, a: Tensor, offs: Tensor, E: int) -> Tensor:
    """Full-TN grouped wgrad. dW[e]=dy_e^T@a_e from NATURAL compact dy[M,P]/a[M,Q]
    (no transpose, no padding). offs [E+1] cumulative. Returns dW [E,P,Q] bf16
    (fp32 mfma accumulate, bf16 store -- matches triton ptgmm; halves write).
    P,Q must be multiples of 32."""
    P, Q = dy.shape[1], a.shape[1]
    dW = torch.empty(E, P, Q, device=dy.device, dtype=torch.bfloat16)
    _opus_moe_wgrad_tn_bf16_raw(
        dy.contiguous(), a.contiguous(), offs.to(torch.int32).contiguous(), dW)
    return dW


def opus_moe_wgrad_mfma_prepared(dyT: Tensor, aT: Tensor, pad_offs: Tensor, dW: Tensor) -> Tensor:
    """wgrad kernel-only: dyT [P,Mp], aT [Q,Mp] (feature-major, route-padded),
    pad_offs [E+1] i32, dW [E,P,Q] fp32."""
    _opus_moe_wgrad_mfma_bf16_raw(dyT, aT, pad_offs, dW)
    return dW


# ---------------------------------------------------------------------------
# M6: opus-backend expert autograd layer. Reuses the reference structure
# (routing / combine / router bwd in torch) from aiter.ops.triton.moe_bwd_ref,
# swapping the three compute callables (dgrad / wgrad / act-bwd) to the fused
# opus MFMA kernels. Forward and backward both run on opus and are validated
# against the pure-torch autograd implementation in moe_bwd_ref.py.
# ---------------------------------------------------------------------------
def _offs_from_lens(lens: Tensor) -> Tensor:
    offs = torch.zeros(lens.numel() + 1, dtype=torch.int32, device=lens.device)
    offs[1:] = lens.to(torch.int32).cumsum(0)
    return offs


def _opus_dgrad(dy_op: Tensor, w: Tensor, lens: Tensor) -> Tensor:
    """M-grouped dgrad; w [E,K,N] (K=contract). a16w16 NT (transposed w = B
    contiguous); raw-TN dgrad was slower (B=w strided read)."""
    return opus_moe_dgrad_mfma_bf16(dy_op.contiguous(), w.contiguous(), _offs_from_lens(lens))


def _opus_wgrad(dy_op: Tensor, a_op: Tensor, lens: Tensor) -> Tensor:
    """K-grouped wgrad -> [E,P,Q] fp32. Full-TN (natural compact, no transpose)."""
    return opus_moe_wgrad_tn_bf16(dy_op, a_op, _offs_from_lens(lens), lens.numel())


def _opus_actbwd(dh: Tensor, act_input: Tensor, act_type: str) -> Tensor:
    return opus_moe_act_bwd_bf16(dh.to(torch.bfloat16), act_input, act_type)


# --- M5 R6: combine backward (opus kernels) ---
@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_combine_bwd_bf16", develop=True)
def _opus_moe_combine_bwd_bf16_raw(
    dout: Tensor, gather: Tensor, p: Tensor, y: Tensor, dy: Tensor, dp: Tensor
) -> None: ...


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_scatter_add_bf16", develop=True)
def _opus_moe_scatter_add_bf16_raw(
    src: Tensor, gather: Tensor, dst: Tensor
) -> None: ...


def opus_moe_combine_bwd_bf16(dout: Tensor, gather: Tensor, p_sorted: Tensor, y: Tensor):
    """dy[m,:]=p[m]*dout[gather[m],:]; dp[m]=<dout[gather[m],:],y[m,:]>.
    p_sorted is FP32, matching SonicMoE routing-score precision. Returns
    (dy [M,H] bf16, dp [M] fp32)."""
    assert p_sorted.dtype == torch.float32
    M, H = y.shape
    dy = torch.empty(M, H, device=y.device, dtype=torch.bfloat16)
    dp = torch.empty(M, device=y.device, dtype=torch.float32)
    _opus_moe_combine_bwd_bf16_raw(
        dout.contiguous(), gather.to(torch.int32).contiguous(),
        p_sorted.contiguous(), y.contiguous(), dy, dp)
    return dy, dp


def opus_moe_scatter_add_bf16(src: Tensor, gather: Tensor, T: int) -> Tensor:
    """dst[gather[m],:] += src[m,:] (topk routes -> token). Returns dst [T,H] fp32."""
    H = src.shape[1]
    dst = torch.zeros(T, H, device=src.device, dtype=torch.float32)
    _opus_moe_scatter_add_bf16_raw(src.contiguous(), gather.to(torch.int32).contiguous(), dst)
    return dst


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_gather_sum_bf16", develop=True)
def _opus_moe_gather_sum_bf16_raw(
    src: Tensor, token_routes: Tensor, dst: Tensor
) -> None: ...


def build_token_routes(gather: Tensor, T: int, topk: int) -> Tensor:
    """Fixed top-k reverse map for the dx gather-sum: token_routes[t,k] = the
    compact route index of token t's k-th selection. Built once per moe-sort
    (stable argsort groups a token's topk routes contiguously). gather [M] i32."""
    order_by_token = torch.argsort(gather.to(torch.int32), stable=True)
    return order_by_token.to(torch.int32).reshape(T, topk).contiguous()


def opus_moe_gather_sum_bf16(src: Tensor, token_routes: Tensor, T: int) -> Tensor:
    """Deterministic dx (no atomics): dst[t,:] = sum over token t's topk routes
    of src. Accumulation is FP32 and the result is stored as BF16, matching the
    input-gradient dtype without a separate full-tensor cast."""
    H = src.shape[1]
    dst = torch.empty(T, H, device=src.device, dtype=torch.bfloat16)
    _opus_moe_gather_sum_bf16_raw(src.contiguous(), token_routes, dst)
    return dst


# --- M5 R7: router backward (opus kernel) ---
@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_router_bwd_bf16", develop=True)
def _opus_moe_router_bwd_bf16_raw(
    dp: Tensor, topk_w: Tensor, topk_ids: Tensor, dlogits: Tensor
) -> None: ...


@compile_ops("module_moe_opus_bwd", fc_name="opus_moe_router_bwd_sigmoid_bf16", develop=True)
def _opus_moe_router_bwd_sigmoid_bf16_raw(
    dp: Tensor, logits: Tensor, topk_ids: Tensor, dlogits: Tensor, renorm: int
) -> None: ...


def opus_moe_router_bwd_bf16(dp_sorted: Tensor, order: Tensor, topk_ids: Tensor,
                             topk_w: Tensor, T: int, topk: int, E: int,
                             scoring: str = "softmax", renorm: bool = True,
                             logits: Tensor = None) -> Tensor:
    """Router backward -> dlogits [T,E] fp32. dp_sorted [M] fp32 (route order) is
    un-sorted to [T,topk] (cheap torch scatter), then the opus kernel does the
    scoring Jacobian + scatter into dlogits.
    scoring="softmax": softmax-over-topk (== topk(softmax)+renorm; renorm ignored).
    scoring="sigmoid": per-expert sigmoid; needs `logits` [T,E]; renorm toggles
    the w=s/Σs Jacobian. Used by DeepSeek/Kimi routing."""
    dp_flat = torch.empty_like(dp_sorted)
    dp_flat[order] = dp_sorted
    dp = dp_flat.reshape(T, topk).contiguous()
    dlogits = torch.zeros(T, E, device=dp.device, dtype=torch.float32)
    ids = topk_ids.to(torch.int32).contiguous()
    if scoring == "sigmoid":
        assert logits is not None, "sigmoid router bwd needs logits [T,E]"
        _opus_moe_router_bwd_sigmoid_bf16_raw(
            dp, logits.float().contiguous(), ids, dlogits, int(renorm))
    else:
        assert topk_w.dtype == torch.float32
        _opus_moe_router_bwd_bf16_raw(dp, topk_w.contiguous(), ids, dlogits)
    return dlogits


_DGRAD_B_M = 128


class OpusMoERefFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, router_logits, topk, act_type):
        # dgrad tiling meta (routing-only) is built ONCE (moe-sort semantics) and
        # shared by all 4 grouped GEMMs (2 fwd + 2 bwd). Forward GEMMs contract the
        # feature dim -> they use the NATURAL weight as the a16w16 B operand (no
        # transpose); only backward needs the transposed weight, produced here once.
        from ...ops.triton.moe_bwd_ref import _act_fwd, _build_routing
        T, H = x.shape
        E, twoI, _ = w1.shape
        I = twoI // 2
        dtype = x.dtype
        topk_vals, topk_ids = router_logits.topk(topk, dim=-1)
        # Match SonicMoE: route scores stay FP32 through combine and router bwd.
        topk_w = torch.softmax(topk_vals.float(), dim=-1)
        order, x_gather_idx, lens = _build_routing(topk_ids, E)
        p_sorted = topk_w.reshape(-1)[order].contiguous()
        x_g = x[x_gather_idx].contiguous()

        offs = _offs_from_lens(lens)
        seid, bms, bme, uniform_m = _build_dgrad_block_meta(offs, _DGRAD_B_M)

        def fwd_gemm(dy, w_nat):  # dh[m,n]=sum_k dy[m,k]*w_nat[e,n,k]
            out = torch.empty(dy.shape[0], w_nat.shape[1], device=dy.device, dtype=torch.bfloat16)
            if uniform_m is not None and _mono_dgrad_shape_ok(dy, w_nat, uniform_m):
                return opus_moe_dgrad_uniform_prepared(dy, w_nat, uniform_m, out)
            return opus_moe_dgrad_mfma_prepared(dy, w_nat, seid, bms, bme, out)

        act_input = fwd_gemm(x_g, w1)                                  # [M,2I]=x_g@W1ᵀ
        h = _act_fwd(act_input, act_type, I).to(dtype)                 # [M,I]
        y = fwd_gemm(h, w2)                                            # [M,H]=h@W2ᵀ

        out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
        out.index_add_(0, x_gather_idx, (y.float() * p_sorted.float()[:, None]))
        out = out.to(dtype)

        ctx.save_for_backward(x_g, w1, w2, act_input, h, y, p_sorted,
                              x_gather_idx, order, lens, topk_ids, topk_w)
        ctx.dims = (T, H, E, I, topk)
        ctx.act_type = act_type
        ctx.dgrad_meta = (seid, bms, bme)
        ctx.uniform_m = uniform_m
        ctx.offs = offs
        # dx reverse map for the deterministic gather-sum (no atomics); built once
        # here (moe-sort semantics), reused in backward.
        ctx.token_routes = build_token_routes(x_gather_idx, T, topk)
        ctx.w1t = w1.transpose(1, 2).contiguous()
        ctx.w2t = w2.transpose(1, 2).contiguous()
        ctx.w2_ref = w2
        return out

    @staticmethod
    def backward(ctx, dout):
        from ...ops.triton.moe_bwd_ref import _moe_ref_backward_impl
        seid, bms, bme = ctx.dgrad_meta
        offs = ctx.offs
        w1t, w2t, w2_ref = ctx.w1t, ctx.w2t, ctx.w2_ref

        def dgrad_prepared(dy_op, w, lens):
            wt = w2t if w is w2_ref else w1t
            out = torch.empty(dy_op.shape[0], wt.shape[1],
                              device=dy_op.device, dtype=torch.bfloat16)
            if ctx.uniform_m is not None and _mono_dgrad_shape_ok(
                dy_op, wt, ctx.uniform_m
            ):
                return opus_moe_dgrad_uniform_prepared(
                    dy_op.contiguous(), wt, ctx.uniform_m, out
                )
            return opus_moe_dgrad_mfma_prepared(dy_op.contiguous(), wt, seid, bms, bme, out)

        # deterministic dx (no atomics): gather-sum over each token's topk routes
        def dx_gather_sum(src, gather, T):
            return opus_moe_gather_sum_bf16(src, ctx.token_routes, T)

        def wgrad_prepared(dy_op, a_op, _lens):
            return opus_moe_wgrad_tn_bf16(dy_op, a_op, offs, offs.numel() - 1)

        return _moe_ref_backward_impl(
            ctx, dout, dgrad_prepared, wgrad_prepared, _opus_actbwd,
            combine_bwd=opus_moe_combine_bwd_bf16, dx_scatter=dx_gather_sum,
            router_bwd=opus_moe_router_bwd_bf16)


def opus_moe_ref(x, w1, w2, router_logits, topk, act_type="Silu"):
    """Opus expert MoE ending at dlogits (g1u1, softmax-over-topk)."""
    return OpusMoERefFunc.apply(x, w1, w2, router_logits, topk, act_type)


def opus_moe(x, w1, w2, router_w, topk, act_type=SONIC_SWIGLU):
    """Complete Sonic-style MoE, including router projection backward."""
    router_logits = F.linear(x, router_w)
    return opus_moe_ref(x, w1, w2, router_logits, topk, act_type)
