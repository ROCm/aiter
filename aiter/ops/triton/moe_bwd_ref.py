# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end BF16 triton backward MoE reference (Tier 2).

A correctness-first, single-GPU reference MoE layer with a full custom backward,
mirroring SonicMoE's structure but built on aiter's triton grouped GEMMs so it
runs on gfx950. It is the AMD-triton analog of SonicMoE and the per-kernel
reference the opus backward port is validated against.

Compute core (triton): dgrad via aiter `gmm` (M-grouped), wgrad via `ptgmm`
(K-grouped), activation-backward via `_act_bwd_kernel`. Routing / permute /
combine / router-backward use torch (memory ops; kept simple and obviously
correct for a reference). Everything is BF16 with FP32 gradient accumulation.

Not tuned for performance. g1u1 (gate/up) experts only. Router: softmax-over-topk.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

from aiter.ops.triton.gmm import gmm, ptgmm

SONIC_SWIGLU = "SonicSwiglu"
_ACT_ID = {
    "No": 0,
    "Silu": 0,
    SONIC_SWIGLU: 0,
    "Gelu": 1,
    "Swiglu": 2,
    "Situv2": 3,
}
_ALPHA = 1.702
_SWIGLU_LIMIT = 7.0


# ---------------------------------------------------------------------------
# activation forward / backward (must stay mutually consistent)
# ---------------------------------------------------------------------------
def _act_fwd(act_input, act_type, inter_dim):
    """h = act(gate, up); act_input [M, 2I] -> [M, I]. Matches _act_bwd_kernel."""
    gate, up = act_input[:, :inter_dim], act_input[:, inter_dim:]
    if act_type == "Swiglu":
        g = gate.clamp(max=_SWIGLU_LIMIT)
        u = up.clamp(min=-_SWIGLU_LIMIT, max=_SWIGLU_LIMIT)
        return g * torch.sigmoid(_ALPHA * g) * (u + 1.0)
    if act_type == "Situv2":
        sg = torch.tanh(gate) * torch.sigmoid(gate)  # beta=1
        return sg * torch.tanh(up)  # linear_beta=1
    act = {
        "Silu": F.silu,
        SONIC_SWIGLU: F.silu,
        "Gelu": F.gelu,
        "No": lambda x: x,
    }[act_type]
    return act(gate) * up


@triton.jit
def _act_bwd_kernel(
    dh_ptr, ai_ptr, dai_ptr, M, I,
    s_dh_m, s_dh_i, s_ai_m, s_ai_n, s_dai_m, s_dai_n,
    ACT: tl.constexpr, LIMIT: tl.constexpr, ALPHA: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (rm[:, None] < M) & (ri[None, :] < I)
    dh = tl.load(dh_ptr + rm[:, None] * s_dh_m + ri[None, :] * s_dh_i, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(ai_ptr + rm[:, None] * s_ai_m + ri[None, :] * s_ai_n, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(ai_ptr + rm[:, None] * s_ai_m + (I + ri[None, :]) * s_ai_n, mask=mask, other=0.0).to(tl.float32)
    if ACT == 3:  # SiTUv2 beta=lb=1
        tg = libdevice.tanh(gate); su = tl.sigmoid(gate); tu = libdevice.tanh(up)
        dg = dh * tu * ((1.0 - tg * tg) * su + tg * su * (1.0 - su))
        du_ = dh * (tg * su) * (1.0 - tu * tu)
    elif ACT == 2:  # Swiglu +1 bias
        g = tl.minimum(gate, LIMIT)
        u = tl.maximum(tl.minimum(up, LIMIT), -LIMIT)
        s = tl.sigmoid(ALPHA * g); f = g * s
        dg = dh * (u + 1.0) * (s + g * s * (1.0 - s) * ALPHA) * tl.where(gate <= LIMIT, 1.0, 0.0)
        du_ = dh * f * tl.where((up >= -LIMIT) & (up <= LIMIT), 1.0, 0.0)
    elif ACT == 1:  # Gelu exact
        phi_c = 0.5 * (1.0 + libdevice.erf(gate * 0.7071067811865476))
        pdf = libdevice.exp(-0.5 * gate * gate) * 0.3989422804014327
        dg = dh * up * (phi_c + gate * pdf)
        du_ = dh * (gate * phi_c)
    else:  # Silu
        sig = tl.sigmoid(gate); silu = gate * sig
        dg = dh * up * (sig + gate * sig * (1.0 - sig))
        du_ = dh * silu
    tl.store(dai_ptr + rm[:, None] * s_dai_m + ri[None, :] * s_dai_n, dg, mask=mask)
    tl.store(dai_ptr + rm[:, None] * s_dai_m + (I + ri[None, :]) * s_dai_n, du_, mask=mask)


def act_bwd_triton(dh, act_input, act_type, swiglu_limit=None):
    """dh [M,I] + act_input [M,2I] (gate;up) -> d_act_input [M,2I]. g1u1 only."""
    M, I = dh.shape
    dai = torch.empty_like(act_input)
    limit = _SWIGLU_LIMIT if swiglu_limit is None else float(swiglu_limit)
    grid = (triton.cdiv(M, 64), triton.cdiv(I, 64))
    _act_bwd_kernel[grid](
        dh, act_input, dai, M, I,
        dh.stride(0), dh.stride(1), act_input.stride(0), act_input.stride(1),
        dai.stride(0), dai.stride(1),
        ACT=_ACT_ID[act_type], LIMIT=limit, ALPHA=_ALPHA, BLOCK_M=64, BLOCK_I=64,
    )
    return dai


@triton.jit
def _fixed_topk_gather_sum_kernel(
    src_ptr,
    token_routes_ptr,
    out_ptr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    block_h = tl.program_id(1)
    h = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        route = tl.load(token_routes_ptr + token * TOPK + k)
        acc += tl.load(src_ptr + route * H + h, mask=mask, other=0.0).to(
            tl.float32
        )
    tl.store(out_ptr + token * H + h, acc, mask=mask)


def fixed_topk_gather_sum_triton(src, token_routes):
    """Reduce fixed-top-k route gradients directly into a BF16 token gradient."""
    T, topk = token_routes.shape
    H = src.shape[1]
    out = torch.empty(T, H, device=src.device, dtype=src.dtype)
    block_h = 256
    _fixed_topk_gather_sum_kernel[(T, triton.cdiv(H, block_h))](
        src,
        token_routes,
        out,
        H=H,
        TOPK=topk,
        BLOCK_H=block_h,
    )
    return out


# ---------------------------------------------------------------------------
# grouped GEMM helpers (triton, via aiter gmm/ptgmm) — single source of truth,
# imported by both the autograd layer here and the backward benchmark.
# ---------------------------------------------------------------------------
def _lens_from_offs(offs):
    return (offs[1:] - offs[:-1]).to(torch.int32)


def _dgrad(dy_op, w, lens):
    """M-grouped: dy_op[M,K] @ w[G,K,N] -> [M,N]."""
    # The forward passes a layout-1 transposed view (stride[1] == 1), while
    # backward passes the natural contiguous weight. State that intent explicitly
    # so square K == N weights remain mathematically unambiguous.
    return gmm(dy_op, w, lens, trans_rhs=(w.stride(1) == 1))


def _wgrad(dy_op, a_op, lens):
    """K-grouped: dy_op_gᵀ @ a_op_g -> [G, out, in]; pass dy_op as a transposed
    view (layout-1) so it stays unambiguous when M == out."""
    return ptgmm(dy_op.transpose(0, 1), a_op, lens)


def dgrad_gmm(dy_op, w, offs):
    """dgrad taking per-expert offsets (offs[G+1]); wrapper over _dgrad."""
    return _dgrad(dy_op, w, _lens_from_offs(offs))


def wgrad_ptgmm(dy_op, a_op, offs):
    """wgrad taking per-expert offsets (offs[G+1]); wrapper over _wgrad."""
    return _wgrad(dy_op, a_op, _lens_from_offs(offs))


# ---------------------------------------------------------------------------
# routing metadata (torch; correctness-first)
# ---------------------------------------------------------------------------
def _build_routing(topk_ids, E):
    """topk_ids [T,topk] int -> (order, x_gather_idx, lens) sorting routes by expert."""
    _T, topk = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)          # expert per route
    order = torch.argsort(flat, stable=True)             # routes sorted by expert
    x_gather_idx = (order // topk).to(torch.int64)       # token of each sorted route
    lens = torch.bincount(flat, minlength=E).to(torch.int32)
    return order, x_gather_idx, lens


def _build_token_routes(x_gather_idx, T, topk):
    """Reverse route map used by the fixed-top-k gather-sum in backward."""
    return (
        torch.argsort(x_gather_idx.to(torch.int32), stable=True)
        .to(torch.int32)
        .reshape(T, topk)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# backend-agnostic forward / backward impls. The backward is parametrized by
# three callables so the same reference (routing / combine / router bwd in torch)
# can be driven by the triton kernels or the opus kernels:
#   dgrad(dy_op[M,K], w[E,K,N], lens)          -> [M,N]   (M-grouped)
#   wgrad(dy_op[M,P], a_op[M,Q], lens)         -> [E,P,Q] (K-grouped)
#   actbwd(dh[M,I], act_input[M,2I], act_type) -> [M,2I]
# ---------------------------------------------------------------------------
def _moe_ref_forward_impl(ctx, x, w1, w2, router_logits, topk, act_type, dgrad,
                          prepare=None):
    """x[T,H], w1[E,2I,H], w2[E,H,I], router_logits[T,E] -> out[T,H].
    Router = softmax-over-topk. Forward uses dgrad for both stage GEMMs.
    prepare(ctx,w1,w2,lens): optional hook to stash bwd metadata (dgrad tiling +
    transposed weights) produced once per step (moe-sort semantics)."""
    T, H = x.shape
    E, twoI, _ = w1.shape
    I = twoI // 2
    dtype = x.dtype
    topk_vals, topk_ids = router_logits.topk(topk, dim=-1)          # [T,topk]
    # SonicMoE keeps routing scores in FP32 through combine and router backward.
    topk_w = torch.softmax(topk_vals.float(), dim=-1)                # [T,topk] fp32

    order, x_gather_idx, lens = _build_routing(topk_ids, E)
    p_sorted = topk_w.reshape(-1)[order].contiguous()               # [M] fp32
    x_g = x[x_gather_idx].contiguous()                             # [M,H]

    # pass transposed weights as VIEWS (gmm 'transposed layout 1') so the
    # rhs stays unambiguous even when the last two dims are equal.
    act_input = dgrad(x_g, w1.transpose(1, 2), lens)              # [M,2I] = x_g @ W1ᵀ
    h = _act_fwd(act_input, act_type, I).to(dtype)                  # [M,I]
    y = dgrad(h, w2.transpose(1, 2), lens)                        # [M,H] = h @ W2ᵀ

    out = torch.zeros(T, H, device=x.device, dtype=torch.float32)
    out.index_add_(0, x_gather_idx, (y.float() * p_sorted.float()[:, None]))
    out = out.to(dtype)

    ctx.save_for_backward(x_g, w1, w2, act_input, h, y, p_sorted,
                          x_gather_idx, order, lens, topk_ids, topk_w)
    ctx.dims = (T, H, E, I, topk)
    ctx.act_type = act_type
    ctx.token_routes = _build_token_routes(x_gather_idx, T, topk)
    if prepare is not None:
        prepare(ctx, w1, w2, lens)
    return out


def _moe_ref_backward_impl(ctx, dout, dgrad, wgrad, actbwd, combine_bwd=None,
                           dx_scatter=None, router_bwd=None):
    (x_g, w1, w2, act_input, h, y, p_sorted, x_gather_idx, order, lens,
     topk_ids, topk_w) = ctx.saved_tensors
    T, H, E, _I, topk = ctx.dims
    act_type = ctx.act_type
    dtype = dout.dtype

    # combine bwd: dy = p·dout (broadcast token->routes); dp = <dout, y>
    if combine_bwd is not None:
        dy, dp_sorted = combine_bwd(dout, x_gather_idx, p_sorted, y)
    else:
        dout_routes = dout[x_gather_idx]                           # [M,H]
        dy = (dout_routes.float() * p_sorted.float()[:, None]).to(dtype)
        dp_sorted = (dout_routes.float() * y.float()).sum(-1)      # [M]

    # stage2
    dh = dgrad(dy, w2, lens)                                       # [M,I] = dy @ W2
    dW2 = wgrad(dy, h, lens)                                      # [E,H,I]

    # activation bwd
    d_act = actbwd(dh, act_input, act_type)                        # [M,2I]

    # stage1
    dA_route = dgrad(d_act, w1, lens)                             # [M,H] = d_act @ W1
    dW1 = wgrad(d_act, x_g, lens)                                  # [E,2I,H]

    # dx: sum topk route contribs back to token
    if dx_scatter is not None:
        dx = dx_scatter(dA_route, x_gather_idx, T).to(dtype)
    else:
        dx = torch.zeros(T, H, device=dout.device, dtype=torch.float32)
        dx.index_add_(0, x_gather_idx, dA_route.float())
        dx = dx.to(dtype)

    # router bwd: dp_sorted -> dp[T,topk] -> softmax-over-topk bwd -> dlogits[T,E]
    if router_bwd is not None:
        dlogits = router_bwd(dp_sorted, order, topk_ids, topk_w, T, topk, E).to(dtype)
    else:
        dp_flat = torch.empty_like(dp_sorted)
        dp_flat[order] = dp_sorted
        dp = dp_flat.reshape(T, topk)                              # [T,topk]
        pw = topk_w.float()
        dtopk_vals = pw * (dp - (dp * pw).sum(-1, keepdim=True))   # softmax jvp
        dlogits = torch.zeros(T, E, device=dout.device, dtype=torch.float32)
        dlogits.scatter_(1, topk_ids.to(torch.int64), dtopk_vals)
        dlogits = dlogits.to(dtype)

    return dx, dW1, dW2, dlogits, None, None


# ---------------------------------------------------------------------------
# autograd Function: full forward + custom triton backward
# ---------------------------------------------------------------------------
class TritonMoERefFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, router_logits, topk, act_type):
        return _moe_ref_forward_impl(ctx, x, w1, w2, router_logits, topk, act_type, _dgrad)

    @staticmethod
    def backward(ctx, dout):
        def dx_gather_sum(src, _gather, _T):
            return fixed_topk_gather_sum_triton(src, ctx.token_routes)

        return _moe_ref_backward_impl(
            ctx,
            dout,
            _dgrad,
            _wgrad,
            act_bwd_triton,
            dx_scatter=dx_gather_sum,
        )


def triton_moe_ref(x, w1, w2, router_logits, topk, act_type="Silu"):
    """Functional entry: BF16 triton MoE with full custom backward."""
    return TritonMoERefFunc.apply(x, w1, w2, router_logits, topk, act_type)


def triton_moe(x, w1, w2, router_w, topk, act_type=SONIC_SWIGLU):
    """Complete Sonic-style MoE, including router projection backward."""
    router_logits = F.linear(x, router_w)
    return triton_moe_ref(x, w1, w2, router_logits, topk, act_type)


# ---------------------------------------------------------------------------
# pure-torch reference (same math) for gradient validation
# ---------------------------------------------------------------------------
def torch_moe_ref(x, w1, w2, router_logits, topk, act_type="Silu"):
    """Differentiable pure-torch MoE (softmax-over-topk), for autograd golden."""
    T, H = x.shape
    _E, twoI, _ = w1.shape
    I = twoI // 2
    topk_vals, topk_ids = router_logits.topk(topk, dim=-1)
    topk_w = torch.softmax(topk_vals.float(), dim=-1)                  # [T,topk] fp32
    out = x.new_zeros(T, H)
    for t in range(T):
        acc = x.new_zeros(H)
        for k in range(topk):
            e = topk_ids[t, k]
            ai = x[t] @ w1[e].t()                                     # [2I]
            hgate = _act_fwd(ai[None, :], act_type, I)[0]             # [I]
            yk = hgate @ w2[e].t()                                    # [H]
            acc = acc + topk_w[t, k] * yk
        out[t] = acc
    return out


def torch_moe(x, w1, w2, router_w, topk, act_type=SONIC_SWIGLU):
    """Complete pure-Torch golden matching :func:`triton_moe`."""
    return torch_moe_ref(x, w1, w2, F.linear(x, router_w), topk, act_type)
