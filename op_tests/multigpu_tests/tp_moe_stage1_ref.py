"""Torch fp32 reference for TPMoEStage1.

Models BOTH the per-1x32 FP8 activation quantization and the MXFP4 weights,
so the residual against the kernel is dominated by MFMA accumulation order
rather than by an unmodelled quantization step.
"""

import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import (
    MxDtypeInt,
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    mxfp4_to_f32,
)


def per_1x32_fp8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Per-1x32 MX FP8 E4M3 quantize-then-dequantize, in fp32.

    Mirrors ``fused_mx_quant_moe_sort_kernel`` (csrc/kernels/quant_kernels.cu):
    ``row_scale = ceil_pow2(amax / 448)`` on gfx950 (RoundUp / OCP e4m3fn),
    then ``value * rcp(row_scale)`` cast to fp8. ``row_scale`` is an exact
    power of two so the reciprocal is exact and ``/`` matches ``* rcp``.
    """
    x = x.float()
    grouped = x.view(*x.shape[:-1], -1, 32)
    amax = grouped.abs().amax(dim=-1)
    e8m0 = f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    scale = e8m0_to_f32(e8m0).unsqueeze(-1)
    q = (grouped / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (q.float() * scale).view_as(x)


def build_mxfp4_w1(experts, inter_dim, model_dim, device, seed):
    """Return (w1_ref, w1_scale_ref, w1_shuffled, w1_scale_shuffled).

    ``w1_ref`` keeps the UNSHUFFLED [E, 2*inter, model_dim/2] fp4x2 layout for
    the reference; the shuffled pair is what the kernel consumes.
    """
    import aiter

    g = torch.Generator(device="cpu").manual_seed(seed)
    w1_bf16 = torch.randn((experts, 2 * inter_dim, model_dim), generator=g).to(
        device=device, dtype=dtypes.bf16
    ) * (model_dim**-0.25)
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_scale = quant(w1_bf16, quant_dtype=dtypes.fp4x2)
    w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, experts, True)
    return w1_q, w1_scale, w1_shuf, w1_scale_shuf


def dequant_w1_expert(w1_q, w1_scale, expert_id, inter_dim):
    """Dequantize one expert's UNSHUFFLED W1 into fp32 [2*inter_dim, model_dim]."""
    experts, rows, _packed_cols = w1_q.shape
    assert rows == 2 * inter_dim, (rows, inter_dim)
    # ``per_1x32_f4_quant`` flattens the leading dims before reducing, so the
    # scale comes back 2D as (E * 2*inter_dim, model_dim//32) even for a 3D
    # weight. Restore the expert axis before indexing: ``w1_scale[expert_id]``
    # on the 2D form yields a single (model_dim//32,) row that broadcasts
    # silently over all 2*inter_dim rows -- shape-correct and numerically wrong.
    scale_e = w1_scale.reshape(experts, rows, -1)[expert_id]
    w = mxfp4_to_f32(w1_q[expert_id])
    s = e8m0_to_f32(scale_e).repeat_interleave(32, dim=-1)
    return (w * s).float()


def reference_inter_row(x_row_f32, w1_deq, swiglu_limit):
    """One route's GEMM1 + clamp + SwiGLU, fp32. Returns [inter_dim]."""
    inter_dim = w1_deq.shape[0] // 2
    gate_up = w1_deq @ x_row_f32
    gate = gate_up[:inter_dim]
    up = gate_up[inter_dim:]
    if swiglu_limit and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def mx_scale_shuffle_idx(scale_n_pad: int, x: int, y: int) -> int:
    """Canonical MX scale shuffle address (csrc/include/mx_quant_utils.h:212-217)."""
    return (
        (x // 32 * scale_n_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )


def read_shuffled_scale(scale_tensor, n_rows: int, n_kgroups: int) -> torch.Tensor:
    """Un-shuffle the stage1 output scale into a plain [n_rows, n_kgroups] fp32."""
    flat = scale_tensor.reshape(-1).view(torch.uint8)
    scale_n_pad = int(scale_tensor.shape[-1])
    xs = torch.arange(n_rows, dtype=torch.int64).view(-1, 1)
    ys = torch.arange(n_kgroups, dtype=torch.int64).view(1, -1)
    idx = (
        (xs // 32 * scale_n_pad) * 32
        + (ys // 8) * 256
        + (ys % 4) * 64
        + (xs % 16) * 4
        + (ys % 8) // 4 * 2
        + (xs % 32) // 16
    ).to(flat.device)
    return e8m0_to_f32(flat[idx.reshape(-1)]).view(n_rows, n_kgroups)


def build_global_weights(experts, inter_global, model_dim, device, seed):
    """Build UNSHARDED bf16 W1/W2 and quantize to MXFP4.

    Scale shapes are 2D (get_torch_quant flattens the leading dims):
        w1_s: (experts * 2*inter_global, model_dim // 32)
        w2_s: (experts * model_dim,      inter_global // 32)
    """
    import aiter

    g = torch.Generator(device="cpu").manual_seed(seed)
    w1 = torch.randn((experts, 2 * inter_global, model_dim), generator=g).to(
        device=device, dtype=dtypes.bf16
    ) * (model_dim**-0.25)
    w2 = torch.randn((experts, model_dim, inter_global), generator=g).to(
        device=device, dtype=dtypes.bf16
    ) * (inter_global**-0.25)
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = quant(w2, quant_dtype=dtypes.fp4x2)
    return w1_q, w1_s, w2_q, w2_s


def shard_w1(w1_q, w1_s, tp_rank, tp_size, inter_global):
    """W1 is column-parallel: TP shards the N axis (2*inter_global).

    Take this rank's [start, start+I_rank) window out of BOTH halves.
    """
    experts = w1_q.shape[0]
    i_rank = inter_global // tp_size
    lo = tp_rank * i_rank

    def _sl(t):
        # torch.cat has no CUDA kernel for float8_e8m0fnu (and fp4x2 is exotic
        # too), so concatenate through a uint8 view -- both are 1-byte dtypes,
        # so the view is a pure reinterpret that preserves the shape.
        u = t.view(torch.uint8)
        return (
            torch.cat(
                (
                    u[:, lo : lo + i_rank],
                    u[:, inter_global + lo : inter_global + lo + i_rank],
                ),
                dim=1,
            )
            .contiguous()
            .view(t.dtype)
        )

    q = _sl(w1_q)
    s = _sl(w1_s.reshape(experts, 2 * inter_global, -1))
    return q, s.reshape(experts * 2 * i_rank, -1).contiguous()


def shard_w2(w2_q, w2_s, tp_rank, tp_size, inter_global, model_dim):
    """W2 is row-parallel: TP shards the contraction axis (inter_global).

    The fp4x2 payload packs two values per byte so its last dim is halved;
    the scale's last dim is inter/32.
    """
    experts = w2_q.shape[0]
    i_rank = inter_global // tp_size
    lo = tp_rank * i_rank
    q = w2_q[:, :, lo // 2 : (lo + i_rank) // 2].contiguous()
    s = w2_s.reshape(experts, model_dim, -1)[:, :, lo // 32 : (lo + i_rank) // 32]
    return q, s.reshape(experts * model_dim, -1).contiguous()


def reference_full_moe(x_g_bf16, ids_g, wts_g, w1_q, w1_s, w2_q, w2_s, swiglu_limit):
    """Full unsharded MoE in fp32, modelling both activation quantizations."""
    m, model_dim = x_g_bf16.shape
    inter_global = w1_q.shape[1] // 2
    x_deq = per_1x32_fp8_quant_dequant(x_g_bf16.float())
    # w2_s is 2D (E * model_dim, inter_global//32); restore the expert axis once.
    w2_s3 = w2_s.reshape(w2_q.shape[0], model_dim, -1)
    out = torch.zeros((m, model_dim), dtype=torch.float32, device=x_deq.device)
    for e in torch.unique(ids_g).tolist():
        rows, slots = (ids_g == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w1_deq = dequant_w1_expert(w1_q, w1_s, e, inter_global)
        w2_deq = (
            mxfp4_to_f32(w2_q[e]) * e8m0_to_f32(w2_s3[e]).repeat_interleave(32, dim=-1)
        ).float()
        for r, s in zip(rows.tolist(), slots.tolist()):
            inter = reference_inter_row(x_deq[r], w1_deq, swiglu_limit)
            inter = per_1x32_fp8_quant_dequant(inter.unsqueeze(0)).squeeze(0)
            out[r] += (w2_deq @ inter) * float(wts_g[r, s].item())
    return out
