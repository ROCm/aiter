# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Helpers shared by the MoE Triton tests."""

import torch
import triton

from aiter import logger
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant
from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    ep_sort_routing,
    EpScatterGeometry,
    ExptData,
    RoutingData,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if ref.numel() == 0:
        return

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3

    # Compare reference values against obtained values.
    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert (
        ref.shape == tri.shape
    ), f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(
        inf_mask_ref, inf_mask_tri
    ), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        logger.info(
            "%s maximum relative error = %f (threshold = %f)",
            description,
            max_err,
            maxtol,
        )
        logger.info(
            "%s RMS relative error = %f (threshold = %f)", description, rms_err, rmstol
        )

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        logger.warning(
            "%d / %d mismatched elements (shape = %s) at coords %s",
            num_nonzero,
            rel_err.numel(),
            tuple(rel_err.shape),
            bad_idxs.tolist(),
        )

        bad_idxs = bad_idxs.unbind(-1)
        logger.warning("ref values: %s", ref[tuple(bad_idxs)].cpu())
        logger.warning("tri values: %s", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


# ---------------
# Shared helpers for GEMM1 SwiGLU tests against fp32 torch reference.
# Used by test_moe_gemm_a4w4, test_moe_gemm_a8w4, test_moe_gemm_a16w4.
# ---------------

E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    device="cuda",
)


def dequant_e2m1(q, s):
    """Unpack e2m1 nibbles and apply e8m0 scales."""
    v = torch.stack((E2M1[(q & 0xF).long()], E2M1[(q >> 4).long()]), -1).flatten(-2)
    return v * torch.exp2(s.float() - 127.0).repeat_interleave(32, -1)


def upcast_fp8(q, s):
    """Upcast fp8 e4m3 values with e8m0 scales."""
    return q.float() * torch.exp2(
        s.view(torch.uint8).float() - 127.0
    ).repeat_interleave(32, -1)[..., : q.shape[-1]]


def make_mxfp4_weight(E_LOCAL, n_out, n_in, seed, device):
    """Create (kernel_weight, kernel_scale, fp32_ref) for MXFP4 weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(E_LOCAL, n_out, n_in, generator=g, device=device, dtype=torch.bfloat16) / n_in**0.5
    q, s = mxfp4_quant(w.reshape(-1, n_in))
    q = q.view(E_LOCAL, n_out, n_in // 2)
    s = s.view(E_LOCAL, n_out, n_in // 32)
    ref = dequant_e2m1(q, s)
    wk = moe_weight_decode_view(moe_shuffle_weight(q.contiguous()).contiguous())
    sk = moe_shuffle_scale(s.contiguous()).transpose(-1, -2)
    return wk, sk, ref


def ep_routing(M, E, E_LOCAL, TOPK, block_m, ids, wts, device):
    """EP sort routing with given ids/wts. Returns (rd, gidx, sidx, gvalid, dst)."""
    emap = torch.full((E + 1,), -1, dtype=torch.int32, device=device)
    emap[0:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=device)
    n_gates = M * TOPK
    bufs = _compute_expt_data_internal(E_LOCAL, n_gates, block_m, device)
    geo = EpScatterGeometry(
        src_token_map=torch.arange(M, dtype=torch.int32, device=device),
        max_tokens_per_rank=M, peer_rows=n_gates,
    )
    hist, gidx, sidx, gscal, gvalid, dst = ep_sort_routing(
        wts, ids, emap, E_LOCAL,
        torch.full((1,), M, dtype=torch.int32, device=device),
        M, TOPK, n_gates, bufs, ep_scatter_geometry=geo,
    )
    h = hist[:E_LOCAL]
    rd = RoutingData(block_m, gscal, h, E_LOCAL, TOPK, ExptData(h, bufs[0], bufs[1], bufs[2]))
    return rd, gidx, sidx, gvalid, dst


def swiglu_ref(h):
    """fp32 SwiGLU: gate=even cols, up=odd cols."""
    return h[:, ::2] * torch.sigmoid(h[:, ::2]) * h[:, 1::2]


def gemm1_swiglu_torch_ref(xr, w1r, E_LOCAL, ids, gidx, n, TOPK):
    """fp32 reference for GEMM1 + SwiGLU. Returns (ref, tok, exp)."""
    gate = gidx.long()[:n]
    tok = gate // TOPK
    exp = ids.flatten().long()[gate]
    I = w1r.shape[1] // 2
    ref = torch.empty(n, I, device=xr.device)
    for e in exp.unique().tolist():
        sel = (exp == e).nonzero(as_tuple=True)[0]
        ref[sel] = swiglu_ref(xr[tok[sel]] @ w1r[e].T)
    return ref, tok, exp
