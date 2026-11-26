# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import kda_fused_layer
from aiter.ops.kda_fused_layer import (
    KDALayerWeights,
    kda_layer_bwd,
    kda_layer_fwd,
    kda_layer_postprocess,
    kda_layer_preprocess,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Kimi-K3 KDA layer (ZhangDanyang-AMD/Lumen#35): 96 heads x 128, short conv
# width 4, hidden 7168, F-decay f_a [128, 7168] / f_b [12288, 128], output gate
# g_proj [12288, 7168].
K3_HEADS, K3_HEAD_DIM, K3_CONV, K3_HIDDEN = 96, 128, 4, 7168


# Kimi-K3 checkpoints store A_log padded to 128 entries (96 real heads, zeros after)
K3_A_LOG_LEN = 128


def make_weights(D, H, K, W, R, dtype):
    """Weights laid out like the Kimi-K3 checkpoint: projections in ``dtype``;
    conv, A_log, dt_bias and the norm weight in fp32."""

    def p(*shape, std, dt=dtype):
        return (torch.randn(*shape) * std).to(dt)

    f32 = torch.float32
    a_log = torch.zeros(max(H, K3_A_LOG_LEN))
    # distinct per-head decay rates so a head/channel mix-up shows
    a_log[:H] = torch.linspace(-0.5, 1.2, H)

    return KDALayerWeights(
        w_q=p(H * K, D, std=D**-0.5),
        w_k=p(H * K, D, std=D**-0.5),
        w_v=p(H * K, D, std=D**-0.5),
        conv_q=p(H * K, W, std=0.5, dt=f32),
        conv_k=p(H * K, W, std=0.5, dt=f32),
        conv_v=p(H * K, W, std=0.5, dt=f32),
        f_a=p(R, D, std=D**-0.5),
        f_b=p(H * K, R, std=R**-0.5),
        A_log=a_log,
        dt_bias=p(H * K, std=1.0, dt=f32),
        w_beta=p(H, D, std=D**-0.5),
        g_proj=p(H * K, D, std=D**-0.5),
        norm_w=1.0 + torch.randn(K) * 0.1,
        w_o=p(D, H * K, std=(H * K) ** -0.5),
    )


def recurrent_kda_ref(q, k, v, g, beta, scale):
    """Token-by-token gated delta rule in fp32 (independent of the chunk form)."""
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1])
    out = []
    for t in range(T):
        S = g[:, t].exp().unsqueeze(-1) * S
        kt = k[:, t]
        u = beta[:, t, :, None] * (v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S))
        S = S + kt.unsqueeze(-1) * u.unsqueeze(-2)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S) * scale)
    return torch.stack(out, 1)


def run_torch(dy, x, w, num_heads, lb=None, eps=1e-6):
    """Reference layer backward: recurrent fwd + autograd, fp32."""
    xs = x.detach().float().requires_grad_(True)
    ws = [t.detach().float().requires_grad_(True) for t in w.tensors()]
    wr = KDALayerWeights.from_tensors(ws)
    q, k, v, g, beta = kda_layer_preprocess(xs, wr, num_heads, eps, lb)
    o = recurrent_kda_ref(q, k, v, g, beta, q.shape[-1] ** -0.5)
    y = kda_layer_postprocess(o, xs, wr, eps)
    grads = torch.autograd.grad(y, [xs, *ws], dy.float())
    return grads[0], list(grads[1:])


def layer_flops(b, t, d, h, k, w, r, c):
    proj = 2 * b * t * d * h * k * 5 + 2 * b * t * d * (r + h) + 2 * b * t * r * h * k
    conv = 2 * b * t * h * k * 3 * w
    chunk = (
        2 * b * t * h * (c * k * 2 + c * k + 2 * k * k)
    )  # intra A_kk/A_qk, solve, state io
    return 3 * (proj + conv + chunk)  # bwd ~= 2x fwd, plus recomputed fwd


@benchmark()
def test_kda_layer_bwd(b, t, d, h, k, w, r, chunk, lb, dtype):
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    ref_dx, ref_dw = run_torch(dy, x, wt, h, lb)

    candidates = {
        "torch_chunk": lambda: kda_layer_bwd(
            dy, x, wt, h, chunk_size=chunk, backend="torch", lower_bound=lb
        ),
        # gfx950 bf16: FlyDSL GEMM epilogues for f_b/g_proj, FlyDSL l2norm,
        # #8741 RMSNorm; conv (#8726) and chunk (#8740) bwd are torch until
        # those land.
        "fused": lambda: kda_layer_bwd(
            dy, x, wt, h, chunk_size=chunk, backend="fused", lower_bound=lb
        ),
    }

    flops = layer_flops(b, t, d, h, k, w, r, chunk)
    nbytes = sum(p.numel() for p in wt.tensors()) * 2 * x.element_size()
    nbytes += (3 * b * t * d + 6 * b * t * h * k) * x.element_size()

    tol = 2e-2 if dtype == dtypes.fp32 else 5e-2
    names = ["dx"] + [f.name for f in KDALayerWeights.__dataclass_fields__.values()]
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        (dx, dw), us = run_perftest(fn, num_iters=3, num_warmup=1)
        worst = 0.0
        for gname, ref, out in zip(names, [ref_dx, *ref_dw], [dx, *dw.tensors()]):
            # grade each gradient against its own scale so a small one is not hidden
            s = ref.abs().max().clamp_min(1e-6)
            err = checkAllclose(
                (ref / s).to(dtypes.fp32),
                (out.float() / s).to(dtypes.fp32),
                rtol=tol,
                atol=tol,
                msg=f"{name}: d{gname}",
            )
            assert err == 0, f"{name}: d{gname} {err:.2%} of elements outside tolerance"
            worst = max(worst, err)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = worst
    return ret


# Longer blocks are graded against the torch_chunk path: the token-by-token
# oracle is too slow there.
ORACLE_MAX_TOKENS = 512


def run_torch_block(dy, x, w, num_heads, cu_seqlens=None, lb=None, eps=1e-6):
    """Reference block: recurrent forward output and all gradients, fp32.

    With ``cu_seqlens`` each packed sequence is run on its own.
    """
    xs = x.detach().float().requires_grad_(True)
    ws = [t.detach().float().requires_grad_(True) for t in w.tensors()]
    wr = KDALayerWeights.from_tensors(ws)
    bounds = [0, x.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    ys = []
    for s, e in itertools.pairwise(bounds):
        xseg = xs[:, s:e]
        q, k, v, g, beta = kda_layer_preprocess(xseg, wr, num_heads, eps, lb)
        o = recurrent_kda_ref(q, k, v, g, beta, q.shape[-1] ** -0.5)
        ys.append(kda_layer_postprocess(o, xseg, wr, eps))
    y = torch.cat(ys, 1)
    grads = torch.autograd.grad(y, [xs, *ws], dy.float())
    return y.detach(), grads


@benchmark()
def test_kda_block(b, t, nseq, d, h, k, w, r, chunk, lb, dtype):
    """End to end: kda_layer_fwd + autograd backward of the whole layer.

    ``nseq > 1`` packs ``nseq`` equal sequences into one row (b must be 1) and
    passes ``cu_seqlens`` through the layer.
    """
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    cu = None
    if nseq > 1:
        assert b == 1 and t % nseq == 0, "varlen rows pack equal sequences into b=1"
        cu = torch.arange(nseq + 1, dtype=torch.int32) * (t // nseq)

    def block(backend):
        def run():
            xs = x.detach().requires_grad_(True)
            ws = [p.detach().requires_grad_(True) for p in wt.tensors()]
            y = kda_layer_fwd(
                xs,
                KDALayerWeights.from_tensors(ws),
                h,
                chunk,
                backend=backend,
                cu_seqlens=cu,
                lower_bound=lb,
            )
            return y, torch.autograd.grad(y, [xs, *ws], dy)

        return run

    if b * t <= ORACLE_MAX_TOKENS:
        ref_name = "recurrent"
        ref_y, ref_g = run_torch_block(dy, x, wt, h, cu, lb)
    else:
        ref_name = "torch_chunk"
        ref_y, ref_g = block("torch")()

    flops = layer_flops(b, t, d, h, k, w, r, chunk) * 4 // 3  # fwd + recompute + bwd
    nbytes = sum(p.numel() for p in wt.tensors()) * 2 * x.element_size()
    nbytes += (3 * b * t * d + 6 * b * t * h * k) * x.element_size()
    tol = 2e-2 if dtype == dtypes.fp32 else 5e-2
    names = ["y", "dx"] + [
        f.name for f in KDALayerWeights.__dataclass_fields__.values()
    ]
    ret = {"gfx": get_gfx_runtime(), "ref": ref_name}
    for name in ("torch", "fused"):
        (y, grads), us = run_perftest(block(name), num_iters=3, num_warmup=1)
        worst = 0.0
        for gname, ref, out in zip(names, [ref_y, *ref_g], [y, *grads]):
            sc = ref.abs().max().clamp_min(1e-6)
            err = checkAllclose(
                (ref / sc).to(dtypes.fp32),
                (out.float() / sc).to(dtypes.fp32),
                rtol=tol,
                atol=tol,
                msg=f"block {name}: {gname}",
            )
            assert err == 0, f"block {name}: {gname} {err:.2%} outside tolerance"
            worst = max(worst, err)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = worst
    # which optional FlyDSL / Triton pieces the fused run actually used
    ret["fused pieces"] = " ".join(
        f"{key}={val}" for key, val in sorted(kda_fused_layer.LAST_PIECES.items())
    )
    return ret


@benchmark()
def test_kda_piece(piece, b, t, d, h, k, w, r, chunk, lb, dtype):
    """Fused layer backward with one stage on FlyDSL and the rest on Triton, so a
    wrong FlyDSL piece shows up on its own row."""
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    if b * t <= ORACLE_MAX_TOKENS:
        ref_name = "recurrent"
        ref_dx, ref_dw = run_torch(dy, x, wt, h, lb)
    else:
        ref_name = "torch_chunk"
        rdx, rdw = kda_layer_bwd(dy, x, wt, h, chunk, backend="torch", lower_bound=lb)
        ref_dx, ref_dw = rdx, rdw.tensors()
    choice = dict.fromkeys(kda_fused_layer.KDA_PIECE_DEFAULTS, "triton")
    choice[piece] = "flydsl"
    kda_fused_layer.set_kda_pieces(**choice)
    try:
        (dx, dw), us = run_perftest(
            lambda: kda_layer_bwd(dy, x, wt, h, chunk, backend="fused", lower_bound=lb),
            num_iters=3,
            num_warmup=1,
        )
        pieces = dict(kda_fused_layer.LAST_PIECES)
    finally:
        kda_fused_layer.set_kda_pieces(**dict.fromkeys(choice))
    tol = 2e-2 if dtype == dtypes.fp32 else 5e-2
    names = ["x"] + [f.name for f in KDALayerWeights.__dataclass_fields__.values()]
    worst, failed = 0.0, []
    for gname, ref, out in zip(names, [ref_dx, *ref_dw], [dx, *dw.tensors()]):
        sc = ref.abs().max().clamp_min(1e-6)
        err = checkAllclose(
            (ref / sc).to(dtypes.fp32),
            (out.float() / sc).to(dtypes.fp32),
            rtol=tol,
            atol=tol,
            msg=f"{piece}=flydsl: d{gname}",
        )
        if err:
            failed.append(f"d{gname}")
        worst = max(worst, err)
    ret = {
        "gfx": get_gfx_runtime(),
        "ref": ref_name,
        # the backend that actually served the stage (flydsl may be skipped)
        "used": pieces.get(piece),
        "skipped": pieces.get(f"{piece}_flydsl_skipped", "-"),
        "us": us,
        "err": worst,
        "failed grads": ",".join(failed) or "-",
    }
    # asserted in main() after the table, so every piece's row is reported
    return ret


@benchmark()
def test_kda_chunk_fwd_backends(b, t, h, k, lb, dtype):
    """Chunk forward backends on the inputs exactly as the layer builds them
    (q/k/v are channel-major views out of the short conv), against
    ``chunk_kda_torch``: isolates the chunk-forward integration from the rest."""
    a = torch.randn(b, t, 3 * h * k)
    conv_w = torch.randn(3 * h * k, 4) * 0.5
    c = kda_fused_layer._short_conv(a, conv_w)
    q, kk, v = (x.reshape(b, t, h, k) for x in c.split(h * k, -1))
    q = torch.nn.functional.normalize(q, dim=-1)
    kk = torch.nn.functional.normalize(kk, dim=-1)
    a_log = torch.zeros(K3_A_LOG_LEN)
    a_log[:h] = torch.linspace(-0.5, 1.2, h)
    g = kda_fused_layer.kda_gate(torch.randn(b, t, h, k), a_log, lb)
    beta = torch.sigmoid(torch.randn(b, t, h))
    scale = k**-0.5
    ref = kda_fused_layer.chunk_kda_torch(q, kk, v, g, beta, scale, 64)
    ret = {"gfx": get_gfx_runtime(), "q contiguous": q.is_contiguous()}
    for backend in ("triton", "flydsl"):
        try:
            fn = kda_fused_layer._PIECE_LOADERS[("chunk_fwd", backend)]()
        except ImportError:
            ret[f"{backend} err"] = "n/a"
            continue
        o, us = run_perftest(fn, q, kk, v, g, beta, scale, 64, num_iters=5)
        err = checkAllclose(
            ref, o.float(), rtol=2e-2, atol=2e-2, msg=f"chunk fwd {backend}"
        )
        assert err == 0, f"chunk fwd {backend}: {err:.2%} outside tolerance"
        ret[f"{backend} us"] = us
        ret[f"{backend} err"] = err
    return ret


KDA_STAGES = (
    "gate_proj",
    "gate_proj_bwd",
    "sigmoid_gate_proj",
    "l2norm_bwd",
    "rmsnorm_bwd",
    "chunk_bwd_triton",
    "chunk_bwd_flydsl",
)


def _stage_case(stage, m, h, k, d, r, lb, dtype):
    """(kernel fn, torch fn, inputs) for one GPU-only stage, or a skip reason.

    Both sides get the same (rounded) inputs; the torch side is fp32 math."""
    from aiter.ops import kda_fused_layer as L

    hk = h * k
    A_log = torch.zeros(max(h, K3_A_LOG_LEN))
    A_log[:h] = torch.linspace(-0.5, 1.2, h)
    if stage in ("gate_proj", "gate_proj_bwd", "sigmoid_gate_proj"):
        if dtype not in (dtypes.bf16, dtypes.fp16) or get_gfx_runtime() != "gfx950":
            return "FlyDSL GEMM epilogues need gfx950 and bf16/fp16"
        if any(n % 8 for n in (k, r, d)):
            return f"GEMM rows need multiples of 8 (k={k}, r={r}, d={d})"
        from aiter.ops.flydsl import kda_layer_bwd as fk

        lr = torch.randn(m, r)  # fp32, as the layer passes it
        f_b = (torch.randn(hk, r) * r**-0.5).to(dtype)
        dt_bias = torch.randn(hk)
        dg = torch.randn(m, hk)
        P = L._TorchProj
        if stage == "gate_proj":
            return (
                lambda: fk.kda_gate_proj(lr, f_b, dt_bias, A_log, h, lb),
                lambda: P.kda_gate_proj(lr, f_b, dt_bias, A_log, h, lb),
            )
        if stage == "gate_proj_bwd":

            def per_head(out):
                dz, dterm = out
                return dz, dterm.view(m, h, k).sum((0, 2))

            return (
                lambda: per_head(
                    fk.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, h, dg, lb)
                ),
                lambda: per_head(
                    P.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, h, dg, lb)
                ),
            )
        x = torch.randn(m, d).to(dtype)
        g_proj = (torch.randn(hk, d) * d**-0.5).to(dtype)
        aux = torch.randn(m, hk)
        return (
            lambda: fk.sigmoid_gate_proj(x, g_proj, aux),
            lambda: P.sigmoid_gate_proj(x, g_proj, aux),
        )
    if stage in ("l2norm_bwd", "rmsnorm_bwd") and k % 64:
        return f"FlyDSL row kernels need a head dim multiple of 64 (k={k})"
    if stage == "l2norm_bwd":
        from aiter.ops.flydsl import kda_layer_bwd as fk

        xx = torch.randn(m, h, k).to(dtype).float()
        dy = torch.randn(m, h, k).to(dtype).float()
        return (
            lambda: fk.l2norm_bwd(xx, dy, L.L2NORM_EPS),
            lambda: L._TorchOps.l2norm_bwd(xx, dy, L.L2NORM_EPS),
        )
    if stage == "rmsnorm_bwd":
        try:
            from aiter.ops.flydsl.gated_rmsnorm_bwd import gated_rmsnorm_bwd
        except ImportError as err:
            return f"#8741 missing: {err}"
        o = torch.randn(1, m, h, k)
        gp = torch.randn(1, m, h, k)
        w = 1.0 + torch.randn(k) * 0.1
        dy = torch.randn(1, m, h, k)
        return (
            lambda: gated_rmsnorm_bwd(o, gp, w, dy, 1e-6),
            lambda: L._TorchOps.gated_rmsnorm_bwd(o, gp, w, dy, 1e-6),
        )
    # chunk backward with the layer's fused prologue (raw q/k, raw gate, beta logit)
    backend = stage.rsplit("_", 1)[1]
    try:
        impl, fuses = L._PIECE_LOADERS[("chunk_bwd", backend)]()
    except ImportError as err:
        return f"{backend} chunk bwd missing: {err}"
    if backend == "flydsl" and dtype != dtypes.bf16:
        return "FlyDSL chunk bwd is bf16-only"
    q = torch.randn(1, m, h, k).to(dtype)
    kk = torch.randn(1, m, h, k).to(dtype)
    v = torch.randn(1, m, h, k).to(dtype)
    g = torch.randn(1, m, h, k)
    beta = torch.randn(1, m, h)
    do = torch.randn(1, m, h, k)
    kw = {
        "use_qk_l2norm_in_kernel": "qk_l2norm" in fuses,
        "use_beta_sigmoid_in_kernel": "beta_sigmoid" in fuses,
    }
    if not kw["use_qk_l2norm_in_kernel"]:
        q, kk = (torch.nn.functional.normalize(t.float(), dim=-1) for t in (q, kk))
    if not kw["use_beta_sigmoid_in_kernel"]:
        beta = torch.sigmoid(beta)
    if "gate" in fuses and (lb is None or L._accepts(impl, "lower_bound")):
        kw.update(use_gate_in_kernel=True, A_log=A_log[:h], dt_bias=torch.randn(hk))
        if lb is not None:
            kw["lower_bound"] = lb
    else:
        g = L.kda_gate(g, A_log, lb)
    extra = {}
    if L._has_param(impl, "compute_dtype"):
        extra["compute_dtype"] = L._compute_dtype_of(dtype)
    scale = k**-0.5

    def grads(out):
        return [t for t in out if t is not None]

    return (
        lambda: grads(impl(do, q, kk, v, g, beta, scale, 64, **kw, **extra)),
        lambda: grads(L.chunk_kda_bwd_torch(do, q, kk, v, g, beta, scale, 64, **kw)),
    )


@benchmark()
def test_kda_stage(stage, m, h, k, d, r, lb, dtype):
    """One GPU-only stage of the fused path vs its torch math on identical
    inputs, so a mismatch is pinned to that stage."""
    case = _stage_case(stage, m, h, k, d, r, lb, dtype)
    if isinstance(case, str):
        return {"gfx": get_gfx_runtime(), "status": f"skip: {case}", "failed": "-"}
    kernel, ref = case
    out, us = run_perftest(kernel, num_iters=3, num_warmup=1)
    want = ref()
    out = out if isinstance(out, (tuple, list)) else (out,)
    want = want if isinstance(want, (tuple, list)) else (want,)
    errs, failed = [], []
    for i, (o, rf) in enumerate(zip(out, want)):
        sc = rf.float().abs().max().clamp_min(1e-6)
        err = checkAllclose(
            (rf.float() / sc).to(dtypes.fp32),
            (o.float() / sc).to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"stage {stage}[{i}]",
        )
        errs.append(err)
        if err:
            failed.append(str(i))
    return {
        "gfx": get_gfx_runtime(),
        "status": "ran",
        "us": us,
        "err": max(errs),
        "failed": ",".join(failed) or "-",
    }


@benchmark()
def test_kda_chunk_bwd_backends(b, t, h, k, lb, dtype):
    """Chunk backward backends on channel-major q/k/v views (as the layer builds
    them) with the fused prologue, in fp32 and bf16, against chunk_kda_bwd_torch."""
    a = torch.randn(b, t, 3 * h * k)
    c = kda_fused_layer._short_conv(a, torch.randn(3 * h * k, 4) * 0.5)
    q, kk, v = (x.reshape(b, t, h, k) for x in c.split(h * k, -1))
    q, kk, v = (x.to(dtype) for x in (q, kk, v))
    a_log = torch.zeros(K3_A_LOG_LEN)
    a_log[:h] = torch.linspace(-0.5, 1.2, h)
    g = torch.randn(b, t, h, k)
    beta = torch.randn(b, t, h)
    do = torch.randn(b, t, h, k)
    kw = {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "A_log": a_log[:h],
        "dt_bias": torch.randn(h * k),
        "use_beta_sigmoid_in_kernel": True,
    }
    if lb is not None:
        kw["lower_bound"] = lb
    scale = k**-0.5
    ref = kda_fused_layer.chunk_kda_bwd_torch(do, q, kk, v, g, beta, scale, 64, **kw)
    ret = {"gfx": get_gfx_runtime(), "q contiguous": q.is_contiguous()}
    for backend in ("triton", "flydsl"):
        try:
            impl, _fuses = kda_fused_layer._PIECE_LOADERS[("chunk_bwd", backend)]()
        except ImportError:
            ret[f"{backend} err"] = "n/a"
            continue
        if backend == "flydsl" and dtype != dtypes.bf16:
            ret[f"{backend} err"] = "bf16 only"
            continue
        call_kw = dict(kw)
        if lb is not None and not kda_fused_layer._accepts(impl, "lower_bound"):
            ret[f"{backend} err"] = "no lower_bound"
            continue
        if kda_fused_layer._has_param(impl, "compute_dtype"):
            call_kw["compute_dtype"] = kda_fused_layer._compute_dtype_of(dtype)
        out, us = run_perftest(
            impl, do, q, kk, v, g, beta, scale, 64, **call_kw, num_iters=3
        )
        worst = 0.0
        for i, (o, rf) in enumerate(zip(out, ref)):
            if rf is None:
                continue
            sc = rf.float().abs().max().clamp_min(1e-6)
            err = checkAllclose(
                (rf.float() / sc).to(dtypes.fp32),
                (o.float() / sc).to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                msg=f"chunk bwd {backend}[{i}]",
            )
            assert err == 0, f"chunk bwd {backend}[{i}]: {err:.2%} outside tolerance"
            worst = max(worst, err)
        ret[f"{backend} us"] = us
        ret[f"{backend} err"] = worst
    return ret


@benchmark()
def test_kda_gproj_bwd_exact(m, d, h, k, dtype):
    """g_proj backward GEMMs (dx = dgp @ g_proj, nn; dW_g = dgp^T @ x, tn) on
    small-integer inputs, so the fp32 result is exact, into NaN-filled outputs:
    any wrong, racy or unwritten element shows up bit for bit."""
    from aiter.ops.flydsl import kda_layer_bwd as fk

    hk = h * k
    dgp = torch.randint(-3, 4, (m, hk)).to(dtype)
    x = torch.randint(-3, 4, (m, d)).to(dtype)
    g_proj = torch.randint(-3, 4, (hk, d)).to(dtype)
    want_dx = dgp.float() @ g_proj.float()
    want_dw = dgp.float().t() @ x.float()
    ret = {"gfx": get_gfx_runtime()}
    for name, kind, w, want in (
        ("dx_nn", "dgrad", g_proj, want_dx),
        ("dW_tn", "wgrad", x, want_dw),
    ):
        out = torch.full(want.shape, float("nan"))
        _, us = run_perftest(fk.proj_gemm, dgp, w, kind, out, num_iters=3)
        bad = (out != want) | out.isnan()
        n_bad = int(bad.sum())
        assert n_bad == 0, (
            f"{name}: {n_bad} of {out.numel()} elements differ "
            f"(max {(out - want).abs().nan_to_num(float('inf')).max().item()})"
        )
        ret[f"{name} us"] = us
        ret[f"{name} wrong"] = n_bad
    return ret


def run_torch_proj(lr, f_b, dt_bias, A_log, h, lb, dg, x, g_proj, aux):
    """Reference for the fused projections, fp32 (A_log: first h entries)."""
    with torch.enable_grad():
        zr = (lr.float() @ f_b.float().t()).requires_grad_(True)
        a_log = A_log[:h].float().requires_grad_(True)
        z = zr + dt_bias.float()
        a = a_log.exp().repeat_interleave(f_b.shape[0] // h)
        if lb is None:
            gate = -a * torch.nn.functional.softplus(z)
        else:
            gate = lb * torch.sigmoid(a * z)
        dz, da = torch.autograd.grad(gate, [zr, a_log], dg)
    # the kernel returns per-element dA_log terms; compare their per-head sums
    gate_out = torch.sigmoid(x.float() @ g_proj.float().t())
    aux16 = aux.to(g_proj.dtype).float()
    return {
        "gate": gate.detach(),
        "gate_bwd": (dz, da),
        "sigmoid_mul": gate_out * aux,
        # the layer feeds dgp to these GEMMs in the weight dtype; grade them on
        # the same rounded input so only the GEMM itself is measured
        "gproj_bwd": (aux16 @ g_proj.float(), aux16.t() @ x.float()),
    }


@benchmark()
def test_kda_proj(m, d, h, k, r, lb, dtype):
    from aiter.ops.flydsl import kda_layer_bwd as fk

    hk = h * k
    lr = torch.randn(m, r).to(dtype)
    f_b = (torch.randn(hk, r) * r**-0.5).to(dtype)
    dt_bias = torch.randn(hk)
    A_log = torch.zeros(max(h, K3_A_LOG_LEN))
    A_log[:h] = torch.linspace(-0.5, 1.2, h)
    dg = torch.randn(m, hk)
    x = torch.randn(m, d).to(dtype)
    g_proj = (torch.randn(hk, d) * d**-0.5).to(dtype)
    aux = torch.randn(m, hk)
    ref = run_torch_proj(lr, f_b, dt_bias, A_log, h, lb, dg, x, g_proj, aux)

    def gate_bwd():
        dz, dterm = fk.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, h, dg, lb)
        return dz, dterm.view(m, h, k).sum((0, 2))

    candidates = {
        "gate": (
            lambda: fk.kda_gate_proj(lr, f_b, dt_bias, A_log, h, lb),
            2 * m * r * hk,
            (m * r + hk * r) * 2 + m * hk * 4,
        ),
        "gate_bwd": (
            gate_bwd,
            2 * m * r * hk,
            (m * r + hk * r) * 2 + 3 * m * hk * 4,
        ),
        "sigmoid_mul": (
            lambda: fk.sigmoid_gate_proj(x, g_proj, aux),
            2 * m * d * hk,
            (m * d + hk * d) * 2 + 2 * m * hk * 4,
        ),
        # aux stands in for dgp: dx += dgp @ g_proj, dg_proj = dgp^T @ x
        "gproj_bwd": (
            lambda: fk.gate_proj_bwd_gemms(aux, x, g_proj),
            4 * m * d * hk,
            (m * d + hk * d) * 2 * 2 + m * hk * 4 + (m * d + hk * d) * 4,
        ),
    }
    ret = {"gfx": get_gfx_runtime()}
    for name, (fn, flops, nbytes) in candidates.items():
        out, us = run_perftest(fn)
        outs = out if isinstance(out, tuple) else (out,)
        refs = ref[name] if isinstance(ref[name], tuple) else (ref[name],)
        worst = 0.0
        for i, (o, rf) in enumerate(zip(outs, refs)):
            err = checkAllclose(
                rf.to(dtypes.fp32),
                o.to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                msg=f"flydsl {name}[{i}]",
            )
            assert err == 0, f"flydsl {name}[{i}]: {err:.2%} outside tolerance"
            worst = max(worst, err)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = worst
    return ret


def test_kda_save_fwd_matches_recompute(
    dtype, cu_seqlens=None, lower_bound=-5.0, chunk_bwd=None
):
    """save_fwd=1 (the forward keeps its activations and, for a chunk backward
    that takes ``fwd_intermediates``, the chunk forward's intermediates) vs the
    default recompute. Both are graded against an fp32 torch reference: the
    saved intermediates can come from a different forward than the backward's
    own prepare (e.g. Triton's bf16 Aqk/Akk vs an fp32 recompute), so save_fwd
    must be about as accurate as recompute, not bit-identical to it.
    ``chunk_bwd="flydsl"`` covers the #8740 ``fwd_intermediates`` path; the
    default pieces cover the FlyDSL chunk forward default."""
    D, H, K, W, R = 256, 2, 128, 4, 16
    total = 128
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    saved = dict(kda_fused_layer._fusion_overrides)
    pieces = dict(kda_fused_layer._piece_overrides)
    if chunk_bwd:
        kda_fused_layer.set_kda_pieces(chunk_bwd=chunk_bwd)
    try:
        w = make_weights(D, H, K, W, R, dtype)
        x = torch.randn(1, total, D).to(dtype)
        dy = torch.randn(1, total, D).to(dtype)

        def run(backend, xx, ww, dyy):
            xs = xx.clone().requires_grad_(True)
            ws = kda_fused_layer.KDALayerWeights.from_tensors(
                [t.detach().clone().requires_grad_(True) for t in ww.tensors()]
            )
            y = kda_fused_layer.kda_layer_fwd(
                xs, ws, H, backend=backend, cu_seqlens=cu, lower_bound=lower_bound
            )
            y.backward(dyy)
            return [y.detach(), xs.grad, *(t.grad for t in ws.tensors())]

        w32 = kda_fused_layer.KDALayerWeights.from_tensors(
            [t.float() for t in w.tensors()]
        )
        ref = run("torch", x.float(), w32, dy.float())

        def err(outs):
            return max(
                ((a.float() - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
                for a, b in zip(outs, ref)
            )

        errs, used = {}, {}
        for mode in ("0", "1"):
            kda_fused_layer.set_kda_fusions(save_fwd=mode)
            errs[mode] = err(run("fused", x, w, dy))
            used[mode] = dict(kda_fused_layer.LAST_PIECES)
        slack = 1e-4 if dtype == torch.float32 else 1e-2
        ok = errs["1"] <= 2 * errs["0"] + slack
        return {
            "pieces": chunk_bwd and f"chunk_bwd={chunk_bwd}" or "default",
            "recompute err": errs["0"],
            "save_fwd err": errs["1"],
            "chunk_fwd (recompute)": used["0"].get("chunk_fwd"),
            "chunk_fwd (save_fwd)": used["1"].get("chunk_fwd"),
            "chunk_bwd": used["1"].get("chunk_bwd"),
            "fwd_intermediates": used["1"].get("chunk_bwd_fwd_intermediates", "-"),
            "passed": ok,
        }
    finally:
        kda_fused_layer._fusion_overrides.clear()
        kda_fused_layer._fusion_overrides.update(saved)
        kda_fused_layer._piece_overrides.clear()
        kda_fused_layer._piece_overrides.update(pieces)


def test_kda_save_fwd_intermediates_plumbing():
    """CPU check of save_fwd=1's intermediates hand-off with stub forwards and a
    registered chunk backward taking ``fwd_intermediates`` (the forwards come
    from a replaced ``_load_save_fwds``, independent of the real modules): the FlyDSL forward
    with ``return_intermediates`` serves when it can (its 10-tuple passed as
    is); a FlyDSL forward without it, or one rejecting the call, falls back to
    the Triton ``disable_recompute`` forward (its 7 tensors reordered). The
    results equal recompute."""
    L = kda_fused_layer
    real_load = L._load_save_fwds
    got = []

    def o_of(q, k, v, g, beta, scale, cu_seqlens=None):
        b = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        return torch.cat(
            [
                L.chunk_kda_torch(
                    *(t[:, s:e].float() for t in (q, k, v, g, beta)), scale, 64
                )
                for s, e in itertools.pairwise(b)
            ],
            1,
        )

    def tagged(n):
        return [torch.full((1,), float(i)) for i in range(n)]

    def fly(
        q,
        k,
        v,
        g,
        beta,
        scale,
        chunk_size=64,
        *,
        cu_seqlens=None,
        return_intermediates=False,
        **kw,
    ):
        if fly.reject:
            raise ValueError("stub: shape not served")
        o = o_of(q, k, v, g, beta, scale, cu_seqlens)
        inter = tagged(10)
        if fly.nan:
            inter[8] = torch.full((1,), float("nan"))
        return (o, None, tuple(inter)) if return_intermediates else (o, None)

    def fly_plain(q, k, v, g, beta, scale, chunk_size=64, *, cu_seqlens=None, **kw):
        return o_of(q, k, v, g, beta, scale, cu_seqlens), None

    def tri(
        q,
        k,
        v,
        g,
        beta,
        scale,
        h0,
        final,
        cu_seqlens=None,
        chunk_size=64,
        safe_gate=False,
        disable_recompute=False,
        **kw,
    ):
        assert disable_recompute
        return (o_of(q, k, v, g, beta, scale, cu_seqlens), None, *tagged(7))

    def bwd(
        do,
        q,
        k,
        v,
        g,
        beta,
        scale,
        cs,
        *,
        cu_seqlens=None,
        fwd_intermediates=None,
        **kw,
    ):
        got.append(fwd_intermediates)
        b = [0, q.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
        outs = [
            L.chunk_kda_bwd_torch(
                *(t[:, s:e] for t in (do, q, k, v, g, beta)), scale, cs
            )
            for s, e in itertools.pairwise(b)
        ]
        return (*(torch.cat([o[i] for o in outs], 1) for i in range(5)), None, None)

    saved_f, saved_p = dict(L._fusion_overrides), dict(L._piece_overrides)
    saved_bwd = (L._chunk_bwd_impl, L._chunk_bwd_fuses)
    rows = []
    try:
        L.register_chunk_bwd(bwd, frozenset())
        D, H, K, W, R = 256, 2, 128, 4, 16
        w = make_weights(D, H, K, W, R, torch.float32)
        for case, fn, reject, want_fwd, want in (
            ("flydsl return_intermediates", fly, False, "flydsl", list(range(10))),
            ("flydsl without it", fly_plain, False, "triton", [0, 3, 4, 5, 6, 1, 2]),
            ("flydsl rejects", fly, True, "triton", [0, 3, 4, 5, 6, 1, 2]),
            ("flydsl NaN Aqk", fly, False, "triton", [0, 3, 4, 5, 6, 1, 2]),
        ):
            fly.reject = reject
            fly.nan = case == "flydsl NaN Aqk"

            def load(want_flydsl, fn=fn):
                if want_flydsl and L._accepts(fn, "return_intermediates"):
                    return fn, tri
                return None, tri

            L._load_save_fwds = load
            for cu in (None, torch.tensor([0, 48, 128], dtype=torch.int32)):
                x = torch.randn(1, 128, D)
                dy = torch.randn(1, 128, D)
                outs = {}
                for mode in ("0", "1"):
                    L.set_kda_fusions(save_fwd=mode)
                    got.clear()
                    xs = x.clone().requires_grad_(True)
                    y = L.kda_layer_fwd(xs, w, H, backend="fused", cu_seqlens=cu)
                    y.backward(dy)
                    outs[mode] = [y.detach(), xs.grad]
                    used = dict(L.LAST_PIECES)
                inter = got[-1]
                tags = [None if t is None else int(t.item()) for t in inter]
                if want_fwd == "triton":
                    want_tags = [None, None, None] + want
                else:
                    want_tags = want
                assert used["chunk_fwd"] == f"{want_fwd}(save intermediates)", used
                assert tags == want_tags, (case, tags)
                diff = max(
                    ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
                    for a, b in zip(outs["1"], outs["0"])
                )
                assert diff <= 1e-5, f"{case}: save_fwd vs recompute rel err {diff:.2e}"
                rows.append(
                    {
                        "case": case,
                        "varlen": cu is not None,
                        "chunk_fwd": used["chunk_fwd"],
                        "passed": True,
                    }
                )
    finally:
        L._load_save_fwds = real_load
        L._fusion_overrides.clear()
        L._fusion_overrides.update(saved_f)
        L._piece_overrides.clear()
        L._piece_overrides.update(saved_p)
        L._chunk_bwd_impl, L._chunk_bwd_fuses = saved_bwd
    return rows


def test_kda_weight_cache_update(dtype=torch.float32, concat_proj=False):
    """Optimizer updates that keep data_ptr and _version (p.data.copy_, a
    distributed optimizer's shared flat buffer) must reach the next call:
    is_first_microbatch None always rebuilds, True rebuilds, and False reuses
    until invalidate_weight_caches. Asserts on CPU or GPU."""
    D, H, K, W, R = 128, 2, 64, 4, 16
    T, saved = 40, dict(kda_fused_layer._fusion_overrides)
    kda_fused_layer.set_kda_fusions(concat_proj=concat_proj)
    seen = []

    def hook(weights, anchors):
        seen.append((weights, len(anchors)))
        assert all(isinstance(a, torch.Tensor) for a in anchors)

    def reuse_probe(*a, **k):
        reuse.append(kda_fused_layer.kda_weight_reuse())
        return NotImplemented

    reuse = []
    kda_fused_layer.register_weight_cache_invalidator(hook)
    prev_gemm = kda_fused_layer._proj_gemm_hook
    kda_fused_layer.register_proj_gemm(reuse_probe)
    try:
        x = torch.randn(1, T, D).to(dtype)
        dy = torch.randn(1, T, D).to(dtype)

        def run(w, first):
            y = kda_fused_layer.kda_layer_fwd(
                x, w, H, backend="fused", is_first_microbatch=first
            )
            dx, dw = kda_fused_layer.kda_layer_bwd(
                dy, x, w, H, backend="fused", is_first_microbatch=first
            )
            return [y, dx, *dw.tensors()]

        def same(a, b):
            return all(torch.equal(p, q) for p, q in zip(a, b))

        names = ("w_q", "w_k", "w_v", "conv_q", "f_a", "w_beta", "g_proj")
        for how in ("copy_", "buffer"):
            kda_fused_layer.invalidate_weight_caches()
            w = make_weights(D, H, K, W, R, dtype)
            if how == "buffer":
                ps = [getattr(w, n) for n in names]
                buf = torch.cat([p.detach().reshape(-1).float() for p in ps])
                off = 0
                for p in ps:
                    view = buf[off : off + p.numel()].view(p.shape)
                    assert p.dtype == buf.dtype, "shared-buffer case needs fp32 weights"
                    p.data = view
                    off += p.numel()
            new = {n: torch.randn_like(getattr(w, n)) for n in names}
            fresh = kda_fused_layer.KDALayerWeights.from_tensors(
                [t.clone() for t in w.tensors()]
            )
            for n in names:
                getattr(fresh, n).data.copy_(new[n])
            want = run(fresh, None)

            reuse.clear()
            run(w, None)
            assert reuse and not any(reuse), f"{how}: kda_weight_reuse under None"
            seen.clear()
            reuse.clear()
            old = run(w, True)
            assert reuse and all(reuse), f"{how}: kda_weight_reuse under True"
            assert seen and seen[0][0] is w, f"{how}: True did not invalidate w"
            assert same(run(w, False), old), f"{how}: False did not reuse"
            if how == "copy_":
                for n in names:
                    getattr(w, n).data.copy_(new[n])
            else:
                buf.copy_(torch.cat([new[n].reshape(-1).float() for n in names]))
            assert same(run(w, None), want), f"{how}: None used stale weights"
            assert same(run(w, True), want), f"{how}: True used stale weights"
            for n in names:
                getattr(w, n).data.copy_(getattr(w, n) * 0.5)
            kda_fused_layer.invalidate_weight_caches(w)
            got = run(w, False)
            for n in names:
                getattr(fresh, n).data.copy_(getattr(w, n))
            assert same(
                got, run(fresh, None)
            ), f"{how}: False after invalidate_weight_caches used stale weights"
    finally:
        kda_fused_layer._fusion_overrides.clear()
        kda_fused_layer._fusion_overrides.update(saved)
        kda_fused_layer._INVALIDATE_HOOKS.remove(hook)
        kda_fused_layer.register_proj_gemm(prev_gemm)
        kda_fused_layer.invalidate_weight_caches()


SELF_CHECK_FAILURES = []


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "kda fused layer bwd unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.fp32, dtypes.bf16],
        help="data type, e.g. -d bf16",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument(
        "-t",
        "--seqlen",
        type=int,
        nargs="*",
        default=[100, 256],
        help="tokens per sequence; 100 exercises a partial last chunk",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(K3_HIDDEN, K3_HEADS, K3_HEAD_DIM, K3_CONV, K3_HEAD_DIM)],
        help="hidden,heads,head_dim,conv_width,gate_rank, e.g. -s 7168,96,128,4,128",
    )
    parser.add_argument("-c", "--chunk", type=int, nargs="*", default=[64])
    parser.add_argument(
        "-l",
        "--block",
        type=dtypes.str2tuple,
        nargs="*",
        # Lumen#35 long sequences: T=8192, 2x4096, varlen 8x2048 (packed, b=1)
        default=[
            (1, 100, 1),
            (1, 256, 1),
            (1, 256, 2),
            (1, 8192, 1),
            (2, 4096, 1),
            (1, 16384, 8),
        ],
        help="block fwd+bwd rows: batch,tokens,num_seqs; num_seqs > 1 packs\n"
        "equal sequences into one row and passes cu_seqlens, e.g. -l 1,16384,8",
    )
    parser.add_argument(
        "-g",
        "--gate_lb",
        type=lambda v: None if v.lower() == "none" else float(v),
        nargs="*",
        default=[-5.0],
        help="decay gate lower bound (Kimi-K3 gate_lower_bound=-5, safe gate);\n"
        "none = the softplus gate, e.g. -g -5 none",
    )
    parser.add_argument(
        "-p",
        "--pieces",
        nargs="*",
        default=list(kda_fused_layer.KDA_PIECE_DEFAULTS),
        help="stages to run on FlyDSL one at a time (the rest on Triton);\n"
        "a bare -p means all of them",
    )
    parser.add_argument(
        "-f",
        "--fusions",
        nargs="*",
        default=["default"],
        help="fusion combos for the block table, AITER_KDA_FUSIONS syntax;\n"
        "e.g. -f default save_fwd=1 save_fwd=1,concat_proj=1",
    )
    args = parser.parse_args()
    # argparse turns a bare `-p` (nargs="*") into [], which would sweep nothing
    if not args.pieces:
        args.pieces = list(kda_fused_layer.KDA_PIECE_DEFAULTS)
    unknown = set(args.pieces) - set(kda_fused_layer.KDA_PIECE_DEFAULTS)
    if unknown:
        parser.error(
            f"unknown -p pieces {sorted(unknown)}; "
            f"expected {sorted(kda_fused_layer.KDA_PIECE_DEFAULTS)}"
        )
    for concat_proj in (False, True):
        try:
            test_kda_weight_cache_update(torch.float32, concat_proj)
        except Exception as err:  # noqa: BLE001
            msg = (
                f"weight cache (concat_proj={concat_proj}): {type(err).__name__}: {err}"
            )
            aiter.logger.error(msg)
            SELF_CHECK_FAILURES.append(msg)
    # self-checks report and continue; __main__ raises after the tables
    try:
        print(pd.DataFrame(test_kda_save_fwd_intermediates_plumbing()).to_markdown())
    except Exception as err:  # noqa: BLE001
        msg = f"save_fwd intermediates plumbing: {type(err).__name__}: {err}"
        aiter.logger.error(msg)
        SELF_CHECK_FAILURES.append(msg)
    save_rows = []
    for dt in (torch.float32, torch.bfloat16):
        bwds = [None] + (
            ["flydsl"] if dt == torch.bfloat16 and get_gfx_runtime() == "gfx950" else []
        )
        for chunk_bwd in bwds:
            for cu in (None, [0, 48, 128]):
                row = {"dtype": str(dt), "cu_seqlens": cu}
                try:
                    row |= test_kda_save_fwd_matches_recompute(
                        dt, cu, chunk_bwd=chunk_bwd
                    )
                except Exception as err:  # noqa: BLE001
                    row |= {"passed": False, "error": f"{type(err).__name__}: {err}"}
                if not row["passed"]:
                    SELF_CHECK_FAILURES.append(f"save_fwd vs recompute: {row}")
                save_rows.append(row)
    print("\n### kda save_fwd vs recompute (both vs fp32 reference)\n")
    print(pd.DataFrame(save_rows).to_markdown(index=False))

    def summarize(title, rows, axes):
        if not rows:
            raise RuntimeError(
                f"{title}: no rows; the sweep is empty ("
                + ", ".join(f"{name}={vals}" for name, vals in axes.items())
                + ")"
            )
        df = pd.DataFrame(rows)
        aiter.logger.info("%s (markdown):\n%s", title, df.to_markdown(index=False))
        return df

    for dtype in args.dtype:
        layer_axes = {
            "batch": args.batch,
            "seqlen": args.seqlen,
            "shape": args.shape,
            "chunk": args.chunk,
            "gate_lb": args.gate_lb,
        }
        summarize(
            "kda fused layer bwd summary",
            [
                test_kda_layer_bwd(b, t, d, h, k, w, r, c, lb, dtype)
                for b, t, (d, h, k, w, r), c, lb in itertools.product(
                    args.batch, args.seqlen, args.shape, args.chunk, args.gate_lb
                )
            ],
            layer_axes,
        )
        block_rows = []
        for combo in args.fusions:
            spec = (
                {}
                if combo == "default"
                else dict(item.split("=", 1) for item in combo.split(",") if item)
            )
            kda_fused_layer.set_kda_fusions(**spec)
            try:
                for (b, t, nseq), (d, h, k, w, r), c, lb in itertools.product(
                    args.block, args.shape, args.chunk, args.gate_lb
                ):
                    row = test_kda_block(b, t, nseq, d, h, k, w, r, c, lb, dtype)
                    row["fusion combo"] = combo
                    block_rows.append(row)
            finally:
                kda_fused_layer.set_kda_fusions(**dict.fromkeys(spec))
        summarize(
            "kda block fwd+bwd summary",
            block_rows,
            {
                "block": args.block,
                "shape": args.shape,
                "chunk": args.chunk,
                "fusions": args.fusions,
            },
        )
        df = summarize(
            "kda per-piece flydsl summary",
            [
                test_kda_piece(piece, b, t, d, h, k, w, r, c, lb, dtype)
                for piece, b, t, (d, h, k, w, r), c, lb in itertools.product(
                    args.pieces,
                    args.batch,
                    args.seqlen,
                    args.shape,
                    args.chunk,
                    args.gate_lb,
                )
            ],
            {"pieces": args.pieces, **layer_axes},
        )
        if dtype == dtypes.bf16:
            summarize(
                "kda chunk fwd backends (layer layout) summary",
                [
                    test_kda_chunk_fwd_backends(b, t, h, k, lb, dtype)
                    for b, t, (_, h, k, _, _), lb in itertools.product(
                        args.batch, args.seqlen, args.shape, args.gate_lb
                    )
                ],
                layer_axes,
            )
        summarize(
            "kda chunk bwd backends (channel-major inputs) summary",
            [
                test_kda_chunk_bwd_backends(b, t, h, k, lb, dtype)
                for b, t, (_, h, k, _, _), lb in itertools.product(
                    args.batch, args.seqlen, args.shape, args.gate_lb
                )
            ],
            layer_axes,
        )
        stage_df = summarize(
            "kda GPU-only stage diff summary",
            [
                test_kda_stage(stage, b * t, h, k, d, r, lb, dtype)
                for stage, b, t, (d, h, k, _, r), lb in itertools.product(
                    KDA_STAGES, args.batch, args.seqlen, args.shape, args.gate_lb
                )
            ],
            layer_axes,
        )
        bad_stage = stage_df[stage_df["failed"] != "-"]
        assert bad_stage.empty, "GPU-only stages outside tolerance:\n" + bad_stage[
            ["stage", "m", "lb", "failed"]
        ].to_string(index=False)
        bad = df[df["failed grads"] != "-"]
        assert bad.empty, "flydsl pieces outside tolerance:\n" + bad[
            ["piece", "t", "used", "failed grads"]
        ].to_string(index=False)

    # the fused projections are the gfx950 bf16/fp16 GEMM with an epilogue
    proj_dtypes = [dt for dt in args.dtype if dt in (dtypes.bf16, dtypes.fp16)]
    if get_gfx_runtime() != "gfx950" or not proj_dtypes:
        aiter.logger.warning(
            "kda projection fusion needs gfx950 and bf16/fp16; skipping its table"
        )
        return
    for dtype in proj_dtypes:
        summarize(
            "kda g_proj bwd GEMM bit-exact summary",
            [
                test_kda_gproj_bwd_exact(b * t, d, h, k, dtype)
                for b, t, (d, h, k, _, _) in itertools.product(
                    args.batch, args.seqlen, args.shape
                )
            ],
            {"batch": args.batch, "seqlen": args.seqlen, "shape": args.shape},
        )
        summarize(
            "kda projection fusion summary",
            [
                test_kda_proj(b * t, d, h, k, r, lb, dtype)
                for b, t, (d, h, k, _, r), lb in itertools.product(
                    args.batch, args.seqlen, args.shape, args.gate_lb
                )
            ],
            {"batch": args.batch, "seqlen": args.seqlen, "shape": args.shape},
        )


if __name__ == "__main__":
    main()
    if SELF_CHECK_FAILURES:
        raise AssertionError("self-checks failed:\n" + "\n".join(SELF_CHECK_FAILURES))
