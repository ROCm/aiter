# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import kda_fused_layer, mxfp8_linear
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
            # (dz, dA_log, d dt_bias)
            return (
                lambda: fk.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, h, dg, lb),
                lambda: P.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, h, dg, lb),
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
    gate_out = torch.sigmoid(x.float() @ g_proj.float().t())
    aux16 = aux.to(g_proj.dtype).float()
    return {
        "gate": gate.detach(),
        # dz, dA_log, d dt_bias
        "gate_bwd": (dz, da, dz.sum(0)),
        "sigmoid_mul": gate_out * aux,
        # the layer feeds dgp to these GEMMs in the weight dtype; grade them on
        # the same rounded input so only the GEMM itself is measured
        "gproj_bwd": (aux16 @ g_proj.float(), aux16.t() @ x.float()),
    }


def _gate_unfused(z, dt_bias, A_log, h, lb):
    """The decay gate as separate torch element-wise ops on an fp32 GEMM output."""
    a = A_log[:h].float().exp().repeat_interleave(z.shape[-1] // h)
    zb = z + dt_bias.float()
    if lb is None:
        return -a * torch.nn.functional.softplus(zb)
    return lb * torch.sigmoid(a * zb)


def _gate_bwd_unfused(z, dt_bias, A_log, h, lb, dg):
    a = A_log[:h].float().exp().repeat_interleave(z.shape[-1] // h)
    zb = z + dt_bias.float()
    if lb is None:
        dz = -dg * a * torch.sigmoid(zb)
        dterm = -dg * a * torch.nn.functional.softplus(zb)
    else:
        sg = torch.sigmoid(a * zb)
        dz = dg * lb * sg * (1 - sg) * a
        dterm = dz * zb
    return dz, dterm.view(z.shape[0], h, -1).sum((0, 2)), dz.sum(0)


@benchmark()
def test_kda_proj(m, d, h, k, r, lb, dtype):
    """Fused epilogue GEMMs against the unfused GEMM + element-wise baselines.

    ``unfused_torch`` is what the layer runs without the fusion (fp32-output
    ``torch.mm`` on hipBLASLt, then torch element-wise); ``unfused_tgemm`` takes
    the GEMM from aiter's tuned bf16 dispatch (``tgemm.mm``, weights [N, K]).
    """
    from aiter.ops.flydsl import kda_layer_bwd as fk
    from aiter.tuned_gemm import tgemm

    hk = h * k
    lr = torch.randn(m, r)  # fp32, as the layer passes it
    f_b = (torch.randn(hk, r) * r**-0.5).to(dtype)
    dt_bias = torch.randn(hk)
    A_log = torch.zeros(max(h, K3_A_LOG_LEN))
    A_log[:h] = torch.linspace(-0.5, 1.2, h)
    dg = torch.randn(m, hk)
    x = torch.randn(m, d).to(dtype)
    g_proj = (torch.randn(hk, d) * d**-0.5).to(dtype)
    aux = torch.randn(m, hk)
    ref = run_torch_proj(lr, f_b, dt_bias, A_log, h, lb, dg, x, g_proj, aux)

    def mm32(a, b):
        return torch.mm(a, b, out_dtype=torch.float32)

    def tmm32(a, w):
        return tgemm.mm(a, w, otype=torch.float32).float()

    aux16 = aux.to(dtype)
    # the gate GEMMs run on lr split into bf16 hi + lo: K = 2R; the unfused
    # baselines get the same split and a pre-built [f_b | f_b]
    f_b2 = torch.cat([f_b, f_b], -1)
    gate_bytes = (m * 2 * r + hk * r) * 2 + m * hk * 4
    sig_bytes = (m * d + hk * d) * 2 + m * hk * (4 + 2)
    candidates = {
        "gate": (
            {
                "fused": lambda: fk.kda_gate_proj(lr, f_b, dt_bias, A_log, h, lb),
                "unfused_torch": lambda: _gate_unfused(
                    mm32(fk._split_lowrank(lr, f_b), f_b2.t()), dt_bias, A_log, h, lb
                ),
                "unfused_tgemm": lambda: _gate_unfused(
                    tmm32(fk._split_lowrank(lr, f_b), f_b2), dt_bias, A_log, h, lb
                ),
            },
            2 * m * 2 * r * hk,
            gate_bytes,
        ),
        "gate_bwd": (
            {
                "fused": lambda: fk.kda_gate_proj_bwd(
                    lr, f_b, dt_bias, A_log, h, dg, lb
                ),
                "unfused_torch": lambda: _gate_bwd_unfused(
                    mm32(fk._split_lowrank(lr, f_b), f_b2.t()),
                    dt_bias,
                    A_log,
                    h,
                    lb,
                    dg,
                ),
                "unfused_tgemm": lambda: _gate_bwd_unfused(
                    tmm32(fk._split_lowrank(lr, f_b), f_b2), dt_bias, A_log, h, lb, dg
                ),
            },
            2 * m * 2 * r * hk,
            gate_bytes + m * hk * 4,
        ),
        "sigmoid_mul": (
            {
                "fused": lambda: fk.sigmoid_gate_proj(x, g_proj, aux),
                "unfused_torch": lambda: (torch.sigmoid(mm32(x, g_proj.t())) * aux).to(
                    dtype
                ),
                "unfused_tgemm": lambda: (torch.sigmoid(tmm32(x, g_proj)) * aux).to(
                    dtype
                ),
            },
            2 * m * d * hk,
            sig_bytes,
        ),
        # aux stands in for dgp: dx += dgp @ g_proj (nn), dW_g = dgp^T @ x (tn,
        # K = m: a token count off the 64-wide K tile exercises the K tail)
        "gproj_bwd": (
            {
                "fused": lambda: fk.gate_proj_bwd_gemms(aux, x, g_proj),
                "unfused_torch": lambda: (
                    mm32(aux16, g_proj),
                    mm32(aux16.t(), x),
                ),
            },
            4 * m * d * hk,
            (m * d + hk * d) * 2 * 2 + m * hk * 2 + (m * d + hk * d) * 4,
        ),
    }
    ret = {"gfx": get_gfx_runtime()}
    tiles = {
        "gate": (m, hk, 2 * r, "nt", 1 if lb is None else 5, dtypes.fp32),
        "gate_bwd": (m, hk, 2 * r, "nt", 2 if lb is None else 6, dtypes.fp32),
        "sigmoid_mul": (m, hk, d, "nt", 4, dtype),
        "gproj_bwd": (hk, d, m, "tn", 0, dtypes.fp32),
    }
    for name, (impls, flops, nbytes) in candidates.items():
        refs = ref[name] if isinstance(ref[name], tuple) else (ref[name],)
        cfg, source = fk.proj_config(*tiles[name], dtype)
        ret[f"{name} cfg"] = f"{source}:{config_name_of(cfg)}"
        for impl, fn in impls.items():
            out, us = run_perftest(fn)
            outs = out if isinstance(out, tuple) else (out,)
            worst = 0.0
            for i, (o, rf) in enumerate(zip(outs, refs)):
                # per-head and per-column sums add m * k terms of O(1)
                atol = 2e-2 if o.dim() == 2 else 2e-2 * max(m, 64) ** 0.5
                err = checkAllclose(
                    rf.to(dtypes.fp32),
                    o.to(dtypes.fp32),
                    rtol=2e-2,
                    atol=atol,
                    msg=f"{impl} {name}[{i}]",
                )
                assert err == 0, f"{impl} {name}[{i}]: {err:.2%} outside tolerance"
                worst = max(worst, err)
            key = f"{name} {impl}"
            ret[f"{key} us"] = us
            ret[f"{key} TFLOPS"] = flops / us / 1e6
            ret[f"{key} TB/s"] = nbytes / us / 1e6
            ret[f"{key} err"] = worst
    return ret


# split_k of the dW_g bit-exactness sweep (each run where the tile count allows)
WGRAD_SPLITS = (1, 2, 4)


@benchmark()
def test_kda_wgrad_split(m, hk, d, dtype):
    """dW_g = dgp^T @ x (tn, K = m tokens) on every split_k and kernel.

    Integer inputs in [-2, 2] keep every partial sum exact in fp32, so every
    split, kernel and the dispatcher's pick must equal the exact result bit
    for bit; the output is NaN-filled, so a tile left unwritten fails. Split-K
    rows run the deterministic partial reduction the KDA dispatch uses.
    m = 300 runs the K tail. The HTI rows show whether the HTI kernel takes
    the tn layout; the KDA dispatcher no longer picks it.
    """
    from aiter.ops.flydsl import kda_layer_bwd as fk
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        GEMM_A16W16_DTYPE_FP32,
        gemm_a16w16,
    )

    dgp = torch.randint(-2, 3, (m, hk)).to(dtype)
    x = torch.randint(-2, 3, (m, d)).to(dtype)
    ref = dgp.float().t() @ x.float()
    selected, source = fk.proj_config(hk, d, m, "tn", 0, torch.float32, dtype)

    def ft(split_k):
        return {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "stages": 2,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 0,
            "use_half_tile_interleaved": False,
            # the KDA dispatch reduces split-K partials in split order
            "split_reduce": split_k if split_k > 1 else 0,
        }

    hti256 = {
        **ft(1),
        "block_m": 256,
        "block_n": 256,
        "n_waves": 4,
        "group_m": 4,
        "use_half_tile_interleaved": True,
    }
    # (config, output dtype); the bf16 rows round the exact sums once, so
    # they are bit-exact too. hti256 bf16 runs the HTI store as #8739 has it
    # (each wave row reads back its own C rows); hti256 fp32 runs the
    # realigned, fenced store that fp32 C needs.
    configs = {f"selected_{source}": (selected, torch.float32)}
    configs.update({f"ft_sk{sk}": (ft(sk), torch.float32) for sk in WGRAD_SPLITS})
    configs["hti256"] = (hti256, torch.float32)
    configs["ft_sk1_bf16"] = (ft(1), dtype)
    configs["hti256_bf16"] = (hti256, dtype)
    ret = {"gfx": get_gfx_runtime(), "selected": config_name_of(selected)}
    flops = 2 * m * hk * d
    failed = []
    for name, (cfg, out_dtype) in configs.items():
        legal = fk._validate(
            hk,
            d,
            m,
            "tn",
            {
                "out_dtype_id": (
                    GEMM_A16W16_DTYPE_FP32
                    if out_dtype == torch.float32
                    else GEMM_A16W16_DTYPE_BF16
                ),
                "epilogue": 0,
                "epi_group": 1,
            },
        )
        # split-K takes at most 2048 output tiles and HTI an even count of K
        # tiles: such a config is not launchable, not wrong
        if not legal(cfg):
            continue
        out = torch.full((hk, d), float("nan"), dtype=out_dtype)

        def run(cfg=cfg, out=out, out_dtype=out_dtype):
            return gemm_a16w16(
                dgp.t(),
                x,
                out=out,
                user_kwargs=dict(cfg),
                layout="tn",
                out_dtype=out_dtype,
            )

        _, us = run_perftest(run, num_iters=10, num_warmup=1)
        exact = ref.to(out_dtype)
        bad = (out != exact).float().mean().item()
        if not torch.equal(out, exact):
            failed.append(
                f"{name} ({config_name_of(cfg)}, {out_dtype}): {bad:.2%} differ, "
                f"{out.isnan().float().mean().item():.2%} NaN"
            )
        nbytes = (m * hk + m * d) * dgp.element_size() + hk * d * out.element_size()
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = bad
    # after the loop, so every config's row is in the table
    assert not failed, "dW_g not bit-exact: " + "; ".join(failed)
    return ret


def config_name_of(user_kwargs):
    from aiter.ops.flydsl.gemm_epilogue_configs import config_name

    return config_name(
        {
            **user_kwargs,
            "hti": user_kwargs["use_half_tile_interleaved"],
        }
    )


# MXFP8 recipe (Megatron-LM Kimi-K3 / TE MXFP8BlockScaling): the aiter path
# against the fp32 emulation of the same quantized operands. One GEMM differs
# only in summation order and in which side of a bf16 rounding a value lands,
# hence the tight elementwise tolerance there.
MXFP8_TIGHT_TOL = 1e-2
# Through the whole layer a bf16 rounding that lands differently can move a
# later fp8 rounding, or a block's amax across a power of two (rescaling its 32
# values), so a few elements of a gradient move by about one fp8 step while the
# tensor stays close; how far depends on the tensor (the low-rank f_a / f_b
# path is the most sensitive). Each tensor is therefore graded against the
# emulated layer by its deviation relative to the recipe's own effect on it,
# rel_l2(aiter, emulated) / rel_l2(emulated, bf16 layer) <= MXFP8_NOISE_RATIO.
# Simulated 1% one-ulp output flips stay <= 0.11 (GPU K3 rows <= 0.17); an
# unquantized operand gives 0.7-0.9, unquantized wgrad operands alone up to 0.37.
MXFP8_NOISE_RATIO = 0.25
# Backstops per comparison: (rel-L2, max |delta| / max|ref|), plus the
# elementwise tolerance of the reported (not asserted) outlier fraction.
MXFP8_BLOCK_GRADES = {
    "emulated": (5e-2, 0.1, 1e-2),
    "torch": (5e-2, 0.2, 5e-2),
}
MXFP8_PROJS = ("qkv", "f_a", "f_b", "w_beta", "g_proj", "w_o")
MXFP8_KINDS = ("fwd", "dgrad", "wgrad")


def mxfp8_proj_shapes(d, h, k, r):
    """(N, K) of each KDA projection weight."""
    hk = h * k
    return {
        "qkv": (3 * hk, d),
        "f_a": (r, d),
        "f_b": (hk, r),
        "w_beta": (h, d),
        "g_proj": (hk, d),
        "w_o": (d, hk),
    }


def _rel_l2(out, ref):
    return (
        (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
    ).item()


@benchmark()
def test_kda_mxfp8_gemm(proj, kind, m, d, h, k, r, dtype):
    """One projection GEMM through ``MXFP8ProjGemm.gemm`` as the layer calls it
    (``kda_fused_layer._proj_gemm`` operand convention): fprop / dgrad on the
    native FP8 GEMM (bf16 out), wgrad on the bf16 GEMM over dequantized
    operands (fp32 out), against the fp32 emulation. ``us`` includes the
    torch-side quantization of both operands."""
    n, kk = mxfp8_proj_shapes(d, h, k, r)[proj]
    if kind == "fwd":
        a, wt = torch.randn(m, kk), torch.randn(n, kk) * kk**-0.5
    elif kind == "dgrad":
        a, wt = torch.randn(m, n), torch.randn(n, kk) * kk**-0.5
    else:
        a, wt = torch.randn(m, n), torch.randn(m, kk)
    a, wt = a.to(dtype), wt.to(dtype)
    hook = kda_fused_layer.MXFP8ProjGemm(dtype)
    mxfp8_linear.set_mxfp8_gemm_impl("emulate")
    try:
        ref = hook.gemm(a, wt, kind)
    finally:
        mxfp8_linear.set_mxfp8_gemm_impl(None)
    # the layer's GPU cast (Triton convert_to_mxfp8, scale_mode="te") must match
    # the torch port of TE's cast bit for bit
    blocks = a.reshape(-1)[: a.numel() // 32 * 32].view(-1, 32)
    qg, sg = mxfp8_linear.mxfp8_quantize(blocks, backend="triton")
    qc, sc = mxfp8_linear.mxfp8_quantize(blocks.cpu())
    bitexact = torch.equal(qg.view(torch.uint8).cpu(), qc.view(torch.uint8)) and (
        torch.equal(sg.cpu(), sc)
    )
    # whether aiter's HIP 1x32 e8m0 quant (RoundUp) could replace the torch cast
    try:
        from aiter.ops.quant import per_group_quant_hip

        rows = blocks.reshape(-1, 256) if blocks.numel() % 256 == 0 else blocks
        qh, shh = per_group_quant_hip(
            rows, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
        )
        qt, st = mxfp8_linear.mxfp8_quantize(rows)
        hip_bitexact = torch.equal(
            qh.view(torch.uint8), qt.view(torch.uint8)
        ) and torch.equal(shh.view(torch.uint8).reshape(st.shape), st)
    except (ImportError, RuntimeError, ValueError, TypeError) as err:
        # reported, not fatal: the layer does not use this kernel
        hip_bitexact = f"n/a ({type(err).__name__})"
    mxfp8_linear.LAST_IMPLS.clear()
    out, us = run_perftest(hook.gemm, a, wt, kind, num_iters=10, num_warmup=2)
    impl = ",".join(sorted(mxfp8_linear.LAST_IMPLS))
    mxfp8_linear.set_mxfp8_fastpath(False)
    try:
        plain = hook.gemm(a, wt, kind)
    finally:
        mxfp8_linear.set_mxfp8_fastpath(None)
    sc_ = ref.abs().max().clamp_min(1e-6)
    err = checkAllclose(
        (ref / sc_).to(dtypes.fp32),
        (out / sc_).to(dtypes.fp32),
        rtol=MXFP8_TIGHT_TOL,
        atol=MXFP8_TIGHT_TOL,
        msg=f"mxfp8 {proj} {kind}",
    )
    flops = 2 * m * n * kk
    nbytes = (a.numel() + wt.numel()) * a.element_size() + out.numel() * 4
    return {
        "gfx": get_gfx_runtime(),
        "impl": impl,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": err,
        "rel l2": _rel_l2(out, ref),
        "quant bitexact": bitexact,
        "hip quant bitexact": hip_bitexact,
        # the numerics-neutral fast paths against the plain path, bit for bit
        "fastpath bitwise": torch.equal(out, plain),
    }


@benchmark()
def test_kda_mxfp8_block(b, t, nseq, d, h, k, w, r, chunk, lb, dtype):
    """kda_layer_fwd + autograd backward with ``gemm_precision="mxfp8"``.

    * ``noise ratio`` / ``rel l2`` / ``err`` / ``max delta`` (worst tensor): the
      fused layer on aiter's MXFP8 GEMMs against the same fused layer with
      every projection GEMM emulated in fp32 (``err``: fraction of elements
      outside 1e-2, reported only);
    * ``deterministic``: two runs of the aiter path (and, up to
      ORACLE_MAX_TOKENS, of the emulated reference) agree bit for bit;
    * ``vs torch ...``: against the torch backend's MXFP8 emulation (chunked
      recurrence in fp32, every GEMM emulated), up to ORACLE_MAX_TOKENS;
    * ``failed``: tensors past ``MXFP8_BLOCK_GRADES``, asserted after the table;
    * ``vs bf16``: how far the MXFP8 recipe moves y and each gradient from the
      bf16 fused layer (reported, not asserted).
    """
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    cu = None
    if nseq > 1:
        assert b == 1 and t % nseq == 0, "varlen rows pack equal sequences into b=1"
        cu = torch.arange(nseq + 1, dtype=torch.int32) * (t // nseq)

    def block(backend, precision):
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
                gemm_precision=precision,
            )
            return y, torch.autograd.grad(y, [xs, *ws], dy)

        return run

    names = ["y", "dx"] + [
        f.name for f in KDALayerWeights.__dataclass_fields__.values()
    ]

    failed = []

    def grade(out, ref, against, effect=None):
        """(worst noise ratio, rel-L2, outlier fraction, max delta); tensors past
        a bound are collected in ``failed`` (asserted after the table).
        ``effect``: per-tensor rel-L2 of the recipe (emulated vs bf16 layer)."""
        rel_max, delta_max, tol = MXFP8_BLOCK_GRADES[against]
        worst = [0.0, 0.0, 0.0, 0.0]
        for gname, rf, o in zip(names, [ref[0], *ref[1]], [out[0], *out[1]]):
            sc = rf.float().abs().max().clamp_min(1e-6)
            rf_n = (rf.float() / sc).to(dtypes.fp32)
            o_n = (o.float() / sc).to(dtypes.fp32)
            frac = checkAllclose(
                rf_n,
                o_n,
                rtol=tol,
                atol=tol,
                tol_err_ratio=1.0,
                printLog=False,
            )
            rel = _rel_l2(o_n, rf_n)
            delta = (o_n - rf_n).abs().max().item()
            ratio = rel / max(effect[gname], 1e-12) if effect else 0.0
            if rel > rel_max or delta > delta_max or ratio > MXFP8_NOISE_RATIO:
                failed.append(
                    f"vs {against} {gname}: noise ratio {ratio:.3f}, rel l2 "
                    f"{rel:.2e}, max {delta:.2e} ({frac:.2e} outside {tol})"
                )
            worst = [max(a, b) for a, b in zip(worst, (ratio, rel, frac, delta))]
        return worst

    mxfp8_linear.set_mxfp8_gemm_impl("emulate")
    try:
        emulated = block("fused", "mxfp8")()
    finally:
        mxfp8_linear.set_mxfp8_gemm_impl(None)
    out, us = run_perftest(block("fused", "mxfp8"), num_iters=3, num_warmup=1)
    gemms = kda_fused_layer.LAST_PIECES.get("gemm", "-")
    flops = layer_flops(b, t, d, h, k, w, r, chunk) * 4 // 3
    ret = {
        "gfx": get_gfx_runtime(),
        "precision": kda_fused_layer.kda_precision(),
        "mxfp8 us": us,
        "mxfp8 TFLOPS": flops / us / 1e6,
        "bwd gemms": gemms,
    }
    mxfp8_linear.set_mxfp8_fastpath(False)
    try:
        plain = block("fused", "mxfp8")()
    finally:
        mxfp8_linear.set_mxfp8_fastpath(None)
    differ = [
        gname
        for gname, o, pl in zip(names, [out[0], *out[1]], [plain[0], *plain[1]])
        if not torch.equal(o, pl)
    ]
    ret["fastpath bitwise"] = not differ
    if differ:
        failed.append("fast path not bit-identical: " + ",".join(differ))
    again = block("fused", "mxfp8")()
    nondet = [
        gname
        for gname, o, o2 in zip(names, [out[0], *out[1]], [again[0], *again[1]])
        if not torch.equal(o, o2)
    ]
    if b * t <= ORACLE_MAX_TOKENS:
        mxfp8_linear.set_mxfp8_gemm_impl("emulate")
        try:
            emulated2 = block("fused", "mxfp8")()
        finally:
            mxfp8_linear.set_mxfp8_gemm_impl(None)
        nondet += [
            f"emulated:{gname}"
            for gname, e, e2 in zip(
                names, [emulated[0], *emulated[1]], [emulated2[0], *emulated2[1]]
            )
            if not torch.equal(e, e2)
        ]
    ret["deterministic"] = not nondet
    ret["nondeterministic"] = ",".join(nondet) or "-"
    bf16, bf16_us = run_perftest(block("fused", "bf16"), num_iters=3, num_warmup=1)
    ret["bf16 us"] = bf16_us
    effect = {
        gname: _rel_l2(e, b16)
        for gname, e, b16 in zip(
            names, [emulated[0], *emulated[1]], [bf16[0], *bf16[1]]
        )
    }
    ratio, rel, frac, delta = grade(out, emulated, "emulated", effect)
    ret.update({"noise ratio": ratio, "rel l2": rel, "err": frac, "max delta": delta})
    if b * t <= ORACLE_MAX_TOKENS:
        _, rel, frac, delta = grade(out, block("torch", "mxfp8")(), "torch")
        ret.update(
            {"vs torch rel l2": rel, "vs torch err": frac, "vs torch max": delta}
        )
    dev = {
        gname: (
            _rel_l2(o, rf),
            ((o.float() - rf.float()).abs().max() / rf.float().abs().max()).item(),
        )
        for gname, o, rf in zip(names, [out[0], *out[1]], [bf16[0], *bf16[1]])
    }
    worst = max(dev, key=lambda g: dev[g][0])
    ret["vs bf16 rel l2 (y)"] = dev["y"][0]
    ret["vs bf16 rel l2 (dx)"] = dev["dx"][0]
    ret["vs bf16 worst"] = f"{worst} {dev[worst][0]:.3g} (max {dev[worst][1]:.3g})"
    ret["failed"] = "; ".join(failed) or "-"
    return ret


def test_kda_mxfp8_weight_cache(concat_proj=False):
    """gemm_precision="mxfp8" keeps quantized weight operands across calls only
    under is_first_microbatch True / False, drops them in
    invalidate_weight_caches, and matches the uncached path bit for bit.
    Asserts on CPU or GPU."""
    D, H, K, W, R, T = 128, 2, 64, 4, 16, 40
    saved = dict(kda_fused_layer._fusion_overrides)
    kda_fused_layer.set_kda_fusions(concat_proj=concat_proj)
    try:
        x = torch.randn(1, T, D).to(dtypes.bf16)
        dy = torch.randn(1, T, D).to(dtypes.bf16)
        w = make_weights(D, H, K, W, R, dtypes.bf16)

        def run(first):
            kw = {"backend": "fused", "is_first_microbatch": first}
            kw["gemm_precision"] = "mxfp8"
            y = kda_layer_fwd(x, w, H, **kw)
            dx, dw = kda_layer_bwd(dy, x, w, H, **kw)
            return [y, dx, *dw.tensors()]

        def same(a, b):
            return all(torch.equal(p, q) for p, q in zip(a, b))

        cached = mxfp8_linear.mxfp8_weight_cache_bytes
        kda_fused_layer.invalidate_weight_caches()
        base = run(None)
        assert cached() == 0, "is_first_microbatch=None kept weight operands"
        assert same(run(True), base) and cached() > 0, "True: not built or differs"
        assert same(run(False), base), "False: reuse differs"
        names = ("w_q", "w_k", "w_v", "f_a", "f_b", "w_beta", "g_proj", "w_o")
        for n in names:  # an update the version counter does not see
            getattr(w, n).data.copy_(getattr(w, n) * 0.5)
        want = run(None)
        kda_fused_layer.invalidate_weight_caches(w)
        assert cached() == 0, "invalidate_weight_caches kept MXFP8 operands"
        assert same(run(False), want), "False after invalidation: stale weights"
        assert same(run(True), want), "True: stale weights"
    finally:
        kda_fused_layer._fusion_overrides.clear()
        kda_fused_layer._fusion_overrides.update(saved)
        kda_fused_layer.invalidate_weight_caches()


def test_kda_save_fwd_matches_recompute(dtype, cu_seqlens=None, lower_bound=-5.0):
    """save_fwd=1 (the forward keeps its activations and, for a chunk backward
    that takes ``fwd_intermediates``, the chunk forward's intermediates) must
    give the default recompute's y and gradients. On gfx950 bf16 the chunk
    backward is FlyDSL, so this covers the #8740 ``fwd_intermediates`` path."""
    D, H, K, W, R = 256, 2, 128, 4, 16
    total = 128
    cu = None if cu_seqlens is None else torch.tensor(cu_seqlens, dtype=torch.int32)
    saved = dict(kda_fused_layer._fusion_overrides)
    pieces = dict(kda_fused_layer._piece_overrides)
    flydsl = dtype == torch.bfloat16 and get_gfx_runtime() == "gfx950"
    if flydsl:
        kda_fused_layer.set_kda_pieces(chunk_bwd="flydsl")
    try:
        w = make_weights(D, H, K, W, R, dtype)
        x = torch.randn(1, total, D).to(dtype)
        dy = torch.randn(1, total, D).to(dtype)
        outs = {}
        for mode in ("0", "1"):
            kda_fused_layer.set_kda_fusions(save_fwd=mode)
            xs = x.clone().requires_grad_(True)
            ws = kda_fused_layer.KDALayerWeights.from_tensors(
                [t.detach().clone().requires_grad_(True) for t in w.tensors()]
            )
            y = kda_fused_layer.kda_layer_fwd(
                xs, ws, H, backend="fused", cu_seqlens=cu, lower_bound=lower_bound
            )
            y.backward(dy)
            outs[mode] = [y.detach(), xs.grad, *(t.grad for t in ws.tensors())]
            pieces_used = dict(kda_fused_layer.LAST_PIECES)
        tol = 2e-2 if dtype != torch.float32 else 1e-5
        errs = [
            (
                (a.float() - b.float()).abs().max()
                / b.float().abs().max().clamp_min(1e-6)
            ).item()
            for a, b in zip(outs["1"], outs["0"])
        ]
        assert max(errs) <= tol, f"save_fwd=1 vs recompute: max rel err {max(errs):.2e}"
        return {
            "max rel err": max(errs),
            "chunk_fwd": pieces_used.get("chunk_fwd"),
            "chunk_bwd": pieces_used.get("chunk_bwd"),
            "fwd_intermediates": pieces_used.get("chunk_bwd_fwd_intermediates", "-"),
        }
    finally:
        kda_fused_layer._fusion_overrides.clear()
        kda_fused_layer._fusion_overrides.update(saved)
        kda_fused_layer._piece_overrides.clear()
        kda_fused_layer._piece_overrides.update(pieces)


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
        "-m",
        "--proj_tokens",
        type=int,
        nargs="*",
        # 300: K tail of dW_g; 16384: the varlen 8x2048 batch
        default=[256, 300, 1024, 4096, 16384],
        help="tokens of the fused projection table (Kimi-K3 tuned CSV range)",
    )
    parser.add_argument(
        "--wgrad_tokens",
        type=int,
        nargs="*",
        # 300: K tail
        default=[256, 300],
        help="tokens of the dW_g split_k bit-exactness table",
    )
    parser.add_argument(
        "--wgrad_tp",
        type=int,
        nargs="*",
        # TP8 (H*K = 1536) keeps split-K under its 2048-tile limit
        default=[1, 8],
        help="tensor-parallel ranks sharding H*K in the dW_g table",
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
        test_kda_weight_cache_update(torch.float32, concat_proj)
        test_kda_mxfp8_weight_cache(concat_proj)
    print(
        pd.DataFrame(
            [
                {"dtype": str(dt), "cu_seqlens": cu}
                | test_kda_save_fwd_matches_recompute(dt, cu)
                for dt in (torch.float32, torch.bfloat16)
                for cu in (None, [0, 48, 128])
            ]
        ).to_markdown(index=False)
    )

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
        if dtype == dtypes.bf16:
            # the Megatron-LM MXFP8 recipe runs on bf16 layers
            gemm_df = summarize(
                "kda mxfp8 projection GEMM summary",
                [
                    test_kda_mxfp8_gemm(proj, kind, b * t, d, h, k, r, dtype)
                    for proj, kind, b, t, (d, h, k, _, r) in itertools.product(
                        MXFP8_PROJS, MXFP8_KINDS, args.batch, args.seqlen, args.shape
                    )
                ],
                layer_axes,
            )
            bad_gemm = gemm_df[
                (gemm_df["err"] != 0)
                | ~gemm_df["quant bitexact"]
                | ~gemm_df["fastpath bitwise"]
            ]
            assert bad_gemm.empty, "mxfp8 GEMMs off the emulation:\n" + bad_gemm[
                [
                    "proj",
                    "kind",
                    "m",
                    "impl",
                    "err",
                    "quant bitexact",
                    "fastpath bitwise",
                ]
            ].to_string(index=False)
            mx_rows = []
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
                        row = test_kda_mxfp8_block(
                            b, t, nseq, d, h, k, w, r, c, lb, dtype
                        )
                        row["fusion combo"] = combo
                        mx_rows.append(row)
                finally:
                    kda_fused_layer.set_kda_fusions(**dict.fromkeys(spec))
            mx_df = summarize(
                "kda mxfp8 block fwd+bwd summary",
                mx_rows,
                {"block": args.block, "shape": args.shape, "fusions": args.fusions},
            )
            bad_mx = mx_df[mx_df["failed"] != "-"]
            assert bad_mx.empty, "mxfp8 block outside its bounds:\n" + bad_mx[
                ["t", "nseq", "fusion combo", "failed"]
            ].to_string(index=False)
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
                test_kda_proj(m, d, h, k, r, lb, dtype)
                for m, (d, h, k, _, r), lb in itertools.product(
                    args.proj_tokens, args.shape, args.gate_lb
                )
            ],
            {"proj_tokens": args.proj_tokens, "shape": args.shape},
        )
        summarize(
            "kda dW_g split_k bit-exactness summary",
            [
                test_kda_wgrad_split(m, h * k // tp, d, dtype)
                for m, tp, (d, h, k, _, _) in itertools.product(
                    args.wgrad_tokens, args.wgrad_tp, args.shape
                )
            ],
            {
                "wgrad_tokens": args.wgrad_tokens,
                "wgrad_tp": args.wgrad_tp,
                "shape": args.shape,
            },
        )


if __name__ == "__main__":
    main()
