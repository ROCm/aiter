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
        summarize(
            "kda block fwd+bwd summary",
            [
                test_kda_block(b, t, nseq, d, h, k, w, r, c, lb, dtype)
                for (b, t, nseq), (d, h, k, w, r), c, lb in itertools.product(
                    args.block, args.shape, args.chunk, args.gate_lb
                )
            ],
            {"block": args.block, "shape": args.shape, "chunk": args.chunk},
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
