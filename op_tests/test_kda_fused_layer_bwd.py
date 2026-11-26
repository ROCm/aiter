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


def make_weights(D, H, K, W, R, dtype):
    def p(*shape, std):
        return (torch.randn(*shape) * std).to(dtype)

    return KDALayerWeights(
        w_q=p(H * K, D, std=D**-0.5),
        w_k=p(H * K, D, std=D**-0.5),
        w_v=p(H * K, D, std=D**-0.5),
        conv_q=p(H * K, W, std=0.5),
        conv_k=p(H * K, W, std=0.5),
        conv_v=p(H * K, W, std=0.5),
        f_a=p(R, D, std=D**-0.5),
        f_b=p(H * K, R, std=R**-0.5),
        # distinct per-head decay rates so a head/channel mix-up shows
        A_log=torch.linspace(-2.0, 1.0, H).to(dtype),
        dt_bias=p(H * K, std=1.0),
        w_beta=p(H, D, std=D**-0.5),
        g_proj=p(H * K, D, std=D**-0.5),
        norm_w=(1.0 + torch.randn(K) * 0.1).to(dtype),
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


def run_torch(dy, x, w, num_heads, eps=1e-6):
    """Reference layer backward: recurrent fwd + autograd, fp32."""
    xs = x.detach().float().requires_grad_(True)
    ws = [t.detach().float().requires_grad_(True) for t in w.tensors()]
    wr = KDALayerWeights.from_tensors(ws)
    q, k, v, g, beta = kda_layer_preprocess(xs, wr, num_heads, eps)
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
def test_kda_layer_bwd(b, t, d, h, k, w, r, chunk, dtype):
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    ref_dx, ref_dw = run_torch(dy, x, wt, h)

    candidates = {
        "torch_chunk": lambda: kda_layer_bwd(
            dy, x, wt, h, chunk_size=chunk, backend="torch"
        ),
        # gfx950 bf16: FlyDSL GEMM epilogues for f_b/g_proj, FlyDSL l2norm,
        # #8741 RMSNorm; conv (#8726) and chunk (#8740) bwd are torch until
        # those land.
        "fused": lambda: kda_layer_bwd(dy, x, wt, h, chunk_size=chunk, backend="fused"),
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


def run_torch_block(dy, x, w, num_heads, eps=1e-6):
    """Reference block: recurrent forward output and all gradients, fp32."""
    xs = x.detach().float().requires_grad_(True)
    ws = [t.detach().float().requires_grad_(True) for t in w.tensors()]
    wr = KDALayerWeights.from_tensors(ws)
    q, k, v, g, beta = kda_layer_preprocess(xs, wr, num_heads, eps)
    o = recurrent_kda_ref(q, k, v, g, beta, q.shape[-1] ** -0.5)
    y = kda_layer_postprocess(o, xs, wr, eps)
    grads = torch.autograd.grad(y, [xs, *ws], dy.float())
    return y.detach(), grads


@benchmark()
def test_kda_block(b, t, d, h, k, w, r, chunk, dtype):
    """End to end: kda_layer_fwd + autograd backward of the whole layer."""
    x = torch.randn(b, t, d).to(dtype)
    dy = torch.randn(b, t, d).to(dtype)
    wt = make_weights(d, h, k, w, r, dtype)
    ref_y, ref_g = run_torch_block(dy, x, wt, h)

    def block(backend):
        def run():
            xs = x.detach().requires_grad_(True)
            ws = [p.detach().requires_grad_(True) for p in wt.tensors()]
            y = kda_layer_fwd(
                xs, KDALayerWeights.from_tensors(ws), h, chunk, backend=backend
            )
            return y, torch.autograd.grad(y, [xs, *ws], dy)

        return run

    flops = layer_flops(b, t, d, h, k, w, r, chunk) * 4 // 3  # fwd + recompute + bwd
    nbytes = sum(p.numel() for p in wt.tensors()) * 2 * x.element_size()
    nbytes += (3 * b * t * d + 6 * b * t * h * k) * x.element_size()
    tol = 2e-2 if dtype == dtypes.fp32 else 5e-2
    names = ["y", "dx"] + [
        f.name for f in KDALayerWeights.__dataclass_fields__.values()
    ]
    ret = {"gfx": get_gfx_runtime()}
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


def run_torch_proj(lr, f_b, dt_bias, A_log, dg, x, g_proj, aux):
    """Reference for the fused projections, fp32."""
    H = A_log.numel()
    z = lr.float() @ f_b.float().t() + dt_bias.float()
    neg_a = -A_log.float().exp().repeat_interleave(f_b.shape[0] // H)
    sp, sig = torch.nn.functional.softplus(z), torch.sigmoid(z)
    gate_out = torch.sigmoid(x.float() @ g_proj.float().t())
    return {
        "gate": neg_a * sp,
        "gate_bwd": (dg * neg_a * sig, dg * neg_a * sp),
        "sigmoid_mul": gate_out * aux,
        "gproj_bwd": (aux @ g_proj.float(), aux.t() @ x.float()),
    }


@benchmark()
def test_kda_proj(m, d, h, k, r, dtype):
    from aiter.ops.flydsl import kda_layer_bwd as fk

    hk = h * k
    lr = torch.randn(m, r).to(dtype)
    f_b = (torch.randn(hk, r) * r**-0.5).to(dtype)
    dt_bias = torch.randn(hk).to(dtype)
    A_log = torch.linspace(-2.0, 1.0, h)
    dg = torch.randn(m, hk)
    x = torch.randn(m, d).to(dtype)
    g_proj = (torch.randn(hk, d) * d**-0.5).to(dtype)
    aux = torch.randn(m, hk)
    ref = run_torch_proj(lr, f_b, dt_bias, A_log, dg, x, g_proj, aux)
    candidates = {
        "gate": (
            lambda: fk.kda_gate_proj(lr, f_b, dt_bias, A_log),
            2 * m * r * hk,
            (m * r + hk * r) * 2 + m * hk * 4,
        ),
        "gate_bwd": (
            lambda: fk.kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, dg),
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
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for b, t, (d, h, k, w, r), c in itertools.product(
            args.batch, args.seqlen, args.shape, args.chunk
        ):
            df.append(test_kda_layer_bwd(b, t, d, h, k, w, r, c, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "kda fused layer bwd summary (markdown):\n%s", df.to_markdown(index=False)
        )
        df = []
        for b, t, (d, h, k, w, r), c in itertools.product(
            args.batch, args.seqlen, args.shape, args.chunk
        ):
            df.append(test_kda_block(b, t, d, h, k, w, r, c, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "kda block fwd+bwd summary (markdown):\n%s", df.to_markdown(index=False)
        )

    # the fused projections are the gfx950 bf16/fp16 GEMM with an epilogue
    proj_dtypes = [dt for dt in args.dtype if dt in (dtypes.bf16, dtypes.fp16)]
    if get_gfx_runtime() != "gfx950" or not proj_dtypes:
        aiter.logger.warning(
            "kda projection fusion needs gfx950 and bf16/fp16; skipping its table"
        )
        return
    for dtype in proj_dtypes:
        df = []
        for b, t, (d, h, k, _, r) in itertools.product(
            args.batch, args.seqlen, args.shape
        ):
            df.append(test_kda_proj(b * t, d, h, k, r, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "kda projection fusion summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
