# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA chunk backward (``chunk_kimi_delta_attn`` autograd) at Kimi-K3 shapes.

Kimi-K3 KDA: 96 heads x head_dim 128 (K = V = 128), hidden 7168. The hidden size
only feeds the projections around the op; the chunk kernel sees
``[B, T, H, 128]``. ``-H`` sweeps the per-rank head count: 96 (tp1), 48 (tp2),
12 (tp8), plus H32 / H16, the heads of the Lumen#35 timing table
(1x8192 H32, 2x4096 H16, 1x32768 H16, varlen 8x2048 H16; all safe_gate).

``-s`` takes ``(b, t, varlen)``: ``varlen=0`` is a dense ``[b, t]`` batch,
``varlen=1`` packs ``b`` sequences of ``t`` tokens into one row with
``cu_seqlens``. ``-m`` selects the gate/prologue mode:

* ``plain``: l2-normalized q/k, a precomputed log decay and post-sigmoid beta.
* ``kimi``: the Lumen#35 setting. Raw q/k, gate and beta logits, with the
  l2norm, the ``lower_bound=-5`` gate (``safe_gate``) and the beta sigmoid in
  kernel.

Gradients of q, k, v, g and beta are graded against autograd through an fp32
chunked (WY / triangular-solve) reference, adapted from ``chunk_kda_torch`` in
the fused-layer backward draft (ROCm/aiter#8739). The reference runs one
sequence and one head group at a time, so 1x32768 at H96 fits one GPU.

Candidates: ``triton`` (the #6498 backward, baseline) and ``flydsl``
(``bwd_backend="flydsl"``: every backward stage on FlyDSL, gfx950 only; the
forward intermediates are recomputed by the Triton prologue in both).
"""

import argparse
import importlib.util
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HAS_FLYDSL = importlib.util.find_spec("flydsl") is not None
CHUNK = 64
GRADS = ("q", "k", "v", "g", "beta")
LOWER_BOUND = -5.0
KIMI_OPTS = {
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
    "use_beta_sigmoid_in_kernel": True,
    "safe_gate": True,
    "lower_bound": LOWER_BOUND,
}
# Reference tokens x heads per autograd call; ~32 GB of fp32 intermediates.
REF_TOKEN_HEADS = 1 << 18


def run_torch(q, k, v, g, beta, scale, chunk_size=CHUNK):
    """fp32 chunked gated delta rule with per-channel log-space decay ``g``.

    q, k, g: [B, T, H, K]; v: [B, T, H, V]; beta: [B, T, H] (post-sigmoid).
    Zero initial state. Returns o [B, T, H, V] in fp32.
    """
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        # zero k/v/beta and zero decay keep the state unchanged on padded steps
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    N = (T + pad) // C

    def chunks(t):  # [B, T, H, ...] -> [B, H, N, C, ...]
        return t.reshape(B, N, C, H, *t.shape[3:]).transpose(1, 3).transpose(2, 3)

    q, k, v, g, beta = map(chunks, (q, k, v, g, beta))
    G = g.cumsum(-2)
    strict = torch.ones(C, C, dtype=torch.bool).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool).tril()
    dG = G.unsqueeze(-2) - G.unsqueeze(-3)
    decay = dG.masked_fill(~incl[..., None], float("-inf")).exp()
    A_kk = torch.einsum("bhnik,bhnjk,bhnijk->bhnij", k, k, decay) * strict
    A_qk = torch.einsum("bhnik,bhnjk,bhnijk->bhnij", q, k, decay)
    L = torch.eye(C) + beta.unsqueeze(-1) * A_kk
    k_end = k * (G[..., -1:, :] - G).exp()
    q_g = q * G.exp()
    k_g = k * G.exp()

    S = q.new_zeros(B, H, K, V)
    out = []
    for n in range(N):
        rhs = beta[:, :, n, :, None] * (v[:, :, n] - k_g[:, :, n] @ S)
        u = torch.linalg.solve_triangular(
            L[:, :, n], rhs, upper=False, unitriangular=True
        )
        out.append(q_g[:, :, n] @ S + A_qk[:, :, n] @ u)
        S = G[:, :, n, -1, :, None].exp() * S + k_end[:, :, n].transpose(-1, -2) @ u
    o = torch.stack(out, 2) * scale
    return o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)[:, :T]


def run_torch_kimi(q, k, v, g, beta, A_log, dt_bias, scale):
    """``run_torch`` behind the in-kernel prologues of ``KIMI_OPTS``."""
    H, K = q.shape[-2:]

    def l2norm(x):
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    gate = LOWER_BOUND * torch.sigmoid(
        A_log.exp().view(H, 1) * (g.float() + dt_bias.view(H, K))
    )
    return run_torch(
        l2norm(q.float()), l2norm(k.float()), v, gate, torch.sigmoid(beta), scale
    )


def ref_grads(x, do, scale, mode, segments):
    """fp32 autograd gradients of q/k/v/g/beta, one sequence and head group per call.

    ``segments`` lists ``(row, start, end)`` token ranges with a zero initial
    state each: every batch row (dense) or every packed sequence (varlen).
    """
    grads = {n: torch.zeros_like(x[n], dtype=dtypes.fp32) for n in GRADS}
    H, K = x["q"].shape[-2:]
    for row, s0, s1 in segments:
        hg = max(1, min(H, REF_TOKEN_HEADS // (s1 - s0)))
        for h0 in range(0, H, hg):
            h1 = min(H, h0 + hg)
            sl = (slice(row, row + 1), slice(s0, s1), slice(h0, h1))
            leaves = [x[n][sl].float().requires_grad_(True) for n in GRADS]
            if mode == "kimi":
                o = run_torch_kimi(
                    *leaves, x["A_log"][h0:h1], x["dt_bias"][h0 * K : h1 * K], scale
                )
            else:
                o = run_torch(*leaves, scale)
            for n, gr in zip(GRADS, torch.autograd.grad(o, leaves, do[sl].float())):
                grads[n][sl] = gr
            del o, leaves
    return [grads[n] for n in GRADS]


def make_inputs(b, t, h, d, dtype, mode, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtypes.fp32)

    if mode == "kimi":
        return {
            "q": r(b, t, h, d).to(dtype),
            "k": r(b, t, h, d).to(dtype),
            "v": r(b, t, h, d).to(dtype),
            "g": r(b, t, h, d).to(dtype),
            "beta": r(b, t, h),
            "A_log": r(h) * 0.5,
            "dt_bias": r(h * d) * 0.5,
        }
    # l2-normalized q/k and a per-channel negative log decay keep the recurrence
    # bounded over thousands of steps, as the in-kernel l2norm and gate do.
    return {
        "q": F.normalize(r(b, t, h, d), dim=-1).to(dtype),
        "k": F.normalize(r(b, t, h, d), dim=-1).to(dtype),
        "v": r(b, t, h, d).to(dtype),
        "g": F.logsigmoid(r(b, t, h, d) + 3).to(dtype),
        "beta": torch.sigmoid(r(b, t, h)),
    }


@benchmark()
def test_kda_chunk_bwd(b, t, varlen, h, d, mode, dtype):
    scale = d**-0.5
    # varlen packs the b sequences of t tokens into one row.
    rows, tokens = (1, b * t) if varlen else (b, t)
    x = make_inputs(rows, tokens, h, d, dtype, mode)
    do = torch.randn(
        rows, tokens, h, d, generator=torch.Generator(device="cuda").manual_seed(7)
    )
    cu_seqlens = None
    if varlen:
        cu_seqlens = torch.arange(0, b * t + 1, t, dtype=torch.int32)
        segments = [(0, i * t, (i + 1) * t) for i in range(b)]
    else:
        segments = [(i, 0, t) for i in range(b)]
    ref = ref_grads(x, do, scale, mode, segments)

    do_k = do.to(dtype)
    opts = KIMI_OPTS if mode == "kimi" else {}
    params = {n: x[n] for n in ("A_log", "dt_bias") if n in x}

    def bwd(backend):
        leaves = {n: x[n].detach().clone().requires_grad_(True) for n in GRADS}
        o, _ = chunk_kimi_delta_attn(
            **leaves,
            **params,
            scale=scale,
            chunk_size=CHUNK,
            cu_seqlens=cu_seqlens,
            bwd_backend=backend,
            **opts,
        )
        inputs = [leaves[n] for n in GRADS]
        return lambda: torch.autograd.grad(o, inputs, do_k, retain_graph=True)

    candidates = {"triton": bwd("triton")}
    # The FlyDSL stages need gfx950's LDS; elsewhere the call falls back to
    # Triton, which would only duplicate the baseline.
    if HAS_FLYDSL and get_gfx_runtime() == "gfx950":
        candidates["flydsl"] = bwd("flydsl")

    # Nominal count: ~2x the chunked forward's matmul work
    # (intra-chunk C x C scores/WY plus K x V state updates per token).
    flops = 2 * rows * h * tokens * (4 * CHUNK * d + 2 * CHUNK * d + 6 * d * d)
    nbytes = sum(x[n].numel() * x[n].element_size() * 2 for n in GRADS)
    nbytes += do_k.numel() * do_k.element_size()

    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        grads, us = run_perftest(fn, num_iters=20, num_warmup=2)
        errs = []
        for n, gk, gr in zip(GRADS, grads, ref):
            assert torch.isfinite(gk).all(), f"{name}: d{n} not finite"
            err = checkAllclose(
                gr.to(dtypes.fp32),
                gk.to(dtypes.fp32),
                rtol=5e-2,
                atol=5e-2,
                msg=f"{name}: kda chunk bwd d{n} [b={b} t={t} varlen={varlen} h={h} {mode}]",
            )
            assert err == 0, f"{name}: d{n} {err:.2%} outside rtol/atol"
            errs.append(err)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(errs)
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "kda chunk bwd unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    if not HAS_FLYDSL:
        aiter.logger.warning(
            "flydsl not installed; benchmarking the triton backend only"
        )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="KDA chunk backward at Kimi-K3 shapes",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        help="input dtype, e.g. -d bf16",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1000, 0),  # partial last chunk
            (1, 2048, 0),
            (1, 8192, 0),  # Lumen#35 timing shapes from here on
            (2, 4096, 0),
            (1, 32768, 0),
            (8, 2048, 1),
        ],
        help="(b, t, varlen): varlen=1 packs b sequences of t tokens, e.g. -s 8,2048,1",
    )
    parser.add_argument(
        "-H",
        "--heads",
        type=int,
        nargs="*",
        default=[96, 48, 32, 16, 12],
        help="per-rank KDA heads: K3 96 (tp1), 48 (tp2), 12 (tp8); Lumen#35 32, 16",
    )
    parser.add_argument("--head-dim", type=int, nargs="*", default=[128])
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        nargs="*",
        default=["plain", "kimi"],
        choices=["plain", "kimi"],
        help="plain: precomputed gate; kimi: l2norm / lower-bound gate / beta sigmoid in kernel",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = [
            test_kda_chunk_bwd(b, t, bool(vl), h, d, mode, dtype)
            for (b, t, vl), h, d, mode in itertools.product(
                args.shape, args.heads, args.head_dim, args.mode
            )
        ]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "kda chunk bwd summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
