# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""KDA chunk forward (``chunk_kimi_delta_attn``) at Kimi-K3 shapes.

Kimi-K3 KDA: 96 heads x head_dim 128 (K = V = 128). ``-H`` sweeps the per-rank
head count: 96 (tp1), 48 (tp2), 12 (tp8), plus 32 / 16. ``-s`` takes ``BxT``
shapes; ``T = 1000`` leaves a partial last chunk, and the Lumen#35 shapes are
1x8192, 2x4096, 1x32768 and varlen 8x2048 (``8x2048v``: packed with
``cu_seqlens``, one initial / final state per sequence).

Modes (``-m``):
  * ``kimi``: the serving call -- raw gate with ``A_log`` / ``dt_bias`` and
    ``lower_bound=-5`` (safe gate), in-kernel q/k l2norm and beta sigmoid.
  * ``kimi_state``: ``kimi`` plus a V-first initial state and the final state.
  * ``plain``: pre-activated log-space gate, normalized q/k, post-sigmoid beta,
    K-first initial and final state.

Candidates: ``triton`` (default backend at ``chunk_size=64``, which keeps it on
the Triton pipeline rather than the gluon FlashKDA chunk-32 path; skipped with a
warning if it fails to run) and ``flydsl``
(``fwd_backend="flydsl"``, gfx950 only). Both are graded against an fp32
token-by-token recurrence; ``flydsl`` is also compared with ``triton``.
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
LOWER_BOUND = -5.0


def run_torch(
    q,
    k,
    v,
    g,
    beta,
    scale,
    A_log=None,
    dt_bias=None,
    lower_bound=None,
    use_gate_in_kernel=False,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    initial_state=None,
    state_v_first=False,
):
    """fp32 recurrence ``S_t = diag(exp(g_t)) S_{t-1} + beta_t k_t (v_t - k_t S)^T``.

    Returns ``o [B, T, H, V]`` and the final state ``[B, H, K, V]``
    (``[B, H, V, K]`` when ``state_v_first``), both fp32.
    """
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    B, T, H, K = q.shape
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
    if use_beta_sigmoid_in_kernel:
        beta = beta.sigmoid()
    if use_gate_in_kernel:
        z = g + (dt_bias.float().view(H, K) if dt_bias is not None else 0.0)
        a = A_log.float().exp().view(H, 1)
        if lower_bound is None:
            g = -a * F.softplus(z)
        else:
            g = lower_bound * torch.sigmoid(a * z)
    S = torch.zeros(B, H, K, v.shape[-1], dtype=torch.float32)
    if initial_state is not None:
        S = initial_state.float()
        if state_v_first:
            S = S.transpose(-1, -2)
        S = S.clone()
    o = torch.empty(B, T, H, v.shape[-1], dtype=torch.float32)
    for t in range(T):
        S = S * g[:, t].exp().unsqueeze(-1)
        kt = k[:, t]
        delta = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt * beta[:, t, :, None], delta)
        o[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S)
    if state_v_first:
        S = S.transpose(-1, -2)
    return o, S.contiguous()


def make_inputs(b, t, h, d, dtype, mode, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=dtypes.fp32)

    x = {"q": r(b, t, h, d), "k": r(b, t, h, d), "v": r(b, t, h, d)}
    kw = {"state_v_first": mode == "kimi_state"}
    if mode.startswith("kimi"):
        x["g"] = r(b, t, h, d)
        x["beta"] = r(b, t, h)
        kw.update(
            A_log=torch.log(torch.rand(h, generator=gen) * 15 + 1),
            dt_bias=r(h * d) * 0.5,
            lower_bound=LOWER_BOUND,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
    else:
        x["q"], x["k"] = (F.normalize(x[n], dim=-1) for n in ("q", "k"))
        x["g"] = F.logsigmoid(r(b, t, h, d) + 3)
        x["beta"] = torch.sigmoid(r(b, t, h))
    for n in ("q", "k", "v", "g"):
        x[n] = x[n].to(dtype)
    if mode != "kimi":
        x["initial_state"] = r(b, h, d, d) * 0.1
    return x, kw


@benchmark()
def test_kda_chunk_fwd(b, t, h, d, dtype, mode, varlen):
    """``varlen``: the ``b`` sequences of ``t`` tokens are packed into one
    ``[1, b * t]`` batch with ``cu_seqlens``; the fp32 reference still runs
    them as independent batch rows."""
    scale = d**-0.5
    x, kw = make_inputs(b, t, h, d, dtype, mode)
    o_ref, s_ref = run_torch(**x, **kw, scale=scale)
    want_state = "initial_state" in x
    call_x = dict(x)
    cu_seqlens = None
    if varlen:
        for n in ("q", "k", "v", "g", "beta"):
            call_x[n] = x[n].reshape(1, b * t, *x[n].shape[2:])
        cu_seqlens = torch.arange(0, b * t + 1, t, dtype=torch.int64)

    def fwd(backend):
        call_kw = dict(kw)
        if call_kw.get("use_gate_in_kernel"):
            call_kw["safe_gate"] = True
        if cu_seqlens is not None:
            call_kw["cu_seqlens"] = cu_seqlens

        def run():
            o, fs = chunk_kimi_delta_attn(
                **call_x,
                **call_kw,
                scale=scale,
                output_final_state=want_state,
                chunk_size=CHUNK,
                fwd_backend=backend,
            )
            return o.reshape(b, t, *o.shape[2:]), fs

        return run

    ret = {"gfx": get_gfx_runtime()}
    candidates = {"triton": fwd("triton")}
    # The FlyDSL intra kernel needs gfx950's LDS.
    if HAS_FLYDSL and get_gfx_runtime() == "gfx950":
        candidates["flydsl"] = fwd("flydsl")

    # Intra-chunk Aqk/Akk/w/u (4 x C x K per token), K5 state update and
    # w @ h (3 x K x V), output (K x V + C x V).
    flops = 2 * b * t * h * (4 * CHUNK * d + 4 * d * d + CHUNK * d)
    nbytes = sum(x[n].numel() * x[n].element_size() for n in ("q", "k", "v", "g"))
    nbytes += x["beta"].numel() * x["beta"].element_size()
    nbytes += b * t * h * d * x["v"].element_size()

    outs = {}
    for name, fn in candidates.items():
        try:
            (o, fs), us = run_perftest(fn, num_iters=20, num_warmup=2)
        except Exception as exc:
            # The Triton baseline is optional: a toolchain it does not build
            # under must not block grading FlyDSL against the fp32 reference.
            if name != "triton":
                raise
            aiter.logger.warning("triton baseline unavailable, skipped: %s", exc)
            continue
        assert torch.isfinite(o).all(), f"{name}: o not finite"
        err = checkAllclose(
            o_ref,
            o.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            tol_err_ratio=1e-3,
            msg=f"{name}: kda chunk fwd o vs fp32 [t={t} h={h} {mode}]",
        )
        assert err <= 1e-3, f"{name}: o {err:.2%} outside rtol/atol"
        max_err = (o.float() - o_ref).abs().max().item()
        assert max_err < 0.1, f"{name}: o max abs err {max_err:.3g}"
        if want_state:
            s_err = checkAllclose(
                s_ref,
                fs.to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                tol_err_ratio=1e-3,
                msg=f"{name}: kda chunk fwd final state [t={t} h={h} {mode}]",
            )
            assert s_err <= 1e-3, f"{name}: final state {s_err:.2%} outside rtol/atol"
            err = max(err, s_err)
        outs[name] = o
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} max_abs"] = max_err
    if "flydsl" in outs and "triton" in outs:
        ret["flydsl vs triton err"] = checkAllclose(
            outs["triton"].float(),
            outs["flydsl"].float(),
            rtol=2e-2,
            atol=2e-2,
            msg=f"flydsl vs triton o [t={t} h={h} {mode}]",
        )
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "kda chunk fwd unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    if not HAS_FLYDSL:
        aiter.logger.warning("flydsl not installed; testing the triton backend only")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="KDA chunk forward at Kimi-K3 shapes",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,"
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=str,
        nargs="*",
        default=[
            "1x1000",
            "1x2048",
            "1x8192",
            "2x4096",
            "1x32768",
            "8x2048v",
            "3x1000v",
        ],
        help="BxT: B sequences of T tokens; a trailing 'v' packs them varlen\n"
        "(cu_seqlens). 1000 leaves a partial last chunk; the 8192 / 2x4096 /\n"
        "32768 / varlen 8x2048 rows are the Lumen#35 shapes; 3x1000v puts\n"
        "partial chunks at every sequence boundary.",
    )
    parser.add_argument(
        "-H",
        "--heads",
        type=int,
        nargs="*",
        default=[96, 48, 32, 16, 12],
        help="per-rank KDA heads: K3 96 (tp1), 48 (tp2), 12 (tp8); 32 / 16 from\n"
        "Lumen#35",
    )
    parser.add_argument("--head-dim", type=int, nargs="*", default=[128])
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        nargs="*",
        default=["kimi", "kimi_state", "plain"],
        choices=["kimi", "kimi_state", "plain"],
    )
    args = parser.parse_args()

    shapes = []
    for sh in args.shape:
        varlen = sh.endswith("v")
        b, t = (int(n) for n in sh.rstrip("v").split("x"))
        shapes.append((b, t, varlen))

    for dtype in args.dtype:
        df = [
            test_kda_chunk_fwd(b, t, h, d, dtype, m, varlen)
            for m, (b, t, varlen), h, d in itertools.product(
                args.mode, shapes, args.heads, args.head_dim
            )
        ]
        df = pd.DataFrame(df)
        aiter.logger.info(
            "kda chunk fwd summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
