# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness / latency test for the gfx1250 MXFP8 backend of ``mla_sparse_prefill_v41``.

The kernel (``aiter/ops/v41fp8_gfx1250.hip``) is reached through the public
``mla_sparse_prefill_v41`` entry when ``AITER_V41_FP8_GFX1250=1``. KV rows use the
DSV4.1 FP8 KV-cache layout: 528 bytes per row, 512 e4m3 values followed by 16 E8M0
scales (one per 32 values). The output is compared against an fp64 PyTorch
reference on the dequantized inputs. Cases cover DSV4.1 decode/MTP shapes, ragged
rows (rows with no keys, partial tiles, ``-1`` indices), no sink, and CUDA-graph
replay (bitwise identical to the eager run).

gfx1250 only; skipped elsewhere.

Example CLI usage::

    PYTHONPATH=. python3 op_tests/test_mla_sparse_prefill_v41_gfx1250.py
    PYTHONPATH=. python3 op_tests/test_mla_sparse_prefill_v41_gfx1250.py -t 480 4096 --bench
"""

from __future__ import annotations

import argparse
import os

import pytest
import torch

from aiter.ops.mla_sparse_prefill import mla_sparse_prefill_v41

H, D, ROWB = 64, 512, 528
_ENV = "AITER_V41_FP8_GFX1250"


def _get_gpu_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except (AttributeError, RuntimeError):
        return None


def _skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CUDA/HIP device not available"
    arch = _get_gpu_arch()
    if arch != "gfx1250":
        return f"the v41 MXFP8 HIP kernel requires gfx1250, found {arch}"
    return None


# ---------------------------------------------------------------------------
# Inputs: MXFP8 quantization (ceil-pow2 E8M0 per 32 values), 528 B KV rows.
# ---------------------------------------------------------------------------


def _mxfp8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[..., 512]`` -> (e4m3 bytes ``[..., 512]``, E8M0 bytes ``[..., 16]``)."""
    g = x.unflatten(-1, (D // 32, 32))
    amax = g.abs().amax(-1).clamp(min=2.0**-120)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    v = (g / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    return v.view(torch.uint8), (e + 127).to(torch.uint8)


def _dequant(vals: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    g = vals.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
    return (g * torch.exp2(scales.float() - 127)[..., None]).flatten(-2)


def _make_pool(rows: int, device, chunk: int = 1 << 18) -> torch.Tensor:
    """``[rows, 528]`` uint8 MXFP8 rows (values at [0, 512), scales at [512, 528))."""
    pool = torch.empty(rows, ROWB, dtype=torch.uint8, device=device)
    for s in range(0, rows, chunk):
        e = min(rows, s + chunk)
        x = torch.randn(e - s, D, device=device)
        x[:, :8] *= 4.0  # a few outlier channels
        v, sc = _mxfp8_quant(x)
        pool[s:e, :D] = v
        pool[s:e, D:] = sc
    return pool


def _build_case(
    t: int,
    device,
    mtp: int = 6,
    topk: int = 512,
    win: int = 128,
    ctx_rows: int = 8192,
    seed: int = 0,
    overlap: float = 0.9,
    n_prefix=None,
    n_extend=None,
    neg_frac: float = 0.0,
) -> dict:
    """DSV4.1-like CSR indices: per request ``mtp`` tokens share a top-k set over a
    ``ctx_rows`` prefix range (``overlap`` of it kept per token) plus a sliding
    window of ``win`` rows in the extend ring."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    nreq = -(-t // mtp)
    ring = win + mtp
    p_rows = nreq * ctx_rows
    e_rows = nreq * ring
    p_idx, p_ptr, e_idx, e_ptr = [], [0], [], [0]
    base = []
    for tok in range(t):
        r, i = divmod(tok, mtp)
        off = (r * ctx_rows) % max(1, p_rows - ctx_rows)
        if i == 0:
            base = (torch.randperm(ctx_rows, generator=g)[:topk] + off).tolist()
        n_p = topk if n_prefix is None else n_prefix(tok)
        sel = list(base[:n_p])
        for k in range(len(sel)):
            if torch.rand(1, generator=g).item() > overlap:
                sel[k] = off + int(torch.randint(ctx_rows, (1,), generator=g))
            if neg_frac > 0 and torch.rand(1, generator=g).item() < neg_frac:
                sel[k] = -1
        p_idx += sel
        p_ptr.append(len(p_idx))
        n_e = win if n_extend is None else n_extend(tok)
        pos = 1000 + i
        e_idx += [r * ring + ((pos - k) % ring) for k in range(n_e)]
        e_ptr.append(len(e_idx))

    def ti(a):
        return torch.tensor(a, dtype=torch.int32, device=device)

    return {
        "p_idx": ti(p_idx),
        "p_ptr": ti(p_ptr),
        "e_idx": ti(e_idx),
        "e_ptr": ti(e_ptr),
        "p_rows": p_rows,
        "e_rows": e_rows,
    }


def _reference(qd, pool_d, kv_d, c, sink, scale, chunk=64) -> torch.Tensor:
    """fp64 sparse attention over the dequantized inputs; ``-1`` / OOB rows are masked;
    the sink only enters the softmax denominator."""
    t = qd.shape[0]
    dev = qd.device
    out = torch.zeros(t, H, D, device=dev)
    pp, ep = c["p_ptr"].tolist(), c["e_ptr"].tolist()
    pi, ei = c["p_idx"], c["e_idx"]
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        lens = [
            max(pp[i + 1] - pp[i], 0) + max(ep[i + 1] - ep[i], 0) for i in range(s, e)
        ]
        k_max = max(max(lens), 1)
        keys = torch.zeros(e - s, k_max, D, device=dev)
        mask = torch.zeros(e - s, k_max, dtype=torch.bool, device=dev)
        for i in range(s, e):
            a = pi[pp[i] : pp[i + 1]].long()
            b = ei[ep[i] : ep[i + 1]].long()
            va = (a >= 0) & (a < pool_d.shape[0])
            vb = (b >= 0) & (b < kv_d.shape[0])
            kk = torch.cat(
                [
                    pool_d[a.clamp(0, pool_d.shape[0] - 1)],
                    kv_d[b.clamp(0, kv_d.shape[0] - 1)],
                ],
                0,
            )
            keys[i - s, : kk.shape[0]] = kk
            mask[i - s, : kk.shape[0]] = torch.cat([va, vb], 0)
        sc = torch.einsum("thd,tkd->thk", qd[s:e].double(), keys.double()) * scale
        sc = sc.masked_fill(~mask[:, None, :], float("-inf"))
        m = sc.amax(-1, keepdim=True)
        if sink is not None:
            m = torch.maximum(m, sink.double()[None, :, None])
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = torch.exp(sc - m)
        den = p.sum(-1, keepdim=True)
        if sink is not None:
            den = den + torch.exp(sink.double()[None, :, None] - m)
        o = torch.einsum("thk,tkd->thd", p, keys.double()) / den.clamp(min=1e-300)
        out[s:e] = o.float()
    return out


def _metrics(out: torch.Tensor, ref: torch.Tensor) -> dict:
    o, r = out.float().flatten(0, 1), ref.float().flatten(0, 1)
    cos = torch.nn.functional.cosine_similarity(o, r, dim=-1)
    rn, on = r.norm(dim=-1), o.norm(dim=-1)
    # an all-masked row with no sink must be exactly zero
    cos = torch.where(
        rn > 0, cos, torch.where(on == 0, torch.ones_like(cos), torch.zeros_like(cos))
    )
    return {
        "min_cos": cos.min().item(),
        "rel_l2": ((o - r).norm() / r.norm().clamp(min=1e-30)).item(),
        "max_abs": (o - r).abs().max().item(),
        "nan": int(torch.isnan(o).sum().item()),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_CASES = {
    "decode": {},
    "ragged": {
        "n_prefix": lambda t: [0, 1, 7, 8, 9, 63, 64, 65, 300, 512][t % 10],
        "n_extend": lambda t: [0, 5, 128, 1, 64, 0, 3, 128, 127, 0][(t * 7) % 10],
        "neg_frac": 0.05,
        "seed": 3,
    },
    "empty": {
        "n_prefix": lambda t: 0 if t % 3 else 17,
        "n_extend": lambda t: 0 if t % 2 == 0 else 3,
    },
}


def _inputs(t: int, case: str, use_sink: bool, device="cuda"):
    torch.manual_seed(0)
    c = _build_case(t, device, **_CASES[case])
    pool = _make_pool(c["p_rows"], device)
    ring = _make_pool(c["e_rows"], device)
    qv, qs = _mxfp8_quant(torch.randn(t, H, D, device=device))
    sink = torch.randn(H, device=device) * 2 if use_sink else None
    args = (
        qv.view(torch.float8_e4m3fn),
        qs,
        pool[:, :D].view(torch.float8_e4m3fn),
        pool[:, D:],
        c["p_idx"],
        c["p_ptr"],
        ring[:, :D].view(torch.float8_e4m3fn),
        ring[:, D:],
        c["e_idx"],
        c["e_ptr"],
        sink,
        D**-0.5,
    )
    return c, pool, ring, qv, qs, sink, args


def run_case(
    t: int, case: str, use_sink: bool = True, graph: bool = True, bench: bool = False
) -> dict:
    c, pool, ring, qv, qs, sink, args = _inputs(t, case, use_sink)
    out = torch.full((t, H, D), float("nan"), dtype=torch.bfloat16, device="cuda")
    mla_sparse_prefill_v41(*args, out=out)
    torch.cuda.synchronize()
    ref = _reference(
        _dequant(qv, qs),
        _dequant(pool[:, :D], pool[:, D:]),
        _dequant(ring[:, :D], ring[:, D:]),
        c,
        sink,
        D**-0.5,
    )
    res = _metrics(out, ref)
    first = out.clone()

    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            mla_sparse_prefill_v41(*args, out=out)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mla_sparse_prefill_v41(*args, out=out)
        same = True
        for _ in range(10):
            out.fill_(float("nan"))
            g.replay()
            torch.cuda.synchronize()
            same &= torch.equal(out.view(torch.int16), first.view(torch.int16))
        res["graph_bitwise"] = bool(same)
        if bench:
            iters = 50
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            g.replay()
            st.record()
            for _ in range(iters):
                g.replay()
            en.record()
            torch.cuda.synchronize()
            res["us"] = st.elapsed_time(en) * 1e3 / iters
    return res


def _check(res: dict):
    assert res["nan"] == 0, res
    assert res["min_cos"] > 0.9999, res
    assert res["rel_l2"] < 0.01, res
    assert res.get("graph_bitwise", True), res


@pytest.mark.parametrize(
    "t, case, use_sink",
    [
        (1, "decode", True),
        (7, "ragged", True),
        (64, "decode", True),
        (64, "decode", False),
        (64, "empty", True),
        (300, "ragged", True),
        (540, "decode", True),
    ],
)
def test_mla_sparse_prefill_v41_gfx1250(monkeypatch, t, case, use_sink):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)
    monkeypatch.setenv(_ENV, "1")
    _check(run_case(t, case, use_sink))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[7, 64, 540])
    parser.add_argument("--case", choices=list(_CASES), nargs="+", default=None)
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--bench", action="store_true", help="time graph replays")
    a = parser.parse_args()
    reason = _skip_reason()
    if reason:
        print(f"SKIP: {reason}")
        return
    os.environ[_ENV] = "1"
    for t in a.tokens:
        for case in a.case or list(_CASES):
            res = run_case(t, case, not a.no_sink, bench=a.bench)
            print(
                f"T={t} {case}: "
                + " ".join(
                    f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in res.items()
                ),
                flush=True,
            )
            _check(res)


if __name__ == "__main__":
    main()
