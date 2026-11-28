# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 packed-varlen hd192 split-KV FMHA: accuracy and timing.

fmha_v3_varlen_fwd num_splits: 0 = auto, 1 = unsplit, 2-8 = forced split.
Auto uses split-3 when Sk >= 8192 and Q_tiles * heads <= 4 * CU / 3.
Inputs are packed THD, batch 1, BF16, D_QK=192 / D_V=128, non-causal.
"""

import argparse
import functools
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_device_name, get_gfx_runtime
from aiter.ops.mha import flash_attn_varlen_func, fmha_v3_varlen_fwd
from aiter.test_common import assertAllclose, benchmark, run_perftest

torch.set_default_device("cuda")

HD_QK = 192
HD_V = 128
Q_TILE = 128
KV_TILE = 32
TOL = {"rtol": 2e-2, "atol": 2e-2, "tol_err_ratio": 0}

# (sq, sk, hq), packed batch 1.
_SHAPES = [
    (129, 511, 4),  # Q/KV tile tails
    (129, 2048, 4),  # split-count coverage
    (4096, 8191, 12),  # last length that stays unsplit
    (4096, 8192, 12),  # first auto split-3
    (8192, 8192, 12),  # auto stays unsplit above the Q-grid limit
    (4096, 42700, 12),  # long-context target
]


def run_torch(q, k, v, scale):
    """FP32 CPU reference in Q tiles. Empty K gives zeros and +inf LSE."""
    sq, hq, _ = q.shape
    if k.shape[0] == 0:
        out = torch.zeros(sq, hq, HD_V, dtype=q.dtype)
        return out, torch.full((hq, sq), float("inf"))
    q_h = q.float().cpu().permute(1, 0, 2)
    k_h = k.float().cpu().permute(1, 2, 0)
    v_h = v.float().cpu().permute(1, 0, 2)
    out = torch.empty(hq, sq, HD_V)
    lse = torch.empty(hq, sq)
    for q0 in range(0, sq, Q_TILE):
        scores = torch.bmm(q_h[:, q0 : q0 + Q_TILE], k_h) * scale
        lse[:, q0 : q0 + Q_TILE] = torch.logsumexp(scores, dim=-1)
        out[:, q0 : q0 + Q_TILE] = torch.bmm(scores.softmax(dim=-1), v_h)
    return out.permute(1, 0, 2).cuda().to(q.dtype), lse.cuda()


def _inputs(sq, sk, hq, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(sq, hq, HD_QK, dtype=dtypes.bf16)
    k = torch.randn(sk, hq, HD_QK, dtype=dtypes.bf16)
    v = torch.randn(sk, hq, HD_V, dtype=dtypes.bf16)
    cu_q = torch.tensor([0, sq], dtype=torch.int32)
    cu_k = torch.tensor([0, sk], dtype=torch.int32)
    return q, k, v, cu_q, cu_k, 1.0 / math.sqrt(HD_QK)


@functools.lru_cache(maxsize=1)
def _case(sq, sk, hq):
    q, k, v, cu_q, cu_k, scale = _inputs(sq, sk, hq)
    return (q, k, v, cu_q, cu_k, scale) + run_torch(q, k, v, scale)


def _fwd(q, k, v, cu_q, cu_k, scale, num_splits, return_lse=True, out=None, **kw):
    if out is None:
        out = torch.empty(q.shape[0], q.shape[1], HD_V, dtype=q.dtype)
    # fmha_v3_varlen_fwd args: min_seqlen_q, dropout_p, scale, soft_cap,
    # zero_tensors, causal, window_left, window_right, lse, randval, bf16_cvt.
    out, lse, _, _ = fmha_v3_varlen_fwd(
        q, k, v, cu_q, cu_k, q.shape[0], k.shape[0],
        0, 0.0, scale, 0.0, False, kw.pop("causal", False), -1, -1,
        return_lse, False, 1, out=out, num_splits=num_splits, **kw,
    )  # fmt: skip
    return out, lse


def _public(q, k, v, cu_q, cu_k, scale, return_lse=True, out=None):
    result = flash_attn_varlen_func(
        q, k, v, cu_q, cu_k, q.shape[0], k.shape[0],
        softmax_scale=scale, return_lse=return_lse, out=out,
    )  # fmt: skip
    return result if return_lse else (result, None)


def _auto_splits(sq, sk, hq):
    q_wgs = (sq + Q_TILE - 1) // Q_TILE * hq
    return 3 if sk >= 8192 and q_wgs <= 4 * get_cu_num() // 3 else 1


def _empty_final_split(sk, num_splits):
    kv_tiles = (sk + KV_TILE - 1) // KV_TILE
    return (kv_tiles + num_splits - 1) // num_splits * (num_splits - 1) >= kv_tiles


def _expect_error(fn, text):
    try:
        fn()
    except RuntimeError as err:
        if text not in str(err):
            raise
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


@benchmark()
def test_fmha_gfx942_asm_splitkv(sq, sk, hq, num_splits, return_lse):
    """num_splits=0 times the public API, which must match the auto choice."""
    q, k, v, cu_q, cu_k, scale, ref_out, ref_lse = _case(sq, sk, hq)
    out = torch.empty(sq, hq, HD_V, dtype=q.dtype)
    if num_splits == 0:
        fn = lambda: _public(q, k, v, cu_q, cu_k, scale, return_lse, out=out)
    else:
        fn = lambda: _fwd(q, k, v, cu_q, cu_k, scale, num_splits, return_lse, out)
    (actual, lse), us = run_perftest(fn, num_rotate_args=1)
    flops = 2.0 * hq * sq * sk * (HD_QK + HD_V)
    tag = f"sq={sq} sk={sk} hq={hq} ns={num_splits}"
    ret = {"us": us, "TFLOPS": flops / us / 1e6 if us else float("nan")}
    ret["err"] = assertAllclose(ref_out.float(), actual.float(), **TOL, msg=f"O {tag}")
    if return_lse:
        assertAllclose(ref_lse, lse, **TOL, msg=f"LSE {tag}")
    if num_splits == 0:
        expected, _ = _fwd(q, k, v, cu_q, cu_k, scale, _auto_splits(sq, sk, hq))
        assert torch.equal(actual, expected), f"auto selection {tag}"
    return ret


def _check_rejections():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 2048, 4)
    args = (q, k, v, cu_q, cu_k, scale, 3, False)
    bad = "incompatible with this input"
    _expect_error(lambda: _fwd(*args, causal=True), bad)
    _expect_error(lambda: _fwd(*args, cu_seqlens_q_padded=cu_q), bad)
    _expect_error(lambda: _fwd(*args, cu_seqlens_k_padded=cu_k), bad)
    q, k, v, cu_q, cu_k, scale = _inputs(129, 511, 4)
    _expect_error(
        lambda: _fwd(q, k, v, cu_q, cu_k, scale, 5), "empty final KV partition"
    )


def _check_logical_intervals():
    """Split sizes come from buffer lengths; cu_seqlens bounds apply on device."""
    q, k, v, cu_q, cu_k, scale = _inputs(129, 8192, 4)
    # Logical Q shorter than the buffer: rows outside it keep the caller's out=.
    for start, end in ((0, 97), (11, 108), (0, 0)):
        cu_q.copy_(torch.tensor([start, end], dtype=torch.int32))
        ref_out, ref_lse = run_torch(q[start:end], k, v, scale)
        for num_splits, return_lse in ((0, True), (3, False), (8, True)):
            out = torch.full((129, 4, HD_V), 42.0, dtype=q.dtype)
            actual, lse = _fwd(q, k, v, cu_q, cu_k, scale, num_splits, return_lse, out)
            assert actual.data_ptr() == out.data_ptr()
            assert torch.all(out[:start] == 42.0) and torch.all(out[end:] == 42.0)
            tag = f"Q [{start},{end}) ns={num_splits}"
            assertAllclose(ref_out.float(), out[start:end].float(), **TOL, msg=tag)
            if return_lse:
                assertAllclose(ref_lse, lse[:, start:end], **TOL, msg=f"LSE {tag}")

    # Logical K shorter than the buffer: trailing splits are empty (O=0, LSE=-inf)
    # and get zero combine weight. Logical K=0 gives zeros and +inf LSE.
    cu_q.copy_(torch.tensor([0, 129], dtype=torch.int32))
    for sk_real in (100, 0):
        cu_k.copy_(torch.tensor([0, sk_real], dtype=torch.int32))
        ref_out, ref_lse = run_torch(q, k[:sk_real], v[:sk_real], scale)
        actual, lse = _fwd(q, k, v, cu_q, cu_k, scale, 3)
        assertAllclose(ref_out.float(), actual.float(), **TOL, msg=f"K={sk_real} O")
        assertAllclose(ref_lse, lse, **TOL, msg=f"K={sk_real} LSE")


def _check_graph_and_compile():
    # Sk=8192 auto-selects split-3, so the capture holds producer and combine.
    q, k, v, cu_q, cu_k, scale = _inputs(129, 8192, 4)
    for _ in range(3):
        eager, _ = _public(q, k, v, cu_q, cu_k, scale, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, _ = _public(q, k, v, cu_q, cu_k, scale, False)
    graph.replay()
    first = captured.clone()
    graph.replay()
    torch.cuda.synchronize()
    # No atomics on the split path, so replays and eager match exactly.
    assert torch.equal(captured, first) and torch.equal(captured, eager)

    def call(q, k, v):
        return fmha_v3_varlen_fwd(
            q, k, v, cu_q, cu_k, q.shape[0], k.shape[0],
            0, 0.0, scale, 0.0, False, False, -1, -1, True, False, 1, num_splits=3,
        )  # fmt: skip

    eager = call(q, k, v)
    compiled = torch.compile(call, fullgraph=True)(q, k, v)
    for e, c in zip(eager, compiled):
        assert (e.dtype, e.shape) == (c.dtype, c.shape)
    assert torch.equal(eager[0], compiled[0]) and torch.equal(eager[1], compiled[1])


def _check_auto_grid_limit():
    hq = 12
    max_tiles = (4 * get_cu_num() // 3) // hq
    for tiles, expected_splits in ((max_tiles, 3), (max_tiles + 1, 1)):
        q, k, v, cu_q, cu_k, scale = _inputs(tiles * Q_TILE, 8192, hq)
        actual, lse = _public(q, k, v, cu_q, cu_k, scale)
        expected, expected_lse = _fwd(q, k, v, cu_q, cu_k, scale, expected_splits)
        assert torch.equal(actual, expected) and torch.equal(lse, expected_lse)


def _check_backward():
    """Backward consumes the split-KV out/LSE; compare dq/dk/dv to FP32 CPU."""
    for sq, sk, hq in ((129, 8192, 4), (4096, 42700, 12)):
        q, k, v, cu_q, cu_k, scale = _inputs(sq, sk, hq, seed=3)
        for t in (q, k, v):
            t.requires_grad_(True)
        out, _ = _public(q, k, v, cu_q, cu_k, scale)
        expected, _ = _fwd(q, k, v, cu_q, cu_k, scale, _auto_splits(sq, sk, hq))
        assert torch.equal(out, expected)
        dout = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), dout)

        q_c, k_c, v_c = (
            t.detach().float().cpu().requires_grad_(True) for t in (q, k, v)
        )
        dout_c = dout.float().cpu()
        for q0 in range(0, sq, Q_TILE):
            scores = torch.einsum("qhd,khd->hqk", q_c[q0 : q0 + Q_TILE], k_c) * scale
            ref = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v_c)
            ref.backward(dout_c[q0 : q0 + Q_TILE])
        for name, actual, ref in zip("qkv", grads, (q_c.grad, k_c.grad, v_c.grad)):
            msg = f"d{name} sq={sq} sk={sk} hq={hq}"
            assertAllclose(ref, actual.float().cpu(), **TOL, msg=msg)


def _check_address_limits():
    """The split producer uses 32-bit byte offsets."""
    # Q span > 4 GiB: auto falls back to unsplit, forced split raises.
    q = torch.empty_strided((1025, 1, HD_QK), (2**21, HD_QK, 1), dtype=dtypes.bf16)
    q[0].zero_()
    q[1024].fill_(1.0)
    _, k, v, _, cu_k, scale = _inputs(1, 8192, 1)
    cu_q = torch.tensor([1024, 1025], dtype=torch.int32)
    out = torch.full((1025, 1, HD_V), 42.0, dtype=q.dtype)
    _fwd(q, k, v, cu_q, cu_k, scale, 0, False, out)
    ref_out, _ = run_torch(q[1024:], k, v, scale)
    assertAllclose(ref_out.float(), out[1024:].float(), **TOL, msg="large Q auto")
    assert torch.all(out[:1024] == 42.0)
    _expect_error(
        lambda: _fwd(q, k, v, cu_q, cu_k, scale, 3, False, out),
        "Q/K/V byte spans below 4 GiB",
    )
    # Empty K never runs the producer, so the limits do not apply.
    cu_k0 = torch.tensor([0, 0], dtype=torch.int32)
    _fwd(q, k[:0], v[:0], cu_q, cu_k0, scale, 8, False, out)
    assert torch.all(out == 0.0)

    # Split partial planes > 4 GiB: forced split raises; empty K still succeeds.
    q, k, v, _, cu_k, scale = _inputs(1, 256, 128)
    q = q.expand(32768, -1, -1)
    cu_q = torch.tensor([0, 32768], dtype=torch.int32)
    out_base = torch.full((1, 128, HD_V), 42.0, dtype=q.dtype)
    out = out_base.expand(32768, -1, -1)
    _expect_error(
        lambda: _fwd(q, k, v, cu_q, cu_k, scale, 8, False, out),
        "partial output byte span must be below 4 GiB",
    )
    _, lse = _fwd(q, k[:0], v[:0], cu_q, cu_k0, scale, 8, True, out)
    assert torch.all(out_base == 0.0) and torch.isposinf(lse).all()


def main():
    # get_gfx_runtime() reads the device, not GPU_ARCHS. MI308 is unsupported.
    gfx = get_gfx_runtime()
    if gfx != "gfx942" or get_device_name() == "MI308":
        aiter.logger.warning("gfx942 hd192 split-KV unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(description="gfx942 hd192 split-KV FMHA")
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_SHAPES,
        help="shape(s) as sq,sk,hq",
    )
    parser.add_argument(
        "-ns",
        "--num_splits",
        type=int,
        nargs="*",
        default=list(range(9)),
        help="0 = public auto, 1 = unsplit, 2-8 = forced split (default: 0..8)",
    )
    parser.add_argument(
        "--lse", type=int, nargs="*", choices=[0, 1], default=[1], help="return_lse"
    )
    args = parser.parse_args()

    rows = [
        test_fmha_gfx942_asm_splitkv(sq, sk, hq, ns, bool(lse))
        for (sq, sk, hq), ns, lse in itertools.product(
            args.shapes, args.num_splits, args.lse
        )
        if ns < 2 or not _empty_final_split(sk, ns)
    ]
    aiter.logger.info(
        "fmha_gfx942_asm_splitkv summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    _check_rejections()
    _check_logical_intervals()
    _check_graph_and_compile()
    _check_auto_grid_limit()
    _check_backward()
    _check_address_limits()


if __name__ == "__main__":
    main()
