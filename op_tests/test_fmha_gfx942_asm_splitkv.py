# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance test for gfx942 packed-varlen hd192 split-KV FMHA.

Public API:  aiter.flash_attn_varlen_func  (the path vLLM / the ticket calls)
Ops layer:   aiter.ops.mha.fmha_v3_varlen_fwd  (num_splits last; default 0)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test).

num_splits (asm_mha_varlen_fwd.cu): 0 = auto, 1 = unsplit production kernel,
2-8 = forced split-KV.  Auto uses split-3 when the kernel contract matches,
Sk >= 8192, and Q_tiles * heads <= 2 * CU.

q/k/v are packed THD, batch 1, BF16, D_QK=192 / D_V=128, non-causal — the
layout the model uses for this kernel.
"""

import triton  # noqa: F401  # isort: skip  # Must precede torch on this ROCm environment.

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_device_name, get_gfx
from aiter.ops.mha import flash_attn_varlen_func, fmha_v3_varlen_fwd
from aiter.test_common import assertAllclose, benchmark, run_perftest

torch.set_default_device("cuda")

# fmha_fwd_v3_splitkv requires gfx942 and rejects MI308.  The public wrapper
# still runs on other gfx942 cards; this file only times the split-KV kernel.
SUPPORTED_GFX = ["gfx942"]

HD_QK = 192
HD_V = 128
KV_TILE = 32

# (sq, sk, hq) — packed batch-1.  Ticket shape is last.
_SHAPES = [
    (129, 511, 4),  # Q/KV tile tails
    (129, 2048, 4),  # split-count coverage
    (4096, 8191, 12),  # last length that stays unsplit
    (4096, 8192, 12),  # first auto split-3
    (4096, 42700, 12),  # Kimi-K3 packed-varlen ticket
]


def run_torch(q, k, v, scale):
    """Packed THD reference, fp32 math.  Not timed, not in the table.

    Returns (out in q.dtype, lse in fp32 with shape [H, Sq]).
    Empty K matches the C++ empty-sequence branch: zeros and +inf LSE.
    Runs on CPU in Q tiles so the softmax reduction fits in host memory.
    """
    sq, hq, _ = q.shape
    sk = k.shape[0]
    if sk == 0:
        out = torch.zeros(sq, hq, HD_V, dtype=q.dtype, device=q.device)
        lse = torch.full((hq, sq), float("inf"), dtype=torch.float32, device=q.device)
        return out, lse
    q_cpu = q.float().cpu()
    k_h = k.float().cpu().permute(1, 2, 0).contiguous()
    v_h = v.float().cpu().permute(1, 0, 2).contiguous()
    out = torch.empty(sq, hq, HD_V, dtype=torch.float32, device="cpu")
    lse = torch.empty(hq, sq, dtype=torch.float32, device="cpu")
    q_tile = 128
    for q0 in range(0, sq, q_tile):
        q1 = min(q0 + q_tile, sq)
        q_h = q_cpu[q0:q1].permute(1, 0, 2).contiguous()
        scores = torch.bmm(q_h, k_h) * scale
        lse[:, q0:q1] = torch.logsumexp(scores, dim=-1)
        out[q0:q1] = torch.bmm(torch.softmax(scores, dim=-1), v_h).permute(1, 0, 2)
    return out.to(device=q.device, dtype=q.dtype), lse.to(device=q.device)


def _flops_bytes(hq, sq, sk, esz):
    """Attention roofline: 2 GEMMs (QK^T, PV), HBM traffic q+k+v+o."""
    flops = 2.0 * hq * sq * sk * (HD_QK + HD_V)
    nbytes = (sq * hq * HD_QK + sk * hq * HD_QK + sk * hq * HD_V + sq * hq * HD_V) * esz
    return flops, nbytes


def _empty_final_split(sk, num_splits):
    kv_tiles = (sk + KV_TILE - 1) // KV_TILE
    split_tiles = (kv_tiles + num_splits - 1) // num_splits
    return split_tiles * (num_splits - 1) >= kv_tiles


def _inputs(sq, sk, hq):
    q = torch.randn(sq, hq, HD_QK, dtype=dtypes.bf16)
    k = (
        torch.empty(0, hq, HD_QK, dtype=dtypes.bf16)
        if sk == 0
        else torch.randn(sk, hq, HD_QK, dtype=dtypes.bf16)
    )
    v = (
        torch.empty(0, hq, HD_V, dtype=dtypes.bf16)
        if sk == 0
        else torch.randn(sk, hq, HD_V, dtype=dtypes.bf16)
    )
    cu_q = torch.tensor([0, sq], dtype=torch.int32)
    cu_k = torch.tensor([0, sk], dtype=torch.int32)
    scale = 1.0 / math.sqrt(HD_QK)
    return q, k, v, cu_q, cu_k, scale


def _v3_fwd(q, k, v, cu_q, cu_k, scale, num_splits, return_lse, out=None, causal=False):
    """Full fmha_v3_varlen_fwd entry: caller args reach the impl, including out=."""
    if out is None:
        out = torch.empty(q.shape[0], q.shape[1], HD_V, dtype=q.dtype, device=q.device)
    return fmha_v3_varlen_fwd(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q.shape[0],
        k.shape[0],
        0,
        0.0,
        scale,
        0.0,
        False,
        causal,
        -1,
        -1,
        return_lse,
        False,
        1,
        out=out,
        num_splits=num_splits,
    )


def _public(q, k, v, cu_q, cu_k, scale, return_lse, out=None):
    # Preallocated out= is the buffer the model can pass through the public API.
    if out is None:
        out = torch.empty(q.shape[0], q.shape[1], HD_V, dtype=q.dtype, device=q.device)
    result = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        q.shape[0],
        k.shape[0],
        softmax_scale=scale,
        causal=False,
        return_lse=return_lse,
        out=out,
    )
    if return_lse:
        return result[0], result[1]
    return result, None


@benchmark()
def test_fmha_gfx942_asm_splitkv(sq, sk, hq, num_splits, return_lse):
    torch.manual_seed(0)
    q = torch.randn(sq, hq, HD_QK, dtype=dtypes.bf16)
    k = torch.randn(sk, hq, HD_QK, dtype=dtypes.bf16)
    v = torch.randn(sk, hq, HD_V, dtype=dtypes.bf16)
    cu_q = torch.tensor([0, sq], dtype=torch.int32)
    cu_k = torch.tensor([0, sk], dtype=torch.int32)
    scale = 1.0 / math.sqrt(HD_QK)
    out_buf = torch.empty(sq, hq, HD_V, dtype=q.dtype)

    ref_out, ref_lse = run_torch(q, k, v, scale)
    flops, nbytes = _flops_bytes(hq, sq, sk, q.element_size())

    candidates = {
        # Production unsplit ASM (num_splits=1).
        "unsplit": lambda: _v3_fwd(q, k, v, cu_q, cu_k, scale, 1, return_lse)[:2],
        # The path the model actually runs.
        "public": lambda: _public(q, k, v, cu_q, cu_k, scale, return_lse, out=out_buf),
    }
    # Forced split is rejected when the last KV partition would be empty.
    if not _empty_final_split(sk, num_splits):
        candidates["splitkv"] = lambda: _v3_fwd(
            q, k, v, cu_q, cu_k, scale, num_splits, return_lse
        )[:2]

    ret = {
        "gfx": get_gfx(),
        "cu": get_cu_num(),
        "q_wgs": ((sq + 127) // 128) * hq,
    }
    outs = {}
    for name, fn in candidates.items():
        (out, lse), us = run_perftest(fn, num_rotate_args=1)
        outs[name] = out
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us else float("nan")
        ret[f"{name} err"] = assertAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name} O sq={sq} sk={sk} hq={hq} ns={num_splits}",
        )
        if return_lse:
            assertAllclose(
                ref_lse.to(dtypes.fp32),
                lse.to(dtypes.fp32),
                rtol=2e-2,
                atol=2e-2,
                msg=f"{name} LSE sq={sq} sk={sk} hq={hq} ns={num_splits}",
            )
    auto_split = sk >= 8192 and ret["q_wgs"] <= 2 * ret["cu"]
    if "public" in outs and "unsplit" in outs:
        public_eq_unsplit = torch.equal(outs["public"], outs["unsplit"])
        ret["public_eq_unsplit"] = int(public_eq_unsplit)
        if auto_split:
            assert not public_eq_unsplit, (
                sq,
                sk,
                hq,
                "public auto should not be unsplit",
            )
        else:
            assert public_eq_unsplit, (sq, sk, hq, "public auto should stay unsplit")
    if "public" in outs and "splitkv" in outs:
        public_eq_splitkv = torch.equal(outs["public"], outs["splitkv"])
        ret["public_eq_splitkv"] = int(public_eq_splitkv)
        if auto_split and num_splits == 3:
            assert public_eq_splitkv, (sq, sk, hq, "public auto should be split-3")
    return ret


@benchmark()
def test_fmha_gfx942_asm_splitkv_empty_k(sq, hq, num_splits):
    torch.manual_seed(1)
    q = torch.randn(sq, hq, HD_QK, dtype=dtypes.bf16)
    k = torch.empty(0, hq, HD_QK, dtype=dtypes.bf16)
    v = torch.empty(0, hq, HD_V, dtype=dtypes.bf16)
    cu_q = torch.tensor([0, sq], dtype=torch.int32)
    cu_k = torch.tensor([0, 0], dtype=torch.int32)
    scale = 1.0 / math.sqrt(HD_QK)
    ref_out, _ = run_torch(q, k, v, scale)
    flops, nbytes = _flops_bytes(hq, sq, 0, q.element_size())

    candidates = {
        "unsplit": lambda: _v3_fwd(q, k, v, cu_q, cu_k, scale, 1, False)[:2],
        "splitkv": lambda: _v3_fwd(q, k, v, cu_q, cu_k, scale, num_splits, False)[:2],
        "public": lambda: _public(q, k, v, cu_q, cu_k, scale, False),
    }
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        (out, _), us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6 if us else float("nan")
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us else float("nan")
        ret[f"{name} err"] = assertAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"{name} empty-K sq={sq} hq={hq} ns={num_splits}",
        )
    return ret


def _check_empty_partition_rejected():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 511, 4)
    try:
        _v3_fwd(q, k, v, cu_q, cu_k, scale, 5, False)
    except RuntimeError as err:
        if "empty final KV partition" not in str(err):
            raise
        return
    raise AssertionError("forced split-5 on Sk=511 should reject an empty partition")


def _check_compile_outputs():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 2048, 4)
    sq, sk = q.shape[0], k.shape[0]

    def call(q, k, v):
        return fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu_q,
            cu_k,
            sq,
            sk,
            0,
            0.0,
            scale,
            0.0,
            False,
            False,
            -1,
            -1,
            True,
            False,
            1,
            num_splits=3,
        )

    eager = call(q, k, v)
    compiled = torch.compile(call, fullgraph=True)(q, k, v)
    for idx, (eager_t, compiled_t) in enumerate(zip(eager, compiled)):
        assert eager_t.dtype == compiled_t.dtype, (idx, eager_t.dtype, compiled_t.dtype)
        assert eager_t.shape == compiled_t.shape, (idx, eager_t.shape, compiled_t.shape)
    assert eager[0].dtype == q.dtype
    assert eager[1].dtype == torch.float32
    assert eager[2].dtype == q.dtype
    assert eager[3].dtype == torch.int64
    assertAllclose(
        eager[0].to(dtypes.fp32),
        compiled[0].to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg="torch.compile O",
    )
    assertAllclose(
        eager[1].to(dtypes.fp32),
        compiled[1].to(dtypes.fp32),
        rtol=2e-4,
        atol=2e-4,
        msg="torch.compile LSE",
    )


def _check_cuda_graph():
    # sk=8192 is the first auto-select length; this captures split producer + combine.
    q, k, v, cu_q, cu_k, scale = _inputs(129, 8192, 4)
    sq, sk = q.shape[0], k.shape[0]
    for _ in range(3):
        flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, sq, sk, softmax_scale=scale, causal=False
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = flash_attn_varlen_func(
            q, k, v, cu_q, cu_k, sq, sk, softmax_scale=scale, causal=False
        )
    graph.replay()
    first = captured.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, first)


def _check_forced_split_writes_out():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 2048, 4)
    out = torch.empty(q.shape[0], q.shape[1], HD_V, dtype=q.dtype)
    sentinel = out.data_ptr()
    result = _v3_fwd(q, k, v, cu_q, cu_k, scale, 3, False, out=out)
    assert result[0].data_ptr() == sentinel, "forced split must write the caller out="
    assert torch.equal(result[0], out)
    ref_out, _ = run_torch(q, k, v, scale)
    assertAllclose(
        ref_out.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg="forced split out=",
    )


def _check_forced_split_causal_rejected():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 2048, 4)
    try:
        _v3_fwd(q, k, v, cu_q, cu_k, scale, 3, False, causal=True)
    except RuntimeError as err:
        if "incompatible with this input" not in str(err):
            raise
        return
    raise AssertionError("forced split with causal=True should raise")


def _check_non_lse():
    q, k, v, cu_q, cu_k, scale = _inputs(129, 2048, 4)
    ref_out, _ = run_torch(q, k, v, scale)
    out, lse, _, _ = _v3_fwd(q, k, v, cu_q, cu_k, scale, 3, False)
    assert lse.numel() == 0
    assertAllclose(
        ref_out.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=2e-2,
        atol=2e-2,
        msg="forced split return_lse=False",
    )


def main():
    if get_gfx() not in SUPPORTED_GFX or get_device_name() == "MI308":
        aiter.logger.warning(
            "gfx942 hd192 split-KV unsupported on %s; skipping",
            get_device_name() if get_gfx() == "gfx942" else get_gfx(),
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
        default=[dtypes.bf16],
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_SHAPES,
        help="shape(s) as sq,sk,hq (default: tails, auto-select bounds, ticket)",
    )
    parser.add_argument(
        "-ns",
        "--num_splits",
        type=int,
        nargs="*",
        default=list(range(2, 9)),
        help="forced split count(s) for the splitkv candidate (default: 2..8)",
    )
    parser.add_argument(
        "--lse",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="return_lse: 0=inference 1=training (default: 1)",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning("hd192 split-KV is bf16-only; skipping %s", dtype)
            continue
        df = []
        for shape, num_splits, return_lse in itertools.product(
            args.shapes, args.num_splits, args.lse
        ):
            sq, sk, hq = shape
            df.append(
                test_fmha_gfx942_asm_splitkv(sq, sk, hq, num_splits, bool(return_lse))
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "fmha_gfx942_asm_splitkv summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

        empty_rows = [
            test_fmha_gfx942_asm_splitkv_empty_k(129, 4, num_splits)
            for num_splits in args.num_splits
        ]
        aiter.logger.info(
            "fmha_gfx942_asm_splitkv empty-K summary (markdown):\n%s",
            pd.DataFrame(empty_rows).to_markdown(index=False),
        )

    _check_empty_partition_rejected()
    _check_forced_split_writes_out()
    _check_forced_split_causal_rejected()
    _check_non_lse()
    _check_compile_outputs()
    _check_cuda_graph()


if __name__ == "__main__":
    main()
