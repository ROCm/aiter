# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf for FMHA varlen/group hd192 at large KV (gfx950).

Public API: aiter.flash_attn_varlen_func (model path; auto-dispatches by KV extent)
  >= 4GiB per-head K/V row -> OPUS hd192 hybrid buffer (gfx950)
  <  4GiB                  -> v3 asm group .co (fwd_hd192_hd128_bf16[_causal]_group.co)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test): mirror
test_quant.py — @benchmark + run_perftest candidate loop, a torch reference,
per-candidate us / TFLOPS / TB/s / err, a markdown summary table, and a __main__
guard so the module is importable.

Reference checks only the last ``check_rows`` query rows (global positions near
S-1). Cost is O(check_rows * S), not O(S^2).

Examples:
    python3 op_tests/test_mha_varlen_large_kv.py
    python3 op_tests/test_mha_varlen_large_kv.py -s 11500000 -c 0
    python3 op_tests/test_mha_varlen_large_kv.py -s 900000 -c 0   # fails: S too small for opus
"""

import argparse
import itertools
import os

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# hd192 hybrid-buffer OPUS + v3 group .co ship for gfx950.
SUPPORTED_GFX = ["gfx950"]

_2P23 = 2**23  # 8,388,608
_U32_LIMIT = 1 << 32
_REF_KV_CHUNK = 8192


def kv_byte_extent(
    seqlen: int, dq: int, dv: int, elem_size: int = 2
) -> tuple[int, int]:
    k_bytes = seqlen * dq * elem_size
    v_bytes = seqlen * dv * elem_size
    return k_bytes, v_bytes


def expect_backend(seqlen: int, dq: int, dv: int) -> str:
    k_bytes, v_bytes = kv_byte_extent(seqlen, dq, dv)
    if k_bytes >= _U32_LIMIT or v_bytes >= _U32_LIMIT:
        return "opus"
    return "asm_v3"


def _broadcast_kv(k, v, gqa_ratio):
    kb = k.transpose(0, 1)
    vb = v.transpose(0, 1)
    if gqa_ratio > 1:
        kb = kb.repeat_interleave(gqa_ratio, dim=0)
        vb = vb.repeat_interleave(gqa_ratio, dim=0)
    return kb, vb


def run_torch(q, k, v, q0, causal, scale, gqa_ratio, kv_chunk=_REF_KV_CHUNK):
    """Exact fp32 attention for query rows [q0:]. q/k/v: THD packed [S, h, d]."""
    qb = q.transpose(0, 1)
    kb, vb = _broadcast_kv(k, v, gqa_ratio)
    h, S, _dq = qb.shape
    dv = vb.shape[2]
    qf = qb[:, q0:, :].float()
    R = qf.shape[1]

    row_max = torch.full((h, R, 1), float("-inf"), device=q.device, dtype=torch.float32)
    softmax_denom = torch.zeros((h, R, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((h, R, dv), device=q.device, dtype=torch.float32)

    q_idx = torch.arange(q0, S, device=q.device).view(1, R, 1)
    for start in range(0, S, kv_chunk):
        end = min(start + kv_chunk, S)
        kc = kb[:, start:end, :].float()
        vc = vb[:, start:end, :].float()
        s = torch.matmul(qf, kc.transpose(-1, -2)) * scale
        if causal:
            k_idx = torch.arange(start, end, device=q.device).view(1, 1, -1)
            s = s.masked_fill(k_idx > q_idx, float("-inf"))
        m_new = torch.maximum(row_max, s.max(dim=-1, keepdim=True).values)
        p = torch.exp(s - m_new)
        alpha = torch.exp(row_max - m_new)
        softmax_denom = softmax_denom * alpha + p.sum(dim=-1, keepdim=True)
        acc = acc * alpha + torch.matmul(p, vc)
        row_max = m_new
    return (acc / softmax_denom.clamp_min(1e-20)).to(q.dtype)


def _run_varlen(q, k, v, cu_q, cu_k, max_seqlen, scale, causal):
    out = aiter.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_seqlen,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=causal,
        window_size=(-1, -1, 0),
        return_lse=False,
        return_attn_probs=False,
    )
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _flops_bytes(S, nheads, nheads_k, dq, dv, causal, elem_size):
    flops = 4.0 * nheads * S * S * (dq + dv)
    if causal:
        flops /= 2.0
    nbytes = (
        S * nheads * dq + S * nheads_k * dq + S * nheads_k * dv + S * nheads * dv
    ) * elem_size
    return flops, nbytes


@benchmark()
def test_mha_varlen_large_kv(
    nheads, gqa_ratio, seqlen, causal, dq, dv, check_rows, seed
):
    assert nheads % gqa_ratio == 0
    nheads_k = nheads // gqa_ratio
    S = seqlen
    scale = dq**-0.5
    dtype = dtypes.bf16
    q0 = max(0, S - check_rows)

    torch.manual_seed(seed)
    q = torch.randn(S, nheads, dq, dtype=dtype)
    k = torch.randn(S, nheads_k, dq, dtype=dtype)
    v = torch.randn(S, nheads_k, dv, dtype=dtype)
    cu_q = torch.tensor([0, S], dtype=torch.int32)
    cu_k = torch.tensor([0, S], dtype=torch.int32)

    k_bytes, v_bytes = kv_byte_extent(S, dq, dv, q.element_size())
    backend = expect_backend(S, dq, dv)
    assert (
        backend == "opus"
    ), f"S={S}: KV k={k_bytes} v={v_bytes} < 4GiB; use -s >= ~11185853 for opus"
    flops, nbytes = _flops_bytes(S, nheads, nheads_k, dq, dv, causal, q.element_size())

    candidates = {
        "varlen": lambda: _run_varlen(q, k, v, cu_q, cu_k, S, scale, causal),
    }

    ret = {
        "gfx": get_gfx(),
        "backend": backend,
        "k_bytes": k_bytes,
        "v_bytes": v_bytes,
        "check_rows": check_rows,
        "q0": q0,
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_iters=1, num_warmup=0, use_cuda_event=True)
        ref = run_torch(q, k, v, q0, causal, scale, gqa_ratio)
        out_last = out[q0:].transpose(0, 1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.to(dtypes.fp32),
            out_last.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} hd192 large-kv S={S} causal={causal} backend={backend}",
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mha_varlen_large_kv unsupported on %s; skipping", get_gfx()
        )
        return

    # Enable OPUS dispatch (only AITER_DISABLE_FMHA_OPUS=1 turns it off).
    os.environ["AITER_DISABLE_FMHA_OPUS"] = "0"

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
        help="Data type (bf16 only for hd192 path).\ne.g.: -d bf16",
    )
    parser.add_argument(
        "-n",
        "--nheads",
        type=int,
        nargs="*",
        default=[1],
        help="Number of Q heads.\ne.g.: -n 1",
    )
    parser.add_argument(
        "-gr",
        "--gqa_ratio",
        type=int,
        nargs="*",
        default=[1],
        help="GQA ratio (nheads // gqa_ratio = nheads_k).\ne.g.: -gr 1",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[11_500_000],
        help="Single-sequence length S (sq == sk == S).\n"
        "Default 11500000 (>=4GiB K row, opus hd192 hybrid buffer).\n"
        "Requires S >= ~11185853 so per-head K row reaches 4GiB.\n"
        "e.g.: -s 11500000",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="Causal mode(s): 0=non-causal 1=causal (default: 0).\ne.g.: -c 0 1",
    )
    parser.add_argument(
        "-d_qk_v",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(192, 128)],
        help="Query/key and value head dims (hd192/hd128).\ne.g.: -d_qk_v 192,128",
    )
    parser.add_argument(
        "--check_rows",
        type=int,
        nargs="*",
        default=[128],
        help="Last N query rows to compare vs torch ref (default: 128).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="*",
        default=[0],
        help="RNG seed(s).\ne.g.: --seed 0",
    )
    args = parser.parse_args()
    causal_modes = [bool(c) for c in args.causal]

    for dtype in args.dtype:
        if dtype != dtypes.bf16:
            aiter.logger.warning(
                "hd192 large-kv path is bf16-only; skipping dtype %s", dtype
            )
            continue
        df = []
        for (
            (dq, dv),
            nheads,
            gqa_ratio,
            seqlen,
            causal,
            check_rows,
            seed,
        ) in itertools.product(
            args.d_qk_v,
            args.nheads,
            args.gqa_ratio,
            args.seqlen,
            causal_modes,
            args.check_rows,
            args.seed,
        ):
            if nheads % gqa_ratio != 0:
                continue
            df.append(
                test_mha_varlen_large_kv(
                    nheads, gqa_ratio, seqlen, causal, dq, dv, check_rows, seed
                )
            )
        if not df:
            continue
        df = pd.DataFrame(df)
        aiter.logger.info(
            "mha_varlen_large_kv summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
