# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + performance test for the native HIP D64 bf16 split-K FMHA forward.

Public API:  aiter.flash_attn_func(..., num_splits=N)  (the path the model calls)
Ops layer:   aiter.ops.mha.mha_fwd_native_splitkv      (registered as
                                                        torch.ops.aiter.mha_fwd_native_splitkv)

Built to the aiter op-test standard (see .claude/skills/aiter-op-test).

num_splits (aiter/ops/mha.py, _flash_attn_forward): 0 = _native_splitkv_heuristic
picks G, 1 = forced no-split (falls through to CK), N>1 = forced split-K.  The
native path also needs can_impl_fmha_native(): gfx942, dense bf16, D64, no
bias/alibi/swa/dropout/sink/varlen, and sk >= sq when causal.

Split-K only pays off when the workgroup count `batch*hq*ceil(sq/128)` cannot fill
the machine but KV is long -- decode / short-query against a long context.  The
shape table sits in that region; the `heuristic_ns` column shows what production
would pick per row, so a forced `num_splits` the heuristic would refuse is visible
in the table rather than implied.

q/k/v are allocated contiguous here.  The dispatcher's BSHD/BHSD/SBHD handling is
swept by test_mha.py; this file targets the split-K math.
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.mha import _native_splitkv_heuristic, mha_fwd_native_splitkv
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# can_impl_fmha_native() gates on get_gfx() == "gfx942".  The kernel is wave64
# (runner/params.hpp hardcodes kWarpSize = 64) and uses mfma_f32_32x32x8bf16_1k,
# so gfx1250 (wave32) cannot run it, and the build config carries no arch
# restriction -- without this allow-list the sweep would build and launch anywhere.
SUPPORTED_GFX = ["gfx942"]

D = 64  # native splitkv is D64 only

# (batch, sq, sk, hq, hk) -- decode / short-query against long KV.
_SHAPES = [
    (1, 1, 4096, 64, 8),  # decode, long ctx
    (1, 1, 16384, 64, 8),  # decode, longer ctx
    (4, 1, 8192, 64, 8),  # decode, batched -- heuristic declines to split here
    (1, 128, 8192, 8, 8),  # short prefill chunk, MHA
]


def run_torch(q, k, v, *, scale, causal):
    """bshd-in / bshd-out reference, fp32 math.  Not timed, not in the table.

    Returns (out in q.dtype, lse in fp32).  lse is the natural-log logsumexp of the
    scaled scores -- the convention the kernel writes (op_epilog.hpp scales its
    base-2 accumulator by ln2 before storing).
    """
    _b, sq, hq, _d = q.shape
    _, sk, hk, _ = k.shape
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=2)
        v = v.repeat_interleave(hq // hk, dim=2)
    scores = torch.einsum("bshd,bkhd->bhsk", q.float(), k.float()) * scale
    if causal:
        # Bottom-right aligned: query i attends keys <= i + (sk - sq), matching
        # mask_shift = seqlen_k - seqlen_q in fused/pipeline.hpp.
        m = torch.triu(
            torch.ones(sq, sk, dtype=torch.bool, device=q.device), sk - sq + 1
        )
        scores = scores.masked_fill(m, float("-inf"))
    max_attn, _ = scores.max(dim=-1)
    shifted = torch.exp(scores - max_attn.unsqueeze(-1))
    denom = shifted.sum(dim=-1)
    out = torch.einsum("bhsk,bkhd->bshd", shifted / denom.unsqueeze(-1), v.float())
    return out.to(q.dtype), torch.log(denom) + max_attn


def _flops_bytes(batch, hq, hk, sq, sk, d, causal, esz):
    """Attention roofline numerators: 2 GEMMs (QK^T, PV), HBM traffic q+k+v+o."""
    flops = 4.0 * batch * hq * sq * sk * d  # 2*(2*M*N*K) over the two matmuls
    if causal:
        # Exact only when sq == sk; a bottom-right rectangular mask keeps more.
        # Kept as /2 to stay comparable with test_fmha_fwd_with_sink_asm.py.
        flops /= 2.0
    nbytes = (
        2 * batch * sq * hq * d  # q read + o write
        + 2 * batch * sk * hk * d  # k + v read
    ) * esz
    return flops, nbytes


def _fa(q, k, v, *, scale, causal, return_lse, num_splits):
    """The model path: aiter.flash_attn_func.  Returns (out, lse or None)."""
    r = aiter.flash_attn_func(
        q,
        k,
        v,
        softmax_scale=scale,
        causal=causal,
        return_lse=return_lse,
        num_splits=num_splits,
    )
    return (r[0], r[1]) if return_lse else (r, None)


@benchmark()
def test_mha_native_splitkv(batch, sq, sk, hq, hk, causal, return_lse, num_splits):
    torch.manual_seed(0)
    q = torch.randn(batch, sq, hq, D, dtype=dtypes.bf16)
    k = torch.randn(batch, sk, hk, D, dtype=dtypes.bf16)
    v = torch.randn(batch, sk, hk, D, dtype=dtypes.bf16)
    scale = 1.0 / math.sqrt(D)
    out_buf = torch.empty(batch, sq, hq, D, dtype=q.dtype)

    ref_out, ref_lse = run_torch(q, k, v, scale=scale, causal=causal)
    flops, nbytes = _flops_bytes(batch, hq, hk, sq, sk, D, causal, q.element_size())

    kw = {"scale": scale, "causal": causal, "return_lse": return_lse}
    candidates = {
        # The CK kernel the heuristic decides against.
        "ck": lambda: _fa(q, k, v, num_splits=1, **kw),
        # Forced split-K, reached the way the model reaches it.
        "splitkv": lambda: _fa(q, k, v, num_splits=num_splits, **kw),
        # Same kernel without the dispatcher, into a preallocated `out` (no
        # _flash_attn_forward caller passes out=, so nothing else covers it).
        # This name is the registered op's dispatcher, so the candidate also
        # asserts the torch.ops.aiter.mha_fwd_native_splitkv schema still matches.
        "splitkv_op": lambda: mha_fwd_native_splitkv(
            q, k, v, out_buf, scale, causal, return_lse, num_splits
        ),
    }

    ret = {
        "gfx": get_gfx(),
        "cu": get_cu_num(),  # heuristic reads it: MI300X=304 vs MI308X=80 change G
        "heuristic_ns": _native_splitkv_heuristic(batch, hq, sq, sk, get_cu_num()),
        # fp32 partial-O [G,B,Hq,Sq,D] + partial-LSE [G,B,Hq,Sq], written then read
        # back by combine -- the traffic the heuristic trades against, and the one
        # cost the roofline nbytes above cannot see.
        "scratch MB": 2 * 4 * num_splits * batch * hq * sq * (D + 1) / 1e6,
    }
    for name, fn in candidates.items():
        # num_rotate_args=1: the candidates are zero-arg closures, so rotation
        # cannot defeat L2 anyway, and the auto path would deepcopy the long KV.
        (out, lse), us = run_perftest(fn, num_rotate_args=1)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err(O)"] = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} O sk={sk} causal={causal} ns={num_splits}",
        )
        if return_lse:
            ret[f"{name} err(LSE)"] = checkAllclose(
                ref_lse.to(dtypes.fp32),
                lse.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                msg=f"{name} LSE sk={sk} causal={causal} ns={num_splits}",
            )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mha_fwd_native_splitkv unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_SHAPES,
        help="shape(s) as batch,sq,sk,hq,hk (default: the decode/long-ctx table)",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="causal mode(s): 0=non-causal 1=causal (default: 0 1)",
    )
    parser.add_argument(
        "--lse",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="return_lse: 0=inference 1=training (default: 0 1)",
    )
    parser.add_argument(
        "-ns",
        "--num_splits",
        type=int,
        nargs="*",
        default=[8],
        help="forced split count(s) for the splitkv candidates (default: 8)",
    )
    args = parser.parse_args()

    df = []
    for shape, causal, return_lse, num_splits in itertools.product(
        args.shapes, args.causal, args.lse, args.num_splits
    ):
        batch, sq, sk, hq, hk = shape
        if causal and sk < sq:
            continue  # can_impl_fmha_native rejects sq > sk causal
        df.append(
            test_mha_native_splitkv(
                batch, sq, sk, hq, hk, bool(causal), bool(return_lse), num_splits
            )
        )
    df = pd.DataFrame(df)
    aiter.logger.info(
        "mha_fwd_native_splitkv summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
