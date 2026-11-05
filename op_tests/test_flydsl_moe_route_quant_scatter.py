# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoE stage1 input prep: bf16 -> MXFP4/MXFP8 quant + per-route scatter + e8m0 preshuffle.

``flydsl_moe_fused_quant_preshuffle`` on the route-indexed branch (``topids_to_rows``
+ ``source_topk``) is what the gfx1250 grouped MoE runs before gemm1: every token's
bf16 row is quantized and copied to each of its ``topk`` grouped rows, and its
e8m0 block scales land in the WMMA-interleaved layout gemm1 reads.

Candidates are the two kernels that branch can launch:

* ``routeks``   -- one warp per route (``moe_fused_quant_preshuffle_routeks_*``),
                  forced with ``AITER_FLYDSL_TOKEN_MULTIDEST_QUANT=0``. This is the
                  original kernel and the bitwise reference.
* ``multidest`` -- one warp per token, quantized once and stored to every route
                  (``moe_token_multidest_quant_*``), the default dispatch.

Routing is the model's: random top-k of E experts per token, rows assigned by the
same route + contiguous-psum kernels the MoE runs. ``valid`` < 1 leaves a dead
tail past ``num_valid_routes`` holding garbage rows (a graph bucket's padding /
an EP dispatch tail); ``dropped`` > 0 marks routes with the negative
dropped-route row (EP non-local experts). Both outputs start from the same
non-zero fill, so the payload and scale buffers -- including every byte neither
kernel may touch -- must match the reference bit for bit.
"""

import argparse
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the route-indexed pk8 kernels are gfx1250-only
_ENV = "AITER_FLYDSL_TOKEN_MULTIDEST_QUANT"
_FILL = 0xA5


def _align_up(x, a):
    return (x + a - 1) // a * a


def make_routing(token_num, experts, topk, align, valid, dropped, seed=0):
    """``topids_to_rows`` exactly as the grouped MoE builds it, plus the tail."""
    from aiter.ops.flydsl.grouped_moe_gfx1250 import contiguous_psum_remap
    from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_topids_to_rows

    g = torch.Generator(device="cuda").manual_seed(seed)
    topk_ids = (
        torch.rand(token_num, experts, generator=g)
        .argsort(dim=1)[:, :topk]
        .to(torch.int32)
        .contiguous()
    )
    max_m = max(align, _align_up(token_num * topk, align))
    contiguous_m = max(align, _align_up(token_num * topk + experts * align - topk, align))
    numel = token_num * topk
    n_valid = max(1, int(round(token_num * valid))) * topk
    nvr = torch.full((1,), n_valid, dtype=torch.int32)
    masked_m, rows = flydsl_moe_topids_to_rows(topk_ids, experts, max_m)
    contiguous_psum_remap(masked_m, rows, experts, max_m, align, num_valid_routes=nvr)
    rows = rows.reshape(-1)
    if dropped > 0:
        drop = torch.rand(n_valid, generator=g) < dropped
        rows[:n_valid][drop] = DROPPED_ROUTE_ROW
    if n_valid < numel:
        # Unwritten in the model; garbage that would clobber live rows if used.
        rows[n_valid:] = torch.randint(
            0, contiguous_m, (numel - n_valid,), generator=g, dtype=torch.int32
        )
    return rows.contiguous(), nvr, contiguous_m, n_valid


def make_hidden(token_num, model_dim, seed=0):
    """bf16 activations whose 32-wide blocks span many e8m0 exponents."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    x = torch.randn(token_num, model_dim // 32, 32, generator=g)
    x = x * torch.exp2(
        torch.randint(-12, 12, (token_num, model_dim // 32, 1), generator=g).float()
    )
    return x.reshape(token_num, model_dim).to(dtypes.bf16)


def run_quant(hidden, rows, nvr, contiguous_m, topk, wmma_rep, quant_mode, out):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

    token_num, model_dim = hidden.shape
    return flydsl_moe_fused_quant_preshuffle(
        hidden.reshape(1, token_num, model_dim),
        1,
        contiguous_m,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        masked_m=None,
        topids_to_rows=rows,
        source_topk=topk,
        num_valid_routes=nvr,
        out_payload=out[0],
        out_scale=out[1],
    )


class _Env:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get(_ENV)
        os.environ[_ENV] = self.value

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = self.old


@benchmark()
def test_route_quant_scatter(
    token_num, model_dim, experts, topk, wmma_rep, quant_mode, valid, dropped
):
    align = wmma_rep * 16
    rows, nvr, contiguous_m, n_valid = make_routing(
        token_num, experts, topk, align, valid, dropped
    )
    hidden = make_hidden(token_num, model_dim)
    pb = model_dim // 2 if quant_mode == "fp4" else model_dim
    ws = model_dim // 32

    def _fresh():
        return (
            torch.full((1, contiguous_m, pb), _FILL, dtype=torch.uint8),
            torch.full(
                (1, contiguous_m // wmma_rep, ws * wmma_rep), _FILL, dtype=torch.uint8
            ),
        )

    args = (hidden, rows, nvr, contiguous_m, topk, wmma_rep, quant_mode)
    with _Env("0"):
        ref = run_quant(*args, _fresh())
    torch.cuda.synchronize()

    candidates = {"routeks": "0", "multidest": "1"}
    live = rows[:n_valid] >= 0
    tokens_read = n_valid // topk
    nbytes = (
        tokens_read * model_dim * hidden.element_size()  # bf16 rows, read once
        + n_valid * 4  # route -> row map
        + int(live.sum()) * (pb + ws)  # payload + e8m0 per live route
    )
    ret = {"gfx": get_gfx(), "MB": nbytes / 1e6}
    for name, env in candidates.items():
        out = _fresh()
        with _Env(env):
            run_quant(*args, out)
            torch.cuda.synchronize()
            # Bit-exact over the whole buffers: written rows, and every byte
            # neither kernel may write (padding, dropped and dead-tail rows).
            err = checkAllclose(
                ref[0].to(dtypes.fp32),
                out[0].to(dtypes.fp32),
                rtol=0,
                atol=0,
                msg=f"{name}: payload",
            )
            err_s = checkAllclose(
                ref[1].to(dtypes.fp32),
                out[1].to(dtypes.fp32),
                rtol=0,
                atol=0,
                msg=f"{name}: scale",
            )
            bitwise = torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1])
            if valid >= 1.0:
                # No dead tail: the null-bound launch (callers that pass no
                # num_valid_routes) must write exactly the same bytes.
                out_null = _fresh()
                run_quant(hidden, rows, None, *args[3:], out_null)
                torch.cuda.synchronize()
                bitwise = bitwise and (
                    torch.equal(ref[0], out_null[0]) and torch.equal(ref[1], out_null[1])
                )
            _, us = run_perftest(run_quant, *args, out, num_iters=50)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(err, err_s)
        ret[f"{name} bitwise"] = bitwise
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("route quant scatter unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MoE route-indexed quant + scatter + scale preshuffle",
    )
    parser.add_argument(
        "-t",
        "--token_num",
        type=int,
        nargs="*",
        default=[64, 128, 256, 384, 512, 1024, 2048, 4096, 8192, 16384, 20480],
        help="tokens (decode: 64-512, prefill: 2k-20k)",
    )
    parser.add_argument(
        "-d",
        "--model_dim",
        type=int,
        nargs="*",
        default=[2048],
        help="hidden size (Qwen3.5-35B-A3B: 2048)",
    )
    parser.add_argument("-e", "--experts", type=int, nargs="*", default=[256])
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[8])
    parser.add_argument(
        "-r",
        "--wmma_rep",
        type=int,
        nargs="*",
        default=[4],
        help="gemm1 WMMA M repeat (scale tile = 16*wmma_rep rows)",
    )
    parser.add_argument("-q", "--quant_mode", nargs="*", default=["fp4"])
    parser.add_argument(
        "--valid",
        type=float,
        nargs="*",
        default=[1.0],
        help="fraction of tokens below num_valid_routes (rest: garbage dead tail)",
    )
    parser.add_argument(
        "--dropped",
        type=float,
        nargs="*",
        default=[0.0],
        help="fraction of live routes carrying the dropped-route row",
    )
    args = parser.parse_args()

    df = []
    for quant_mode, (d, e, k, r, v, dr, t) in itertools.product(
        args.quant_mode,
        itertools.product(
            args.model_dim,
            args.experts,
            args.topk,
            args.wmma_rep,
            args.valid,
            args.dropped,
            args.token_num,
        ),
    ):
        df.append(test_route_quant_scatter(t, d, e, k, r, quant_mode, v, dr))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "route quant scatter summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
