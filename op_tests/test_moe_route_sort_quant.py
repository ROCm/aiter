# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The fused MoE routing kernel (DeepSeek-V4.1-Flash TP4 shape) against the unfused chain.

Chain: topk_gating (sqrtsoftplus, renorm) + shared expert 384 appended in slot 7 + moe_sorting +
fused_dynamic_mxfp8_quant_moe_sort. The kernel must give the same top-k ids and weights, MXFP8 input and e8m0
scales bit for bit, the same valid-row count and block expert ids, and per expert the same sorted rows (rows within
one expert come out in atomic order, not token order).
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import moe_route_sort_quant as mrsq
from aiter.ops.flydsl.kernels import moe_route_sort_quant as rsq
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]


def run_chain(x, logits, bias, block_m, shared_w=1.0):
    m = x.shape[0]
    tw = torch.empty((m, rsq.TOPK_R), dtype=dtypes.fp32)
    ti = torch.empty((m, rsq.TOPK_R), dtype=dtypes.i32)
    aiter.topk_gating(tw, ti, logits, bias, True, 1.0, score_func="sqrtsoftplus")
    ids = torch.cat([ti, torch.full((m, 1), rsq.E_ROUTED, dtype=dtypes.i32)], 1)
    w = torch.cat([tw, torch.full((m, 1), shared_w, dtype=dtypes.fp32)], 1)
    sid, sw, seid, nv, _ = moe_sorting(ids, w, rsq.E_ALL, rsq.DIM, dtypes.bf16, block_m)
    a1, a1s = fused_dynamic_mxfp8_quant_moe_sort(
        x,
        sorted_ids=sid,
        num_valid_ids=nv,
        token_num=m,
        topk=rsq.TOPK,
        block_size=block_m,
        sorted_weights=sw,
        num_experts_upper_bound=rsq.E_ALL,
    )
    return {
        "ids": ids,
        "w": w,
        "sid": sid,
        "seid": seid,
        "nv": nv,
        "a1": a1,
        "a1s": a1s,
    }


def _bytes(t):
    return t.contiguous().view(torch.uint8)


def check(x, ref, got):
    m = x.shape[0]
    assert torch.equal(got["topk_ids"], ref["ids"]), "topk ids differ"
    assert torch.equal(_bytes(got["topk_w"]), _bytes(ref["w"])), "topk weights differ"
    a1 = _bytes(got["a1"]).reshape(m, -1)
    assert torch.equal(a1, _bytes(ref["a1"]).reshape(m, -1)), "MXFP8 a1 differs"
    n_pad = int(ref["nv"][0])
    assert int(got["num_valid"][0]) == n_pad, "num_valid_ids differs"
    bm = got["block_m"]
    nb = n_pad // bm
    seid = ref["seid"][:nb]
    assert torch.equal(got["sorted_eids"][:nb], seid), "sorted expert ids differ"
    rows = seid.long().repeat_interleave(bm)
    gs, rs = got["sorted_ids"][:n_pad], ref["sid"][:n_pad]
    for e in torch.unique(rows).tolist():
        sel = rows == e
        assert torch.equal(
            torch.sort(gs[sel]).values, torch.sort(rs[sel]).values
        ), f"expert {e}: sorted rows differ"
    tok, slot = (gs & 0xFFFFFF).long(), (gs >> 24).long()
    real = tok < m
    want_w = torch.zeros(n_pad, dtype=dtypes.fp32)
    want_w[real] = got["topk_w"][tok[real], slot[real]]
    assert torch.equal(
        _bytes(got["sorted_w"][:n_pad]), _bytes(want_w)
    ), "sorted weights differ"
    # the reference scales for the kernel's own row order
    _, a1s = fused_dynamic_mxfp8_quant_moe_sort(
        x,
        sorted_ids=got["sorted_ids"],
        num_valid_ids=got["num_valid"],
        token_num=m,
        topk=rsq.TOPK,
        block_size=bm,
        sorted_weights=got["sorted_w"],
        num_experts_upper_bound=rsq.E_ALL,
    )
    want, have = _bytes(a1s).reshape(-1), _bytes(got["a1_scale"]).reshape(-1)
    n = min(want.numel(), have.numel())
    # pad rows' scale bytes are not written by the kernel (stage 1 drops pad rows)
    bad = ((want[:n] != have[:n]) & (want[:n] != 0)).sum().item()
    assert bad == 0, f"{bad} e8m0 scale bytes differ"


@benchmark()
def test_moe_route_sort_quant(m):
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn((m, rsq.DIM), dtype=dtypes.bf16, generator=gen)
    logits = torch.randn((m, rsq.E_ROUTED), dtype=dtypes.fp32, generator=gen) * 2
    bias = torch.randn(rsq.E_ROUTED, dtype=dtypes.fp32, generator=gen) * 0.1
    block_m = mrsq.block_m_for(m)
    router = mrsq.MoeRouteSortQuant()
    ref = run_chain(x, logits, bias, block_m)
    check(x, ref, router(logits, bias, x))
    # read logits and x, write a1 and the sorted outputs
    nbytes = m * (rsq.E_ROUTED * 4 + rsq.DIM * 2 + rsq.DIM + rsq.GROUPS * 2)
    ret = {"gfx": get_gfx_runtime(), "block_m": block_m}
    candidates = {
        "chain": lambda: run_chain(x, logits, bias, block_m),
        "fused": lambda: router(logits, bias, x),
    }
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "moe_route_sort_quant unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="fused MoE routing (top-k + sort + MXFP8 quant) vs the chain",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[257, 288, 384, 512, 768, 1024, 1536, 2048],
        help="tokens (257-2048)",
    )
    args = parser.parse_args()
    df = pd.DataFrame([test_moe_route_sort_quant(m) for m in args.m])
    aiter.logger.info(
        "moe_route_sort_quant summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
