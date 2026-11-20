# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MoE route sort + MXFP8 activation quant (`moe_sort_mx_quant_fwd`).

The kernel replaces ``moe_sorting`` followed by
``fused_dynamic_mxfp8_quant_moe_sort`` in one launch. Its outputs are checked
byte for byte against that pair: the sorted routing map (expert blocks, valid
count, padding and tail fill), the fp8 activations, the e8m0 scale of every
sorted row, and the zeroed moe_buf. Within an expert the routes come out in
routed-index order, which is also the reference's, so the maps compare exactly.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    _fused_decode_sort_quant,
    _is_fused_decode_sort_quant_enabled,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.moe_sort_mx_quant import moe_sort_mx_quant
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")


def _running_on_gfx950():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    try:
        return get_gfx_runtime() == "gfx950"
    except (KeyError, OSError, RuntimeError):
        return False


def _make_inputs(tokens, model_dim, experts, topk, skew=0.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + tokens)
    pop = torch.ones(experts)
    if skew:
        pop = (torch.randn(experts, generator=g) * skew).exp()
    topk_ids = torch.multinomial(
        pop.expand(tokens, experts), topk, replacement=False, generator=g
    ).to(dtypes.i32)
    topk_weights = torch.rand((tokens, topk), generator=g, dtype=dtypes.fp32)
    row_gain = torch.rand((tokens, 1), generator=g) * 4
    hidden = (torch.randn((tokens, model_dim), generator=g) * row_gain).to(dtypes.bf16)
    hidden[0, :32] = 0  # an all-zero group takes the amax floor
    return hidden, topk_ids, topk_weights


def _unswizzle(scale, rows, model_dim):
    """Row-major (rows, model_dim / 32) view of an `mx_scale_shuffle_idx` buffer."""
    sn = model_dim // 32
    sn_pad = (sn + 7) // 8 * 8
    x = torch.arange(rows).view(-1, 1)
    y = torch.arange(sn).view(1, -1)
    idx = (
        (x // 32 * sn_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )
    return scale.view(torch.uint8).view(-1)[idx]


def _reference(hidden, topk_ids, topk_weights, experts, block_m):
    tokens, model_dim = hidden.shape
    ids, weights, expert_ids, num_valid, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, dtypes.bf16, block_m
    )
    out, scale = fused_dynamic_mxfp8_quant_moe_sort(
        hidden,
        sorted_ids=ids,
        num_valid_ids=num_valid,
        token_num=tokens,
        topk=topk_ids.shape[1],
        block_size=block_m,
    )
    return ids, weights, expert_ids, num_valid, out, scale


def _check_outputs(fused, ref, hidden, topk_ids, topk_weights, block_m, tag):
    f_ids, f_w, f_e, f_nv, f_out, f_scale = fused
    r_ids, r_w, r_e, r_nv, r_out, r_scale = ref
    tokens, model_dim = hidden.shape
    topk = topk_ids.shape[1]
    nv = int(r_nv[0])
    assert f_nv.tolist() == [nv, tokens], f"{tag}: num_valid {f_nv.tolist()}"
    assert torch.equal(f_e[: nv // block_m], r_e[: nv // block_m]), f"{tag}: experts"
    assert torch.equal(f_ids[:nv], r_ids[:nv]), f"{tag}: sorted ids"
    assert torch.equal(f_w[:nv], r_w[:nv]), f"{tag}: sorted weights"
    sentinel = (topk << 24) | tokens
    assert (f_ids[nv:] == sentinel).all() and (f_w[nv:] == 0).all(), f"{tag}: tail"
    assert (f_e[nv // block_m :] == 0).all(), f"{tag}: tail expert ids"
    assert torch.equal(
        f_out.view(torch.uint8), r_out.view(torch.uint8)
    ), f"{tag}: fp8 activations"
    real = f_ids[:nv] != sentinel
    assert torch.equal(
        _unswizzle(f_scale, nv, model_dim)[real],
        _unswizzle(r_scale, nv, model_dim)[real],
    ), f"{tag}: sorted scales"


@benchmark()
def test_moe_sort_mx_quant(dtype, token_num, model_dim, E, topk, block_size, skew):
    hidden, topk_ids, topk_weights = _make_inputs(token_num, model_dim, E, topk, skew)
    ref, ref_us = run_perftest(
        _reference, hidden, topk_ids, topk_weights, E, block_size
    )
    moe_buf = torch.ones((token_num, model_dim), dtype=dtype)
    fused, fused_us = run_perftest(
        moe_sort_mx_quant, hidden, topk_ids, topk_weights, E, block_size, moe_buf
    )
    tag = f"M={token_num} dim={model_dim} E={E} topk={topk} bm={block_size}"
    _check_outputs(fused, ref, hidden, topk_ids, topk_weights, block_size, tag)
    assert (moe_buf == 0).all(), f"{tag}: moe_buf not zeroed"
    return {"ref_us": ref_us, "fused_us": fused_us, "uplift": ref_us / fused_us}


def _check_pad_routes(model_dim=5120, experts=384, topk=6, block_m=32):
    """Routes to -1 (padded rows) are not routes: the map is the one over the
    real rows alone, while every row is still quantized."""
    tokens, valid = 40, 33
    hidden, topk_ids, topk_weights = _make_inputs(tokens, model_dim, experts, topk)
    pad_ids = topk_ids.clone()
    pad_ids[valid:] = -1
    f_ids, f_w, f_e, f_nv, f_out, f_scale = moe_sort_mx_quant(
        hidden, pad_ids, topk_weights, experts, block_m
    )
    r_ids, r_w, r_e, r_nv, r_out, r_scale = moe_sort_mx_quant(
        hidden[:valid].contiguous(),
        topk_ids[:valid].contiguous(),
        topk_weights[:valid].contiguous(),
        experts,
        block_m,
    )
    nv = int(r_nv[0])
    assert f_nv.tolist() == [nv, tokens]
    assert torch.equal(f_e[: nv // block_m], r_e[: nv // block_m])
    real = r_ids[:nv] != ((topk << 24) | valid)
    assert torch.equal(f_ids[:nv][real], r_ids[:nv][real])
    assert (f_ids[:nv][~real] == ((topk << 24) | tokens)).all()
    assert torch.equal(f_w[:nv], r_w[:nv])
    assert torch.equal(f_out[:valid].view(torch.uint8), r_out.view(torch.uint8))
    assert torch.equal(
        _unswizzle(f_scale, nv, model_dim)[real],
        _unswizzle(r_scale, nv, model_dim)[real],
    )
    aiter.logger.info("pad routes OK")


def _check_cuda_graph_replay(model_dim=5120, experts=384, topk=6, block_m=32):
    hidden, topk_ids, topk_weights = _make_inputs(64, model_dim, experts, topk)
    ref = _reference(hidden, topk_ids, topk_weights, experts, block_m)
    moe_buf = torch.empty((64, model_dim), dtype=dtypes.bf16)
    moe_sort_mx_quant(hidden, topk_ids, topk_weights, experts, block_m, moe_buf)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused = moe_sort_mx_quant(
            hidden, topk_ids, topk_weights, experts, block_m, moe_buf
        )
    for _ in range(2):
        moe_buf.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        _check_outputs(
            fused, ref, hidden, topk_ids, topk_weights, block_m, "graph replay"
        )
        assert (moe_buf == 0).all()
    aiter.logger.info("CUDA graph replay OK")


_GATE_KWARGS = {
    "quant_type": aiter.QuantType.per_1x32,
    "q_dtype_a": dtypes.fp8,
    "q_dtype_w": dtypes.fp4x2,
    "activation": aiter.ActivationType.Silu,
    "is_g1u1": True,
    "expert_mask": None,
    "need_local_topk_ids": False,
    "run_1stage": False,
    "stage1_is_flydsl": True,
    "hidden_pad": 0,
    "ksplit": 0,
}


def _check_gate():
    """The gate admits only what the kernel is instantiated for; anything else
    would abort in its host checks."""
    table = [
        # tokens, model_dim, experts, topk, block_m, extra kwargs, expected
        (256, 5120, 384, 6, 32, {}, True),
        (1, 4096, 256, 8, 16, {}, True),
        (1024, 7168, 1024, 8, 64, {}, True),
        (256, 5120, 384, 6, 32, {"prequant": False}, False),
        (256, 5120, 384, 6, 24, {}, False),  # block_m not a power of two
        (256, 6144, 384, 6, 32, {}, False),  # uninstantiated model_dim
        (256, 4096, 1025, 6, 32, {}, False),  # more experts than the scan holds
        (2049, 5120, 384, 4, 32, {}, False),  # above the M cap
        (1024, 5120, 384, 20, 32, {}, False),  # > 64 routes per block
        (256, 5120, 384, 6, 32, {"num_local_tokens": torch.ones(1)}, False),
    ]
    for tokens, dim, experts, topk, bm, extra, expected in table:
        hidden, topk_ids, topk_weights = _make_inputs(
            tokens, dim, min(experts, 1024), topk
        )
        got = _is_fused_decode_sort_quant_enabled(
            hidden,
            topk_ids,
            topk_weights,
            global_experts=experts,
            block_m=bm,
            **{**_GATE_KWARGS, **extra},
        )
        assert got is expected, (tokens, dim, experts, topk, bm, extra)
    aiter.logger.info("gate OK")


def _check_fused_moe_wrapper(model_dim=5120, experts=384, topk=6, block_m=32):
    """`_fused_decode_sort_quant` zeroes a caller output in place as moe_buf,
    allocates one otherwise, and skips it when stage 2 does not accumulate."""
    tokens = 48
    hidden, topk_ids, topk_weights = _make_inputs(tokens, model_dim, experts, topk)
    ref = _reference(hidden, topk_ids, topk_weights, experts, block_m)
    output = torch.ones((tokens, model_dim), dtype=dtypes.bf16)
    for accumulate, out in ((True, output), (True, None), (False, None)):
        res = _fused_decode_sort_quant(
            hidden,
            topk_ids,
            topk_weights,
            model_dim=model_dim,
            global_experts=experts,
            block_m=block_m,
            moebuf_dtype=dtypes.bf16,
            accumulate=accumulate,
            q_dtype_a=dtypes.fp8,
            output=out,
        )
        tag = f"wrapper accumulate={accumulate} output={out is not None}"
        _check_outputs(
            res[:4] + res[5:], ref, hidden, topk_ids, topk_weights, block_m, tag
        )
        moe_buf = res[4]
        if not accumulate:
            assert moe_buf.numel() == 0, tag
            continue
        assert moe_buf.shape == (tokens, model_dim), tag
        assert (moe_buf == 0).all(), tag
        if out is not None:
            assert moe_buf.data_ptr() == out.data_ptr(), tag
    aiter.logger.info("fused_moe wrapper OK")


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="moe_sort_mx_quant: fused route sort + MXFP8 quant",
)
parser.add_argument(
    "-dim", type=int, nargs="*", default=[4096, 5120, 7168], help="Model dimension."
)
parser.add_argument(
    "-ek",
    "--expert_topk",
    type=dtypes.str2tuple,
    nargs="*",
    default=[[384, 6], [256, 8], [1024, 8]],
    help="""Number of experts and top-k.
    e.g.: -ek 384,6""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 2, 7, 16, 64, 128, 256, 333, 512, 1024, 2048],
    help="Tokens.",
)
parser.add_argument(
    "-bm",
    "--block_m",
    type=int,
    nargs="*",
    default=[16, 32, 64],
    help="Sort block size.",
)
parser.add_argument(
    "--skew",
    type=float,
    nargs="*",
    default=[0.0, 2.0],
    help="Expert popularity ~ lognormal(sigma); 0 routes uniformly.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if not _running_on_gfx950():
        aiter.logger.info("moe_sort_mx_quant runs on gfx950 only; skipped")
    else:
        _check_gate()
        _check_pad_routes()
        _check_cuda_graph_replay()
        _check_fused_moe_wrapper()
        df = []
        for dim, (E, topk), m, bm, skew in itertools.product(
            args.dim, args.expert_topk, args.m, args.block_m, args.skew
        ):
            if E > dim // 4 or -(-m // get_cu_num()) * topk > 64:
                continue
            df.append(test_moe_sort_mx_quant(dtypes.bf16, m, dim, E, topk, bm, skew))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "moe_sort_mx_quant summary (markdown):\n%s", df.to_markdown(index=False)
        )
