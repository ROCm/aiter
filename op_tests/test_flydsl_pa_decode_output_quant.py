# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 sparse PA with FP8 per-token output, against PA + the standalone
per-token quant (bit-exact). MiniMax-M3 TP4 shapes: one KV head, GQA 16,
head_dim 128, page 16.

    python3 op_tests/test_flydsl_pa_decode_output_quant.py   # checks + timing table
    python3 -m pytest -q op_tests/test_flydsl_pa_decode_output_quant.py
"""

import argparse
import importlib
import itertools
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes, pertoken_quant
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.pa_decode_plan import _lds_max_batch
from aiter.ops.quant import dynamic_per_token_scaled_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

# The module, not the function of the same name that aiter.ops.flydsl re-exports.
pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
CTX = 2048
# (output path pa_decode must pick, rows, max_partitions, mixed context lengths)
CASES = [
    ("cluster8", 1, 8, False),
    ("cluster8", 16, 8, False),  # bs=4 x 4 EAGLE3 verify tokens
    ("cluster8", 64, 8, False),
    ("cluster8", 16, 8, True),  # empty, single-task and split sequences
    ("direct", 4536, 1, False),  # traced sparse-prefill token count
    ("direct", 64, 1, True),  # with empty sequences
    ("split", 256, 8, False),  # too wide for clusters: PA, then the standalone quant
]


def make_case(mode, rows, max_partitions, mixed, pad=0):
    torch.manual_seed(0)
    dev, pages = "cuda", CTX // 16
    q = torch.randn(rows, 16, 128, dtype=torch.bfloat16, device=dev)
    k, k_scale = pertoken_quant(
        torch.randn(pages, 1, 16, 128, device=dev), quant_dtype=dtypes.fp8
    )
    v, v_scale = pertoken_quant(
        torch.randn(pages, 1, 16, 128, device=dev), quant_dtype=dtypes.fp8
    )
    if mode == "direct":
        ctx = torch.arange(rows, dtype=torch.int32, device=dev) % CTX + 1
    else:
        ctx = torch.full((rows,), CTX, dtype=torch.int32, device=dev)
    if mixed:
        ctx[::5] = 0
        ctx[1::5] = 1
        ctx[2::5] = 257
    plan = pa.plan_pa_decode(
        ctx,
        1,
        max_partitions=max_partitions,
        # sparse prefill raises the default cap to the planner's LDS limit
        max_batch_size=_lds_max_batch(
            torch.cuda.get_device_properties(dev).shared_memory_per_block
        ),
        query_length=1,
    )
    scratch = (1, plan.capacity, 16)
    return SimpleNamespace(
        rows=rows,
        args=(
            q,
            k.reshape(pages, 1, 16, 8, 16).permute(0, 1, 3, 2, 4).contiguous(),
            v.reshape(pages, 1, 1, 16, 128).permute(0, 1, 2, 4, 3).contiguous(),
            ctx,
            torch.arange(pages, dtype=torch.int32, device=dev).repeat(rows, 1),
            128**-0.5,
            1,
        ),
        kwargs={
            "compute_type": dtypes.fp8,
            "key_scale": k_scale,
            "value_scale": v_scale,
            "work_plan": plan,
            "max_context_length": CTX,
            "exp_sums": torch.empty(scratch, device=dev),
            "max_logits": torch.empty(scratch, device=dev),
            "temporary_output": torch.empty(*scratch, 128, dtype=q.dtype, device=dev),
        },
        out=torch.empty_like(q),
        fp8=torch.empty(rows + pad, 16, 128, dtype=dtypes.fp8, device=dev),
        scale=torch.empty(rows + pad, 1, device=dev),
    )


def run_separate(c, out, fp8, scale):
    pa.pa_decode(out, *c.args, **c.kwargs)
    dynamic_per_token_scaled_quant(fp8.view(c.rows, -1), out.view(c.rows, -1), scale)


def run_fused(c, store_output=True, pad=0):
    pa.pa_decode(
        c.out,
        *c.args,
        **c.kwargs,
        quantized_output=c.fp8,
        output_scale=c.scale,
        store_output=store_output,
        quantized_pad_rows=pad,
    )


def check_case(monkeypatch, mode, rows, max_partitions, mixed, store_output, pad):
    c = make_case(mode, rows, max_partitions, mixed, pad)
    ref_out, ref_fp8 = torch.empty_like(c.out), torch.empty_like(c.fp8[:rows])
    ref_scale = torch.empty_like(c.scale[:rows])
    compiled = []
    tile = pa.compile_pa_decode_tile
    monkeypatch.setattr(
        pa, "compile_pa_decode_tile", lambda **kw: compiled.append(kw) or tile(**kw)
    )

    def poisoned_call(fn):
        # Sentinels: anything the call leaves unwritten fails the comparison.
        c.out.fill_(float("nan"))
        c.fp8.view(torch.uint8).fill_(0x7F)
        c.scale.fill_(float("nan"))
        fn()
        run_separate(c, ref_out, ref_fp8, ref_scale)
        assert torch.equal(c.out, ref_out) if store_output else c.out.isnan().all()
        assert torch.equal(c.fp8[:rows].view(torch.uint8), ref_fp8.view(torch.uint8))
        assert torch.equal(c.scale[:rows], ref_scale)
        assert (c.fp8[rows:].view(torch.uint8) == 0).all() and (
            c.scale[rows:] == 1
        ).all()

    poisoned_call(lambda: run_fused(c, store_output, pad))
    # The fused epilogue, not the fallback, must have run where it applies.
    tile_kwargs = next(
        kw for kw in compiled if kw.get("quant_output") or mode == "split"
    )
    assert tile_kwargs.get("quant_output", False) is (mode != "split")
    assert tile_kwargs["cluster_merge"] == (8 if mode == "cluster8" else 0)

    # Graph replay must read new query values and keep no scale/max scratch.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_fused(c, store_output, pad)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_fused(c, store_output, pad)
    c.args[0].neg_()
    poisoned_call(graph.replay)


@pytest.mark.skipif(get_gfx_runtime() != "gfx1250", reason="gfx1250 only")
@pytest.mark.parametrize("pad", [0, 37])
@pytest.mark.parametrize("store_output", [True, False])
@pytest.mark.parametrize("case", CASES)
def test_sparse_pa_output_quant(monkeypatch, case, store_output, pad):
    check_case(monkeypatch, *case, store_output, pad)


@benchmark()
def bench(rows, store_output):
    """Per call: PA then the standalone quant vs the fused epilogue."""
    c = make_case(
        "cluster8" if rows <= 64 else "direct", rows, 8 if rows <= 64 else 1, False
    )
    sep = (torch.empty_like(c.out), torch.empty_like(c.fp8), torch.empty_like(c.scale))
    tokens = int(c.args[3].sum())
    # FP8 K and V per context token (128 B + 4 B scale each), BF16 query in,
    # FP8 + scale out, and the BF16 output where it is stored.
    nbytes = tokens * 2 * 132 + rows * (2048 * 3 + 4)
    candidates = {
        "pa+quant": (lambda: run_separate(c, *sep), sep[1], sep[2], True),
        "fused": (lambda: run_fused(c, store_output), c.fp8, c.scale, store_output),
    }
    ret = {"gfx": get_gfx_runtime()}
    for name, (fn, fp8, scale, bf16_out) in candidates.items():
        _, us = run_perftest(fn)
        err = checkAllclose(
            sep[1].float() * sep[2][:, :, None],
            fp8.float() * scale[:, :, None],
            rtol=0,
            atol=0,
            msg=name,
        )
        assert err == 0
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 4 * 16 * 128 * tokens / us / 1e6  # QK^T + PV
        ret[f"{name} TB/s"] = (nbytes + bf16_out * rows * 4096) / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("sparse PA FP8 output is gfx1250 only; skipping")
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--rows", type=int, nargs="*", default=[4, 16, 64, 2517, 4536]
    )
    args = parser.parse_args()
    for case, store_output, pad in itertools.product(CASES, [True, False], [0, 37]):
        with pytest.MonkeyPatch.context() as monkeypatch:
            check_case(monkeypatch, *case, store_output, pad)
    df = pd.DataFrame(
        [bench(r, s) for r, s in itertools.product(args.rows, [True, False])]
    )
    aiter.logger.info(
        "sparse PA FP8 output (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
