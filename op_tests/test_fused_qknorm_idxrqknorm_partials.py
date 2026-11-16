# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_qknorm_idxrqknorm(qkv_partials=): the split-K reduce done by the consumer.

A split-K GEMM can leave its fp32 partials to the kernel that reads its output instead
of reducing them in a kernel of its own. Every check is bitwise against the reduce
kernel of the gfx1250 split-K GEMM followed by fused_qknorm_idxrqknorm on its output:

* consumer: random partials of a sparse attention layer (with index heads) and of
  dense ones, bf16 and fp8 caches, 2 / 4 / 8 splits, q to q_out or in place.
* edge_values: partials whose sums are subnormal, overflow, NaN or bf16 ties.
* rejects: the calls with no partials instance fail on the host.

Runs as `python3 <file>` (summary tables) or under pytest (same cases and checks).
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the split-K GEMM that leaves fp32 partials
HEAD_DIM = 128
M_LIST = [1, 4, 8, 16, 32, 48, 64]  # decode token counts, where GEMMs split K
SPLITS = [2, 4, 8]
# (q heads, kv heads, index heads, rotary_dim): a sparse layer with an index branch
# (MiniMax-M3 TP4) and dense layers without one.
LAYERS = [(16, 1, 1, 64), (16, 1, 0, 64), (8, 2, 0, 128)]
KV_DTYPES = ["fp8", "auto"]

pytestmark = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_GFX, reason="gfx1250 only"
)


def bitwise_equal(a, b):
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def reduce_ref(partials, out):
    """out = the bf16 sum of contiguous fp32 [splits, rows, row] partials, from the
    reduce kernel run_preshuffle_gemm_a8_gfx1250 launches after a split-K GEMM."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    splits, rows, row = partials.shape
    assert partials.is_contiguous() and out.shape == (rows, row)
    _run_compiled(
        compile_gemm_a8w8_splitk_reduce(
            split_k=splits, out_dtype_str="bf16", partial_dtype_str="f32"
        ),
        ptr_arg(partials),
        ptr_arg(out),
        rows * row,
        1,
        row,
        rows * row * partials.element_size(),
        fx.Stream(torch.cuda.current_stream()),
    )
    return out


def make_layer(m, nq, nkv, niq, rotary_dim, block_size=16, max_pos=4096):
    """Norm weights, positions and page-16 SHUFFLE cache slots of one layer."""
    torch.manual_seed(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    nb = (m + block_size - 1) // block_size + 2
    return {
        "heads": (nq, nkv, niq),
        "row": (nq + 2 * nkv + (niq + 1 if niq else 0)) * HEAD_DIM,
        "norm_w": [torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1 for _ in range(4)],
        "cos_sin": torch.cat([freqs.cos(), freqs.sin()], -1).to(dtypes.bf16),
        "positions": torch.randint(0, max_pos, (m,), dtype=torch.int64),
        "slots": torch.randperm(nb * block_size)[:m].to(torch.int64),
        "nb": nb,
        "bs": block_size,
        "rotary_dim": rotary_dim,
    }


def make_outputs(m, layer, kv_cache_dtype, in_place=False):
    """Zeroed q / index q / page-16 SHUFFLE cache outputs of one layer."""
    nq, nkv, niq = layer["heads"]
    fp8 = kv_cache_dtype == "fp8"
    cdt = dtypes.fp8 if fp8 else dtypes.bf16
    x = 16 // torch.empty(0, dtype=cdt).element_size()
    nb, bs = layer["nb"], layer["bs"]
    return {
        "q_out": None if in_place else torch.zeros(m, nq * HEAD_DIM, dtype=dtypes.bf16),
        "index_q_out": torch.zeros(m, niq * HEAD_DIM, dtype=dtypes.bf16) if niq else None,
        "kv_cache_k": torch.zeros(nb, nkv, HEAD_DIM // x, bs, x, dtype=cdt),
        "kv_cache_v": torch.zeros(nb, nkv, bs // x, HEAD_DIM, x, dtype=cdt),
        "index_cache": torch.zeros(nb * bs, HEAD_DIM, dtype=cdt) if niq else None,
        "k_scale": torch.zeros(nb, nkv, bs) if fp8 else None,
        "v_scale": torch.zeros(nb, nkv, bs) if fp8 else None,
    }  # fmt: skip


def qknorm_args(qkv, layer, kv_cache_dtype, out, qkv_partials=None, **overrides):
    """(args, kwargs) of fused_qknorm_idxrqknorm in the insert mode."""
    nq, nkv, niq = layer["heads"]
    qw, kw, iqw, ikw = layer["norm_w"]
    args = (
        qkv,
        qw,
        kw,
        layer["cos_sin"],
        layer["positions"],
        nq,
        nkv,
        layer["rotary_dim"],
        1e-6,
    )
    kwargs = {
        "index_q_norm_weight": iqw if niq else None,
        "index_k_norm_weight": ikw if niq else None,
        "num_index_heads": niq,
        "slot_mapping": layer["slots"],
        "kv_cache_k": out["kv_cache_k"],
        "kv_cache_v": out["kv_cache_v"],
        "index_cache": out["index_cache"],
        "block_size": layer["bs"],
        "q_out": out["q_out"],
        "index_q_out": out["index_q_out"],
        "index_slot_mapping": layer["slots"] if niq else None,
        "kv_cache_dtype": kv_cache_dtype,
        "index_cache_dtype": kv_cache_dtype if niq else "auto",
        "k_scale": out["k_scale"],
        "v_scale": out["v_scale"],
        "asm_layout": True,
        "qkv_partials": qkv_partials,
    }
    kwargs.update(overrides)
    return args, kwargs


def run_qknorm(qkv, layer, kv_cache_dtype, out, qkv_partials=None, **overrides):
    """fused_qknorm_idxrqknorm in the insert mode; returns the outputs with qkv."""
    args, kwargs = qknorm_args(
        qkv, layer, kv_cache_dtype, out, qkv_partials, **overrides
    )
    aiter.fused_qknorm_idxrqknorm(*args, **kwargs)
    return {**out, "qkv": qkv}


def check(m, layer, kv_cache_dtype, partials, in_place=False):
    """qknorm(qkv_partials=) on a NaN qkv vs reduce + qknorm, every output bit for
    bit (qkv holds the reduced row after both). partials may hold more rows than m."""
    qkv_ref = reduce_ref(partials, torch.empty(partials.shape[1:], dtype=dtypes.bf16))
    qkv_ref = qkv_ref[:m].clone()
    base = run_qknorm(
        qkv_ref, layer, kv_cache_dtype, make_outputs(m, layer, kv_cache_dtype, in_place)
    )
    qkv = torch.full_like(qkv_ref, float("nan"))
    test = run_qknorm(qkv, layer, kv_cache_dtype, make_outputs(m, layer, kv_cache_dtype, in_place),
                      partials[:, :m])  # fmt: skip
    return [
        n for n, t in base.items() if t is not None and not bitwise_equal(test[n], t)
    ]


@benchmark()
def consumer(m, layer_cfg, kv_cache_dtype, splits, in_place):
    layer = make_layer(m, *layer_cfg)
    torch.manual_seed(splits)
    # Three rows more than m: a split stride longer than the token count.
    partials = torch.randn(splits, m + 3, layer["row"]) * 4
    diff = check(m, layer, kv_cache_dtype, partials, in_place)
    ret = {"gfx": get_gfx_runtime(), "bitwise": not diff}
    assert (
        not diff
    ), f"{m=} {layer_cfg=} {kv_cache_dtype=} {splits=} {in_place=}: {diff}"
    out = make_outputs(m, layer, kv_cache_dtype, in_place)
    qkv = torch.empty(m + 3, layer["row"], dtype=dtypes.bf16)

    def two_kernels():
        reduce_ref(partials, qkv)
        return run_qknorm(qkv[:m], layer, kv_cache_dtype, out)

    _, ret["reduce+qknorm us"] = run_perftest(two_kernels)
    _, ret["qknorm(partials) us"] = run_perftest(
        run_qknorm, qkv[:m], layer, kv_cache_dtype, out, partials[:, :m]
    )
    return ret


def edge_partials(splits, rows, row):
    """Random fp32 partials plus columns whose sums are subnormal, overflow, NaN or a
    bf16 tie: where the consumer's denormal-flushing arithmetic could differ."""
    torch.manual_seed(splits)
    p = torch.randn(splits, rows, row) * 4
    tiny = torch.finfo(torch.float32).tiny
    cases = [
        (1.5 * tiny, -tiny),  # subnormal sum of normal partials
        (3e-39, 1e-39),  # subnormal partials
        (3.3e38, 3.3e38),  # overflow to inf
        (float("inf"), float("-inf")),  # NaN
        (1.0, 2.0**-8),  # tie, rounds down to the even 1.0
        (1.0 + 2.0**-7, 2.0**-8),  # tie, rounds up to the even 1 + 2^-6
    ]
    for c, (a, b) in enumerate(cases):
        cols = slice(c, None, len(cases) + 2)
        p[:, :, cols] = 0.0
        p[0, :, cols], p[1, :, cols] = a, b
    return p


@benchmark()
def edge_values(m, splits):
    layer = make_layer(m, *LAYERS[0])
    diff = check(m, layer, "auto", edge_partials(splits, m, layer["row"]))
    ret = {"gfx": get_gfx_runtime(), "bitwise": not diff}
    assert not diff, f"{m=} {splits=}: outputs differ: {diff}"
    return ret


def run_suite(case, params, title=None):
    rows = [case(*p) for p in params]
    if title:
        table = pd.DataFrame(rows).to_markdown(index=False)
        aiter.logger.info("%s summary (markdown):\n%s", title, table)
    return rows


@pytest.mark.parametrize("layer_cfg", LAYERS)
@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_consumer(layer_cfg, kv_cache_dtype):
    params = itertools.product(M_LIST, [layer_cfg], [kv_cache_dtype], SPLITS, [False])
    run_suite(consumer, list(params))


def test_consumer_q_in_place():
    run_suite(consumer, [(m, cfg, "fp8", 4, True) for m in (1, 64) for cfg in LAYERS])


@pytest.mark.parametrize("splits", SPLITS)
def test_edge_values(splits):
    run_suite(edge_values, [(m, splits) for m in M_LIST])


REJECT_CHILD = (
    "import sys, torch, aiter; a, kw = torch.load(sys.argv[1]); "
    "aiter.fused_qknorm_idxrqknorm(*a, **kw); torch.cuda.synchronize()"
)


def test_rejects():
    """No partials instance: not the insert mode, an index branch skipped (columns no
    slot covers), an fp8 q_out; and partials the kernel cannot read. A failed host
    check aborts the process, so each call runs in a child ("good" shows the harness
    is sound)."""
    m = 4
    layer = make_layer(m, *LAYERS[0])
    row = layer["row"]
    ok = torch.zeros(2, m, row)
    cases = {
        "good": (ok, {}),
        "not_insert": (
            ok,
            {"kv_cache_k": None, "kv_cache_v": None, "index_cache": None},
        ),
        "skip_index_branch": (ok, {"skip_index_branch": True, "index_cache": None}),
        "fp8_q_out": (ok, {"q_out": torch.zeros(m, 16 * HEAD_DIM, dtype=dtypes.fp8)}),
        "3_splits": (torch.zeros(3, m, row), {}),
        "fp16_partials": (ok.half(), {}),
        "short_rows": (torch.zeros(2, m - 1, row), {}),
        "unaligned": (torch.zeros(2 * m * row + 1)[1:].view(2, m, row), {}),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, (partials, overrides) in cases.items():
            qkv = torch.zeros(m, row, dtype=dtypes.bf16)
            out = make_outputs(m, layer, "auto")
            path = os.path.join(tmp, name)
            torch.save(
                qknorm_args(qkv, layer, "auto", out, partials, **overrides), path
            )
            cmd = [sys.executable, "-c", REJECT_CHILD, path]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if name == "good":
                assert r.returncode == 0, r.stderr[-500:]
            else:
                assert r.returncode != 0, f"{name}: accepted"
                assert "qkv_partials" in r.stderr, f"{name}: {r.stderr[-500:]}"


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("qkv_partials test unsupported on %s", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(description="fused_qknorm_idxrqknorm qkv_partials")
    parser.add_argument("-m", type=int, nargs="*", default=M_LIST)
    parser.add_argument("--kv_cache_dtype", nargs="*", default=KV_DTYPES)
    args = parser.parse_args()
    prod = itertools.product
    cases = prod(args.m, LAYERS, args.kv_cache_dtype, SPLITS, [False])
    run_suite(consumer, list(cases), "qkv_partials consumer")
    run_suite(edge_values, list(prod(args.m, SPLITS)), "qkv_partials edge values")
    test_consumer_q_in_place()
    test_rejects()


if __name__ == "__main__":
    main()
