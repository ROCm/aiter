# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K reduction left to the consumer of the gfx1250 PTPC bpreshuffle GEMM.

gemm_a8w8_bpreshuffle_splitk_deferred skips the standalone split-K reduce of a
decode-size GEMM and leaves the fp32 partials to the kernel that reads the output.
Every check is bitwise against the two-kernel path (GEMM + reduce):

* deferred_gemm: deferred GEMM + reduce_splitk_partials vs gemm_a8w8_bpreshuffle
  (MiniMax-M3 TP4 sparse QKV, dense QKV, dense gate/up).
* sparse_qkv_consumer: deferred GEMM + fused_qknorm_idxrqknorm(qkv_partials=) vs
  GEMM + reduce + fused_qknorm_idxrqknorm as the sparse attention layer calls it.
* edge_values: crafted partials whose sums are subnormal, overflow, NaN or ties;
  the qkv the consumer writes back vs the reduce kernel's output.

Runs as `python3 <file>` (summary tables) or under pytest (same cases and checks).
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the split-K PTPC GEMM is the gfx1250 WMMA backend
HEAD_DIM = 128
MAX_M = 64  # largest M the deferred path takes (its max_m default)
M_LIST = [1, 4, 8, 16, 32, 48, 64]  # decode tokens per TP4 rank
NK_LIST = [(2560, 6144), (2304, 6144), (6144, 6144)]  # sparse QKV, dense QKV, gate/up
SPARSE_N, SPARSE_K = NK_LIST[0]
KV_DTYPES = ["fp8", "auto"]

pytestmark = pytest.mark.skipif(
    get_gfx_runtime() not in SUPPORTED_GFX, reason="gfx1250 only"
)


def bitwise_equal(a, b):
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def split_of(partials):
    return 1 if partials is None else partials.shape[0]


def gemm_inputs(m, n, k):
    """fp8 per-token operands (weight preshuffled) and an fp32 torch reference."""
    torch.manual_seed(0)
    xq, xs = aiter.pertoken_quant(
        torch.randn(m, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    wq, ws = aiter.pertoken_quant(
        torch.randn(n, k, dtype=dtypes.bf16), quant_dtype=dtypes.fp8
    )
    ref = ((xq.float() * xs) @ (wq.float() * ws).t()).to(dtypes.bf16)
    return (xq, shuffle_weight(wq, layout=(16, 16)), xs, ws), ref


def sparse_layer(m, rotary_dim=64, block_size=16, max_pos=4096):
    """One MiniMax-M3 TP4 sparse layer: 16 q heads, 1 kv head, 1 index head,
    partial rotary, page-16 SHUFFLE caches."""
    torch.manual_seed(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    nb = (m + block_size - 1) // block_size + 2
    return {
        "norm_w": [torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1 for _ in range(4)],
        "cos_sin": torch.cat([freqs.cos(), freqs.sin()], -1).to(dtypes.bf16),
        "positions": torch.randint(0, max_pos, (m,), dtype=torch.int64),
        "slots": torch.randperm(nb * block_size)[:m].to(torch.int64),
        "nb": nb,
        "bs": block_size,
        "rotary_dim": rotary_dim,
    }


def qknorm_outputs(m, layer, kv_cache_dtype):
    """Zeroed q / index q / page-16 SHUFFLE cache outputs of one sparse layer."""
    fp8 = kv_cache_dtype == "fp8"
    cdt = dtypes.fp8 if fp8 else dtypes.bf16
    x = 16 // torch.empty(0, dtype=cdt).element_size()
    nb, bs = layer["nb"], layer["bs"]
    return {
        "q_out": torch.zeros(m, 16 * HEAD_DIM, dtype=dtypes.bf16),
        "index_q_out": torch.zeros(m, HEAD_DIM, dtype=dtypes.bf16),
        "kv_cache_k": torch.zeros(nb, 1, HEAD_DIM // x, bs, x, dtype=cdt),
        "kv_cache_v": torch.zeros(nb, 1, bs // x, HEAD_DIM, x, dtype=cdt),
        "index_cache": torch.zeros(nb * bs, HEAD_DIM, dtype=cdt),
        "k_scale": torch.zeros(nb, 1, bs) if fp8 else None,
        "v_scale": torch.zeros(nb, 1, bs) if fp8 else None,
    }


def run_qknorm(qkv, layer, kv_cache_dtype, out, qkv_partials=None):
    """fused_qknorm_idxrqknorm as SparseMHAPagedAttentionImpl calls it; returns the
    outputs with qkv, which the consumer writes back."""
    qw, kw, iqw, ikw = layer["norm_w"]
    aiter.fused_qknorm_idxrqknorm(
        qkv, qw, kw, layer["cos_sin"], layer["positions"], 16, 1,
        layer["rotary_dim"], 1e-6, iqw, ikw, 1, layer["slots"],
        out["kv_cache_k"], out["kv_cache_v"], out["index_cache"], layer["bs"],
        out["q_out"], out["index_q_out"], layer["slots"],
        kv_cache_dtype=kv_cache_dtype, index_cache_dtype=kv_cache_dtype,
        k_scale=out["k_scale"], v_scale=out["v_scale"], asm_layout=True,
        qkv_partials=qkv_partials,
    )  # fmt: skip
    return {**out, "qkv": qkv}


@benchmark()
def deferred_gemm(m, n, k):
    args, ref = gemm_inputs(m, n, k)
    base = aiter.gemm_a8w8_bpreshuffle(*args)
    y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
    partials = aiter.splitk_partials(y)
    ret = {"gfx": get_gfx_runtime(), "split_k": split_of(partials)}
    ret["bitwise"] = bitwise_equal(aiter.reduce_splitk_partials(y, partials), base)
    assert ret["bitwise"], f"deferred GEMM differs from gemm_a8w8_bpreshuffle {m=} {n=}"
    candidates = {
        "gemm+reduce": lambda: aiter.gemm_a8w8_bpreshuffle(*args),
        "deferred+reduce": lambda: aiter.reduce_splitk_partials(
            aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
        ),
    }
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        assert bitwise_equal(out, base), f"{name} differs from gemm+reduce {m=} {n=}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = 2 * m * n * k / us / 1e6
        ret[f"{name} err"] = checkAllclose(ref.float(), out.float(), msg=name)
    return ret


@benchmark()
def sparse_qkv_consumer(m, kv_cache_dtype):
    args, _ = gemm_inputs(m, SPARSE_N, SPARSE_K)
    layer = sparse_layer(m)

    def outputs():
        return qknorm_outputs(m, layer, kv_cache_dtype)

    def two_kernel(out):
        qkv = aiter.gemm_a8w8_bpreshuffle(*args)
        return run_qknorm(qkv, layer, kv_cache_dtype, out)

    def deferred(out):
        qkv = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(*args, max_m=MAX_M)
        partials = aiter.splitk_partials(qkv)
        return run_qknorm(qkv, layer, kv_cache_dtype, out, partials), split_of(partials)

    ret = {"gfx": get_gfx_runtime()}
    _, ret["gemm+reduce+qknorm us"] = run_perftest(two_kernel, outputs())
    _, ret["deferred+qknorm us"] = run_perftest(deferred, outputs())
    # Each path once more on fresh zeroed outputs, then every output bit for bit.
    base, (test, ret["split_k"]) = two_kernel(outputs()), deferred(outputs())
    diff = [
        n for n, t in base.items() if t is not None and not bitwise_equal(test[n], t)
    ]
    ret["bitwise"] = not diff
    assert not diff, f"{m=} {kv_cache_dtype=}: outputs differ: {diff}"
    return ret


def edge_partials(split_k, m, n):
    """Random fp32 partials plus columns whose sums are subnormal, overflow, NaN or a
    bf16 tie: where the consumer's denormal-flushing arithmetic could differ."""
    torch.manual_seed(split_k)
    p = torch.randn(split_k, m, n) * 4
    tiny = torch.finfo(torch.float32).tiny
    cases = [
        (1.5 * tiny, -tiny),  # subnormal sum of normal partials
        (3e-39, 1e-39),  # subnormal partials
        (3.3e38, 3.3e38),  # overflow to inf
        (float("inf"), float("-inf")),  # NaN
        (1.0, 2.0**-9),  # round-to-nearest-even tie
    ]
    for c, (a, b) in enumerate(cases):
        cols = slice(c, None, len(cases) + 2)
        p[:, :, cols] = 0.0
        p[0, :, cols], p[1, :, cols] = a, b
    return p


@benchmark()
def edge_values(m, split_k):
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        reduce_splitk_partials_gfx1250,
    )

    partials = edge_partials(split_k, m, SPARSE_N)
    ref = torch.empty(m, SPARSE_N, dtype=dtypes.bf16)
    reduce_splitk_partials_gfx1250(partials, ref)  # the standalone reduce kernel
    qkv = torch.empty(m, SPARSE_N, dtype=dtypes.bf16)
    layer = sparse_layer(m)
    out = qknorm_outputs(m, layer, "auto")
    _, us = run_perftest(run_qknorm, qkv, layer, "auto", out, partials, num_iters=11)
    ret = {"gfx": get_gfx_runtime(), "qknorm us": us}
    ret["bitwise"] = bitwise_equal(qkv, ref)
    assert ret["bitwise"], f"{m=} {split_k=}: write-back differs from the reduce"
    return ret


def run_suite(case, params, title=None):
    """Run case(*p) for every p; with a title, log the summary table. A split_k 1
    row compares the two-kernel path with itself, so when some M is deferrable,
    require that the tuned configs still split K somewhere."""
    rows = [case(*p) for p in params]
    if title:
        table = pd.DataFrame(rows).to_markdown(index=False)
        aiter.logger.info("%s summary (markdown):\n%s", title, table)
    if "split_k" in rows[0] and any(r["m"] <= MAX_M for r in rows):
        assert any(r["split_k"] > 1 for r in rows), f"no case split K: {params}"
    return rows


@pytest.mark.parametrize("n,k", NK_LIST)
def test_deferred_gemm(n, k):
    run_suite(deferred_gemm, [(m, n, k) for m in M_LIST])


@pytest.mark.parametrize("kv_cache_dtype", KV_DTYPES)
def test_sparse_qkv_consumer(kv_cache_dtype):
    run_suite(sparse_qkv_consumer, [(m, kv_cache_dtype) for m in M_LIST])


@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_edge_values(split_k):
    run_suite(edge_values, [(m, split_k) for m in M_LIST])


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("split-K consumer unsupported on %s", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(description="gfx1250 split-K consumer reduce")
    parser.add_argument("-m", type=int, nargs="*", default=M_LIST)
    parser.add_argument("-s", "--nk", type=dtypes.str2tuple, nargs="*", default=NK_LIST)
    parser.add_argument("--kv_cache_dtype", nargs="*", default=KV_DTYPES)
    args = parser.parse_args()
    prod = itertools.product
    nk = [(m, n, k) for (n, k), m in prod(args.nk, args.m)]
    run_suite(deferred_gemm, nk, "deferred GEMM + reduce")
    kv = [(m, d) for d, m in prod(args.kv_cache_dtype, args.m)]
    run_suite(sparse_qkv_consumer, kv, "sparse QKV consumer")
    sk = [(m, s) for s, m in prod((2, 4, 8), args.m)]
    run_suite(edge_values, sk, "sparse QKV consumer edge values")


if __name__ == "__main__":
    main()
