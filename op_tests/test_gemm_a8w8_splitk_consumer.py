# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K reduction left to the consumer of the gfx1250 PTPC bpreshuffle GEMM.

gemm_a8w8_bpreshuffle_splitk_deferred skips the standalone split-K reduce kernel
of a decode-size GEMM and leaves the fp32 partials to the kernel that reads the
output. Both tests run the MiniMax-M3 TP4 decode shapes and require every output
to be bitwise what the two-kernel path (gemm_a8w8_bpreshuffle) produces:

* test_deferred_gemm: deferred GEMM + reduce_splitk_partials, the fallback for a
  consumer that does not take partials (sparse QKV, dense QKV, dense gate/up).
* test_sparse_qkv_consumer: deferred GEMM + fused_qknorm_idxrqknorm(qkv_partials=)
  against GEMM + reduce + fused_qknorm_idxrqknorm as the sparse attention layer
  calls it: q_out, index q, page-16 SHUFFLE K/V cache and per-token scales, index
  cache, and the qkv the consumer writes back.
* test_sparse_consumer_edge_values: crafted partials whose sums are subnormal,
  overflow, NaN or ties; the written-back qkv against the reduce kernel's output.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # the split-K PTPC GEMM is the gfx1250 WMMA backend
HEAD_DIM = 128
MAX_M = 64  # ATOM_SPLITK_CONSUMER_REDUCE_MAX_M


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


def make_gemm_inputs(m, n, k, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=dtypes.bf16)
    w = torch.randn(n, k, dtype=dtypes.bf16)
    xq, xs = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, ws = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    return xq, shuffle_weight(wq, layout=(16, 16)), xs, ws, (xq, wq)


def run_torch(xq, wq, xs, ws):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    return ((xq.float() * xs) @ (wq.float() * ws).t()).to(dtypes.bf16)


def split_factor(y):
    """split_k a deferred output carries (claims and reduces it), else 1."""
    partials = aiter.splitk_partials(y)
    if partials is None:
        return 1
    aiter.reduce_splitk_partials(y, partials)
    return partials.shape[0]


@benchmark()
def test_deferred_gemm(m, n, k):
    xq, wsh, xs, ws, raw = make_gemm_inputs(m, n, k)
    ref = run_torch(*raw, xs, ws)

    def deferred_then_reduce():
        y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(xq, wsh, xs, ws, max_m=MAX_M)
        return aiter.reduce_splitk_partials(y)

    candidates = {
        "gemm+reduce": lambda: aiter.gemm_a8w8_bpreshuffle(xq, wsh, xs, ws),
        "deferred+reduce": deferred_then_reduce,
    }
    base = candidates["gemm+reduce"]()
    y = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(xq, wsh, xs, ws, max_m=MAX_M)
    ret = {"gfx": get_gfx(), "split_k": split_factor(y)}
    ret["bitwise"] = bitwise_equal(y, base)
    assert ret["bitwise"], f"deferred GEMM differs from gemm_a8w8_bpreshuffle {m=}"

    flops = 2 * m * n * k
    nbytes = m * k + n * k + m * n * 2
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        assert bitwise_equal(out, base), f"{name} differs from gemm+reduce {m=}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref.float(), out.float(), msg=f"{name}: M={m} N={n} K={k}"
        )
    return ret


def make_sparse_layer(m, rotary_dim=64, block_size=16, seed=1):
    """One MiniMax-M3 TP4 sparse layer's q/k norm + RoPE + cache-insert state: 16 q
    heads, 1 kv head, 1 index head, partial rotary, page-16 SHUFFLE caches."""
    torch.manual_seed(seed)
    max_pos = 4096
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = torch.outer(torch.arange(max_pos).float(), inv_freq)
    num_blocks = (m + block_size - 1) // block_size + 2
    return {
        "q_norm_weight": torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1,
        "k_norm_weight": torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1,
        "index_q_norm_weight": torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1,
        "index_k_norm_weight": torch.randn(HEAD_DIM, dtype=dtypes.bf16) * 0.1,
        "cos_sin_cache": torch.cat([freqs.cos(), freqs.sin()], -1).to(dtypes.bf16),
        "positions": torch.randint(0, max_pos, (m,), dtype=torch.int64),
        "slot_mapping": torch.randperm(num_blocks * block_size)[:m].to(torch.int64),
        "num_blocks": num_blocks,
        "block_size": block_size,
        "rotary_dim": rotary_dim,
    }


def make_sparse_outputs(m, layer, kv_cache_dtype):
    fp8 = kv_cache_dtype == "fp8"
    cache_dtype = dtypes.fp8 if fp8 else dtypes.bf16
    x = 16 // torch.empty(0, dtype=cache_dtype).element_size()
    nb, bs = layer["num_blocks"], layer["block_size"]
    return {
        "q_out": torch.empty(m, 16 * HEAD_DIM, dtype=dtypes.bf16),
        "index_q_out": torch.empty(m, HEAD_DIM, dtype=dtypes.bf16),
        "kv_cache_k": torch.zeros(nb, 1, HEAD_DIM // x, bs, x, dtype=cache_dtype),
        "kv_cache_v": torch.zeros(nb, 1, bs // x, HEAD_DIM, x, dtype=cache_dtype),
        "index_cache": torch.zeros(nb * bs, HEAD_DIM, dtype=cache_dtype),
        "k_scale": torch.zeros(nb, 1, bs, dtype=dtypes.fp32) if fp8 else None,
        "v_scale": torch.zeros(nb, 1, bs, dtype=dtypes.fp32) if fp8 else None,
    }


def run_sparse_consumer(qkv, layer, out, kv_cache_dtype, qkv_partials=None):
    # As SparseMHAPagedAttentionImpl.rope_cache calls it (layer without skip).
    aiter.fused_qknorm_idxrqknorm(
        qkv,
        layer["q_norm_weight"],
        layer["k_norm_weight"],
        layer["cos_sin_cache"],
        layer["positions"],
        16,
        1,
        layer["rotary_dim"],
        1e-6,
        layer["index_q_norm_weight"],
        layer["index_k_norm_weight"],
        1,
        layer["slot_mapping"],
        out["kv_cache_k"],
        out["kv_cache_v"],
        out["index_cache"],
        layer["block_size"],
        out["q_out"],
        out["index_q_out"],
        layer["slot_mapping"],
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype=kv_cache_dtype,
        k_scale=out["k_scale"],
        v_scale=out["v_scale"],
        asm_layout=True,
        qkv_partials=qkv_partials,
    )


@benchmark()
def test_sparse_qkv_consumer(m, n, k, kv_cache_dtype):
    xq, wsh, xs, ws, _ = make_gemm_inputs(m, n, k)
    layer = make_sparse_layer(m)
    outs = {
        name: make_sparse_outputs(m, layer, kv_cache_dtype)
        for name in ("gemm+reduce+qknorm", "deferred+qknorm")
    }
    qkvs = {}

    def two_kernel_path():
        qkv = aiter.gemm_a8w8_bpreshuffle(xq, wsh, xs, ws)
        run_sparse_consumer(qkv, layer, outs["gemm+reduce+qknorm"], kv_cache_dtype)
        qkvs["gemm+reduce+qknorm"] = qkv

    def deferred_path():
        qkv = aiter.gemm_a8w8_bpreshuffle_splitk_deferred(xq, wsh, xs, ws, max_m=MAX_M)
        partials = aiter.splitk_partials(qkv)
        qkvs["split_k"] = 1 if partials is None else partials.shape[0]
        run_sparse_consumer(
            qkv, layer, outs["deferred+qknorm"], kv_cache_dtype, partials
        )
        qkvs["deferred+qknorm"] = qkv

    candidates = {
        "gemm+reduce+qknorm": two_kernel_path,
        "deferred+qknorm": deferred_path,
    }
    nbytes = m * k + n * k + 2 * m * n * 2  # GEMM in/weight, qkv written and read
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    ret["split_k"] = qkvs["split_k"]

    # One more run of each on zeroed outputs, then every output bit for bit.
    for name, fn in candidates.items():
        for t in outs[name].values():
            if t is not None:
                t.zero_()
        fn()
    base, test = outs["gemm+reduce+qknorm"], outs["deferred+qknorm"]
    checks = {"qkv": bitwise_equal(qkvs["deferred+qknorm"], qkvs["gemm+reduce+qknorm"])}
    checks.update(
        {
            key: bitwise_equal(test[key], base[key])
            for key in base
            if base[key] is not None
        }
    )
    ret["deferred+qknorm err"] = checkAllclose(
        base["q_out"].float(),
        test["q_out"].float(),
        rtol=0,
        atol=0,
        msg=f"deferred+qknorm q_out: M={m} {kv_cache_dtype}",
    )
    ret["bitwise"] = all(checks.values())
    assert ret["bitwise"], f"{m=} {kv_cache_dtype=}: outputs differ: {checks}"
    return ret


def edge_partials(split_k, m, n):
    """Random fp32 partials, with columns whose sums are subnormal, overflow, NaN or
    a bf16 tie: where the consumer's (denormal-flushing) arithmetic could part from
    the reduce kernel's."""
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
        p[:, :, c :: len(cases) + 2] = 0.0
        p[0, :, c :: len(cases) + 2] = a
        p[1, :, c :: len(cases) + 2] = b
    return p


@benchmark()
def test_sparse_consumer_edge_values(m, split_k):
    # Reference: the standalone reduce kernel over the same partials.
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import (
        reduce_splitk_partials_gfx1250,
    )

    partials = edge_partials(split_k, m, 2560)
    ref = torch.empty(m, 2560, dtype=dtypes.bf16)
    reduce_splitk_partials_gfx1250(partials, ref)
    qkv = torch.empty(m, 2560, dtype=dtypes.bf16)
    layer = make_sparse_layer(m)
    out = make_sparse_outputs(m, layer, "auto")
    _, us = run_perftest(
        run_sparse_consumer, qkv, layer, out, "auto", partials, num_iters=11
    )
    ret = {"gfx": get_gfx(), "qknorm us": us, "bitwise": bitwise_equal(qkv, ref)}
    assert ret["bitwise"], f"{m=} {split_k=}: write-back differs from the reduce"
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("split-K consumer reduce unsupported on %s", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="gfx1250 split-K reduction left to the consumer",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 4, 8, 16, 32, 48, 64],
        help="decode tokens per TP4 rank (bs x (1 + draft tokens))",
    )
    parser.add_argument(
        "-s",
        "--nk",
        type=dtypes.str2tuple,
        nargs="*",
        # MiniMax-M3 TP4: sparse QKV, dense QKV, dense gate/up
        default=[(2560, 6144), (2304, 6144), (6144, 6144)],
        help="GEMM N,K for test_deferred_gemm, e.g. -s 2560,6144",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        nargs="*",
        default=["fp8", "auto"],
        help="sparse layer KV/index cache dtype",
    )
    args = parser.parse_args()

    rows = [
        test_deferred_gemm(m, n, k) for (n, k), m in itertools.product(args.nk, args.m)
    ]
    aiter.logger.info(
        "deferred GEMM + reduce summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    rows = [
        test_sparse_qkv_consumer(m, 2560, 6144, kv)
        for kv, m in itertools.product(args.kv_cache_dtype, args.m)
    ]
    aiter.logger.info(
        "sparse QKV consumer summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    rows = [
        test_sparse_consumer_edge_values(m, sk)
        for sk, m in itertools.product((2, 4, 8), args.m)
    ]
    aiter.logger.info(
        "sparse QKV consumer edge values summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
