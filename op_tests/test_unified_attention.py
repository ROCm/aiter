# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 unified attention: public dispatch.

Timings use warm, fixed buffers; TB/s is logical traffic, not measured HBM bandwidth.
"""

import argparse
import itertools
import os
import sys
from functools import partial

import pandas as pd
import torch

# Plain-script CI invocation must select this checkout, not another editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
import aiter.ops.unified_attention as ua
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.quant import per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.triton_tests.utils.paged_attn_ref import ref_paged_attn

PAGE, H, HKV, D = 64, 64, 4, 128


def q8(x):
    return per_tensor_quant(x, quant_dtype=torch.float8_e4m3fn)


def make_case(query_lens, kv_lens, dtype, causal=True, seed=3, group=H // HKV):
    heads = HKV * group
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [max(0, (n + PAGE - 1) // PAGE) for n in kv_lens]
    n_pages = max(1, sum(pages))
    q, qs = q8(
        torch.randn(
            sum(query_lens), heads, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    k, ks = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    v, vs = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=dtypes.bf16, generator=g
        )
    )
    perm = torch.randperm(
        n_pages, generator=torch.Generator().manual_seed(seed + 1)
    ).tolist()
    bt = torch.zeros(len(pages), max(1, max(pages)), device="cuda", dtype=torch.int32)
    offset = 0
    for i, count in enumerate(pages):
        bt[i, :count] = torch.tensor(
            perm[offset : offset + count], device="cuda", dtype=torch.int32
        )
        offset += count
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(query_lens)), device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.full(q.shape, float("nan"), device="cuda", dtype=dtype),
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": D**-0.5,
        "causal": causal,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": qs,
        "k_descale": ks,
        "v_descale": vs,
    }


def reference(case, query_lens, kv_lens):
    return ref_paged_attn(
        case["q"],
        case["k"],
        case["v"],
        query_lens,
        kv_lens,
        case["block_table"],
        case["softmax_scale"],
        q_descale=case["q_descale"],
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        causal=case["causal"],
        sinks=case.get("sinks"),
        soft_cap=case["softcap"],
        sliding_window=(
            None if case["window_size"][0] < 0 else case["window_size"][0] + 1
        ),
    )


def shuffle_kv(k, v):
    nb, page, hkv, d = k.shape
    k = k.permute(0, 2, 3, 1).contiguous().view(nb, hkv, d // 16, 16, page)
    v = v.permute(0, 2, 3, 1).contiguous().view(nb, hkv, d, page // 16, 16)
    return k.permute(0, 1, 2, 4, 3).contiguous(), v.permute(0, 1, 3, 2, 4).contiguous()


def compare(want, got, atol, name):
    want, got = want.float(), got.float()
    err = checkAllclose(want, got, rtol=0, atol=atol, tol_err_ratio=0, msg=name)
    assert err == 0, f"{name}: {err:.3%} elements exceed atol={atol}"
    return err


def measure(candidates, case, want, query_lens, kv_lens, atol=None, bad_rows=False):
    # Preserve the old adapter's 8% global-scale gate, not element-relative rtol.
    if atol is None:
        atol = 0.08 * want.abs().max().item()
    heads = case["q"].shape[1]
    flops = 4 * heads * D * sum(q * k for q, k in zip(query_lens, kv_lens))
    nbytes = sum(query_lens) * heads * D * (1 + case["out"].element_size())
    nbytes += 2 * sum(kv_lens) * HKV * D
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        got = fn()
        err = compare(want, got, atol, name)
        if bad_rows:
            bad = int(
                (~torch.isclose(want.float(), got.float(), rtol=0, atol=atol))
                .any(dim=-1)
                .any(dim=-1)
                .sum()
                .item()
            )
            assert bad == 0, f"{name}: {bad} bad rows"
            ret[f"{name} bad_rows"] = bad
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{name}: empty timing"
        compare(want, got, atol, f"{name} after timing")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


PREFILL_CASES = {
    "base": ([512, 256], [512, 256], 16),
    # Small-Sq causal prefill is never declined for underfill; it has no host sync, so it captures.
    "small_sq": ([64, 64], [2048, 2048], 16),
    # Ragged pure prefill: M=970 > (B-1)*S+1=961, so no row can have query length 1.
    "varlen": ([320, 300, 200, 150], [2048, 5000, 8192, 3000], 16),
    # Ragged rows with an empty-KV row and partial last pages, per packed-BN64 GQA group.
    "varlen_causal": ([93, 80, 71, 60], [157, 100, 90, 70], 4),
    **{f"ragged_g{g}": ([93, 1, 17, 9], [157, 0, 5, 1], g) for g in (1, 4, 8, 16)},
}


@benchmark()
def test_prefill(dtype, causal, kind, layout):
    query_lens, kv_lens, group = PREFILL_CASES[kind]
    case = make_case(query_lens, kv_lens, dtype, causal, group=group)
    if kind.startswith("ragged"):
        # Poison V past each row's valid tokens: stale tails and page-zero prefetches
        # must not reach PV. Out is a slice of a NaN canary buffer, so any write past
        # the last query row is detected.
        for row, n in enumerate(kv_lens):
            if n % PAGE:
                page = int(case["block_table"][row, n // PAGE])
                case["v"].view(torch.uint8)[page, n % PAGE :] = 0x7F
        total = sum(query_lens)
        storage = torch.full(
            (total + 128, *case["q"].shape[1:]),
            float("nan"),
            device="cuda",
            dtype=dtype,
        )
        case["out"] = storage[:total]
    # An empty-KV row has no softmax support; the kernel writes zeros there.
    want = torch.nan_to_num(reference(case, query_lens, kv_lens))
    if layout == "vectorized":
        case["k"], case["v"] = shuffle_kv(case["k"], case["v"])
    case["shuffled_kv_cache"] = layout == "vectorized"
    call = partial(ua.unified_attention, **case, backend="flydsl")
    ret = measure({"flydsl": call}, case, want, query_lens, kv_lens)
    if kind.startswith("ragged"):
        empty = query_lens[0], query_lens[0] + query_lens[1]
        assert torch.count_nonzero(case["out"][empty[0] : empty[1]]) == 0
        assert torch.isnan(storage[total:]).all(), "write past the last query row"
    if kind == "small_sq" and causal:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        case["out"].fill_(float("nan"))
        graph.replay()
        compare(want, case["out"], 0.08 * want.abs().max().item(), "small-Sq graph")
    return ret


@benchmark()
def test_decode(dtype, depth, layout):
    query_lens = [1] * 8
    # Only the FlyDSL layout defines empty-KV output (zero); the Triton fallback gives NaN.
    empty_kv = 0 if layout == "vectorized" else 1
    kv_lens = [depth, max(1, depth // 2 + 3), depth, 65, depth, 1, depth, empty_kv]
    case = make_case(query_lens, kv_lens, dtype)
    want = reference(case, query_lens, kv_lens)
    if layout == "vectorized":
        case["k"], case["v"] = shuffle_kv(case["k"], case["v"])
    case["shuffled_kv_cache"] = layout == "vectorized"
    backend = "flydsl" if layout == "vectorized" else None
    name = "flydsl" if layout == "vectorized" else "triton_fallback"
    candidates = {name: partial(ua.unified_attention, **case, backend=backend)}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    if layout == "vectorized":
        compare(torch.zeros_like(want[-1]), case["out"][-1], 0, "empty KV")
    if layout == "linear":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidates[name]()
        case["out"].fill_(float("nan"))
        graph.replay()
        compare(want, case["out"], 0.08 * want.abs().max().item(), "decode graph")
    return ret


@benchmark()
def test_paged_addressing(pool_blocks, layout):
    query_lens, kv_lens = [128], [256]
    original = make_case(query_lens, kv_lens, dtypes.bf16)
    want = reference(original, query_lens, kv_lens)
    candidates, cases = {}, {}
    count = original["k"].shape[0]
    for name, ids in (
        ("identity", list(range(count))),
        ("reversed", list(reversed(range(count)))),
    ):
        case = dict(original, out=torch.empty_like(original["out"]))
        ids = [pool_blocks - count + i for i in ids]
        for key in ("k", "v"):
            pool = torch.zeros(
                pool_blocks, PAGE, HKV, D, device="cuda", dtype=original[key].dtype
            )
            for logical, pid in enumerate(ids):
                pool[pid] = original[key][int(original["block_table"][0, logical])]
            case[key] = pool
        case["block_table"] = torch.tensor([ids], device="cuda", dtype=torch.int32)
        if layout == "vectorized":
            case["k"], case["v"] = shuffle_kv(case["k"], case["v"])
        case["shuffled_kv_cache"] = layout == "vectorized"
        cases[name] = case
        candidates[name] = partial(ua.unified_attention, **case, backend="flydsl")
    ret = measure(candidates, cases["identity"], want, query_lens, kv_lens, atol=0.1)
    compare(
        cases["identity"]["out"],
        cases["reversed"]["out"],
        0,
        "page permutation invariance",
    )
    return ret


def main():
    arch = get_gfx_runtime()
    if get_gfx() != "gfx950" or arch != "gfx950":
        aiter.logger.warning(
            "FlyDSL unified attention requires gfx950; skipping (build=%s, attached=%s)",
            get_gfx(),
            arch,
        )
        return
    from aiter.ops.flydsl.unified_attention_kernels import is_flydsl_available

    if (
        not is_flydsl_available(torch.cuda.current_device())
        or torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        != 256
    ):
        aiter.logger.warning(
            "FlyDSL unified attention requires FlyDSL and a full-chip gfx950; skipping"
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FP8 unified attention correctness and warm-buffer timing sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        default=[dtypes.bf16, dtypes.fp16],
        choices=[dtypes.bf16, dtypes.fp16],
    )
    parser.add_argument("--causal", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    parser.add_argument("--decode-depth", type=int, nargs="+", default=[64, 8192])
    parser.add_argument(
        "--layout",
        choices=["linear", "vectorized"],
        nargs="+",
        default=["linear", "vectorized"],
    )
    args = parser.parse_args()
    sweeps = [
        (
            "prefill",
            test_prefill,
            [
                (d, c, k, lay)
                for d, c, k, lay in itertools.product(
                    args.dtype, args.causal, list(PREFILL_CASES), args.layout
                )
                # Causal batches with a length-1 row are possibly-mixed and decline
                # to the fallback by design; only non-causal reaches FlyDSL.
                if not (c and k.startswith("ragged"))
            ],
        ),
        (
            "decode",
            test_decode,
            itertools.product(args.dtype, args.decode_depth, args.layout),
        ),
        # 65535 pages stay below the dualwave launcher's signed-i32 element limit.
        (
            "paged addressing",
            test_paged_addressing,
            itertools.product([65535], args.layout),
        ),
    ]
    for name, fn, parameters in sweeps:
        rows = [fn(*values) for values in parameters]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )
    aiter.logger.info(
        "PASS: all unified-attention test groups; all candidate timings non-zero"
    )


if __name__ == "__main__":
    main()
