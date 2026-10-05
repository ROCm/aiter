# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 unified attention: public dispatch and focused direct-launch regressions.

Covers the FlyDSL gfx942 backend on Gemma-4's two layer shapes against an
independent fp32 paged-attention oracle. Timings use warm, fixed buffers;
TB/s is logical traffic, not measured HBM bandwidth. The run-only check
covers warm JIT-cache reuse, not wheel AOT packaging.
"""

import argparse
import importlib.util
import itertools
import math
import os
import sys
from functools import partial
from unittest import mock

import pandas as pd
import torch

# Plain-script CI invocation must select this checkout, not another editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
import aiter.ops.unified_attention as ua
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

PAGE = 64
# Gemma-4-31B layers: (head dim, query heads, KV heads, sliding window in keys).
SHAPES = {"full": (512, 32, 4, None), "sliding": (256, 32, 16, 1024)}
# Max |err| against the oracle, as a fraction of max |ref|.
TOLERANCE = 0.04
# Speed-vs-Triton column: FlyDSL time over Triton time, below 1.0 is faster.
RATIO = "flydsl / triton"
# One-token sequences at the decode boundaries, plus a (0, 0) padding entry.
DECODE_QUERY_LENS = [1] * 8 + [0]
DECODE_KV_LENS = [1, 65, 1023, 1024, 1025, 2047, 4097, 8192, 0]
# Serving-size batches for the opt-in Triton comparison: (query lens, KV lens).
SERVING = {
    "prefill 16K": ([16384], [16384]),
    "decode b1 1K": ([1], [1024]),
    "decode b4 1K": ([1] * 4, [1024] * 4),
    "decode b16 16K": ([1] * 16, [16384] * 16),
    "decode b64 4K": ([1] * 64, [4096] * 64),
    "1K prompt + 32 decodes 16K": ([1024] + [1] * 32, [1024] + [16384] * 32),
}


def q8(x):
    dtype = dtypes.get_dtype_fp8()
    s = x.abs().amax().clamp(min=1e-4) / torch.finfo(dtype).max
    return (x / s).to(dtype), s.reshape(1).float().to(x.device)


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    out_dtype=torch.float32,
    sliding_window=None,
    soft_cap=None,
    sinks=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    output_scale=None,
    causal=1,
):
    # Local copy of the established paged oracle; never import a test suite.
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start_idx = 0
    query = query.to(torch.float32)
    key_cache = key_cache.to(torch.float32)
    value_cache = value_cache.to(torch.float32)
    if q_descale is not None:
        query = query * q_descale
    if k_descale is not None:
        key_cache = key_cache * k_descale
    if v_descale is not None:
        value_cache = value_cache * v_descale
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        if causal:
            attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len
    out = torch.cat(outputs, dim=0)
    if output_scale is not None:
        out = out / output_scale
    return out.to(out_dtype)


def make_case(query_lens, kv_lens, shape, page=PAGE, seed=3):
    head_dim, num_heads, num_kv_heads, window = SHAPES[shape]
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*size):
        return torch.randn(*size, device="cuda", dtype=dtypes.bf16, generator=g)

    pages = [(n + page - 1) // page for n in kv_lens]
    n_pages = max(1, sum(pages))
    q, qs = q8(randn(sum(query_lens), num_heads, head_dim))
    k, ks = q8(randn(n_pages, page, num_kv_heads, head_dim))
    v, vs = q8(randn(n_pages, page, num_kv_heads, head_dim))
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
    # Slots past a sequence's last key hold FNUZ NaN: reading one poisons the output.
    for i, kv_len in enumerate(kv_lens):
        if kv_len % page:
            last = int(bt[i, pages[i] - 1])
            k.view(torch.uint8)[last, kv_len % page :] = 0x80
            v.view(torch.uint8)[last, kv_len % page :] = 0x80
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(query_lens)), device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.full(q.shape, float("nan"), device="cuda", dtype=dtypes.bf16),
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": head_dim**-0.5,
        "causal": True,
        "window_size": (-1, -1) if window is None else (window - 1, 0),
        "block_table": bt,
        "softcap": 0,
        "q_descale": qs,
        "k_descale": ks,
        "v_descale": vs,
    }


def window_of(case):
    return None if case["window_size"][0] < 0 else case["window_size"][0] + 1


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
        soft_cap=case["softcap"],
        sinks=case.get("sinks"),
        sliding_window=window_of(case),
    )


def interleave_kv(k, v):
    """K and V as views of one [blocks, kv_heads, page, 2 * head_dim] cache,
    the layout vLLM's ROCm attention backends split into K and V."""
    num_blocks, page, num_kv_heads, head_dim = k.shape
    cache = torch.empty(
        num_blocks, num_kv_heads, page, 2 * head_dim, device=k.device, dtype=k.dtype
    )
    cache[..., :head_dim] = k.transpose(1, 2)
    cache[..., head_dim:] = v.transpose(1, 2)
    kv = cache.transpose(1, 2)
    return kv[..., :head_dim], kv[..., head_dim:]


def direct_candidate(case, kv_lens, splits=None):
    """Launch past the support gate and dispatch policy; splits forces the
    decode split count."""
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    def launch():
        return adapter._launch(
            case["q"],
            case["k"],
            case["v"],
            case["out"],
            case["cu_seqlens_q"],
            case["max_seqlen_q"],
            case["seqused_k"],
            case["max_seqlen_k"],
            case["softmax_scale"],
            window_of(case),
            case["block_table"],
            case["q_descale"],
            case["k_descale"],
            case["v_descale"],
            num_kv_heads=case["k"].shape[2],
            block_size=case["k"].shape[1],
            num_seqs=len(kv_lens),
            num_kv_splits=splits,
        )

    return launch


def flydsl_candidate(case, kv_lens):
    """The public wrapper, or a direct launch where dispatch cedes to Triton."""
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    if adapter._cede_to_triton(
        case["q"].shape[-1], case["max_seqlen_q"], len(kv_lens), case["max_seqlen_k"]
    ):
        return direct_candidate(case, kv_lens)
    return partial(ua.unified_attention, **case, backend="flydsl")


def attended_pairs(query_lens, kv_lens, window):
    pairs = 0
    for q_len, kv_len in zip(query_lens, kv_lens):
        for pos in range(kv_len - q_len, kv_len):
            pairs += pos + 1 if window is None else min(pos + 1, window)
    return pairs


def compare(want, got, atol, name):
    want, got = want.float(), got.float()
    err = checkAllclose(want, got, rtol=0, atol=atol, tol_err_ratio=0, msg=name)
    assert err == 0, f"{name}: {err:.3%} elements exceed atol={atol}"
    return err


def measure(candidates, case, want, query_lens, kv_lens, atol=None):
    peak = want.float().abs().max().item()
    if atol is None:
        atol = TOLERANCE * peak
    _, num_heads, head_dim = case["q"].shape
    num_kv_heads = case["k"].shape[2]
    window = window_of(case)
    flops = 4 * num_heads * head_dim * attended_pairs(query_lens, kv_lens, window)
    keys_read = sum(
        kv if window is None else min(kv, q + window - 1)
        for q, kv in zip(query_lens, kv_lens)
    )
    nbytes = sum(query_lens) * num_heads * head_dim * (1 + case["out"].element_size())
    nbytes += 2 * keys_read * num_kv_heads * head_dim
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        got = fn()
        err = compare(want, got, atol, name)
        ret[f"{name} max err / max ref"] = (
            got.float() - want.float()
        ).abs().max().item() / peak
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{name}: empty timing"
        compare(want, got, atol, f"{name} after timing")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_prefill(shape, page):
    query_lens, kv_lens = [1024, 77], [1024, 77]
    case = make_case(query_lens, kv_lens, shape, page)
    want = reference(case, query_lens, kv_lens)
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    return measure(candidates, case, want, query_lens, kv_lens)


@benchmark()
def test_decode(shape, page, layout):
    case = make_case(DECODE_QUERY_LENS, DECODE_KV_LENS, shape, page)
    want = reference(case, DECODE_QUERY_LENS, DECODE_KV_LENS)
    if layout == "vllm":
        case["k"], case["v"] = interleave_kv(case["k"], case["v"])
    candidates = {"flydsl": flydsl_candidate(case, DECODE_KV_LENS)}
    return measure(candidates, case, want, DECODE_QUERY_LENS, DECODE_KV_LENS)


@benchmark()
def test_mixed_batch(shape, page, scale):
    query_lens = [1000, 37, 513, 1, 1, 1, 1, 1]
    kv_lens = [1000, 4000, 1213, 1500, 3000, 64, 4097, 9]
    case = make_case(query_lens, kv_lens, shape, page)
    if scale == "1 (Gemma-4)":
        case["softmax_scale"] = 1.0
    want = reference(case, query_lens, kv_lens)
    _, num_heads, head_dim = case["q"].shape
    storage = torch.full(
        (sum(query_lens) + 64, num_heads, head_dim),
        float("nan"),
        device="cuda",
        dtype=dtypes.bf16,
    )
    case["out"] = storage[: sum(query_lens)]
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    assert torch.isnan(
        storage[sum(query_lens) :]
    ).all(), "kernel wrote beyond packed output"
    return ret


@benchmark()
def test_cross_attention_mask(shape, query_len, kv_len):
    case = make_case([query_len], [kv_len], shape)
    want = reference(case, [query_len], [kv_len])
    candidates = {"flydsl": partial(ua.unified_attention, **case, backend="flydsl")}
    return measure(candidates, case, want, [query_len], [kv_len])


@benchmark()
def test_splitk_combine(shape, splits):
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    case = make_case(DECODE_QUERY_LENS, DECODE_KV_LENS, shape)
    want = reference(case, DECODE_QUERY_LENS, DECODE_KV_LENS)
    launch = direct_candidate(case, DECODE_KV_LENS, splits)
    launch()
    # Every split must write its partial before the combine reads it.
    for workspace in adapter._workspaces.values():
        workspace.fill_(float("nan"))
    case["out"].fill_(float("nan"))
    ret = measure({"splitk": launch}, case, want, DECODE_QUERY_LENS, DECODE_KV_LENS)
    ret["splits run"] = splits
    return ret


@benchmark()
def test_paged_addressing(shape, boundary_gib, layout):
    query_lens, kv_lens = [1, 33], [65, 1057]
    original = make_case(query_lens, kv_lens, shape)
    want = reference(original, query_lens, kv_lens)
    count = original["k"].shape[0]
    first = (boundary_gib << 30) // original["k"][0].numel() + 1
    candidates, cases = {}, {}
    for name, ids in (
        ("identity", list(range(count))),
        ("reversed", list(reversed(range(count)))),
    ):
        # Pages sit above the byte boundary; the pages below them are NaN, so
        # an offset that wraps at the boundary reads poison.
        case = dict(original, out=torch.full_like(original["out"], float("nan")))
        for key in ("k", "v"):
            pool = torch.empty(
                (first + count,) + tuple(original[key].shape[1:]),
                device="cuda",
                dtype=original[key].dtype,
            )
            pool[: count + 1].view(torch.uint8).fill_(0x80)
            pool[first + torch.tensor(ids, device="cuda")] = original[key]
            case[key] = pool
        remap = first + torch.tensor(ids, device="cuda", dtype=torch.int32)
        case["block_table"] = remap[original["block_table"].long()]
        if layout == "vllm":
            case["k"], case["v"] = interleave_kv(case["k"], case["v"])
        cases[name] = case
        candidates[name] = direct_candidate(case, kv_lens)
    ret = measure(candidates, cases["identity"], want, query_lens, kv_lens)
    compare(
        cases["identity"]["out"],
        cases["reversed"]["out"],
        0,
        "page permutation invariance",
    )
    ret["K pool GiB"] = cases["identity"]["k"].numel() / 2**30
    return ret


@benchmark()
def test_routing_backend_gate(config):
    import aiter.ops.flydsl.unified_attention_kernels as adapter

    if config == "ceded decode":
        query_lens, kv_lens = [1] * 2, [1024] * 2
        case = make_case(query_lens, kv_lens, "full")
    else:
        query_lens, kv_lens = [256, 1], [256, 700]
        case = make_case(query_lens, kv_lens, "sliding")
    if config == "softcap":
        case["softcap"] = 30.0
    elif config == "sinks":
        case["sinks"] = torch.zeros(case["q"].shape[1], device="cuda")
    want = reference(case, query_lens, kv_lens)
    explicit = partial(ua.unified_attention, **case, backend="flydsl")
    if config != "supported":
        try:
            explicit()
        except RuntimeError as exc:
            assert "does not support this configuration" in str(exc)
        else:
            raise AssertionError("explicit FlyDSL silently accepted a declined config")
    real = adapter.flydsl_unified_attention
    with mock.patch.object(adapter, "flydsl_unified_attention", wraps=real) as spy:
        auto = ua.unified_attention(**case).clone()
        assert spy.call_count > 0, "automatic dispatch never offered the call to FlyDSL"
    compare(want, auto, TOLERANCE * want.abs().max().item(), "automatic routing")
    candidates = {"triton": partial(ua.unified_attention, **case, backend="triton")}
    with mock.patch.object(
        adapter,
        "flydsl_unified_attention",
        side_effect=AssertionError("explicit backend selected FlyDSL"),
    ):
        ret = measure(candidates, case, want, query_lens, kv_lens)
        if config != "supported":
            compare(auto, case["out"], 0, "declined config vs Triton bitwise")
    if config == "supported":
        ret.update(measure({"flydsl": explicit}, case, want, query_lens, kv_lens))
    return ret


@benchmark()
def test_warm_cache_run_only(shape, path):
    from aiter.aot.flydsl.common import run_only_env

    query_lens, kv_lens = ([256], [256]) if path == "prefill" else ([1] * 8, [4096] * 8)
    case = make_case(query_lens, kv_lens, shape, seed=17)
    want = reference(case, query_lens, kv_lens)
    call = flydsl_candidate(case, kv_lens)
    call()
    torch.cuda.synchronize()
    case["out"].fill_(float("nan"))
    # Missing artifacts must raise rather than quietly compile a replacement.
    with run_only_env():
        return measure({"warm_run_only": call}, case, want, query_lens, kv_lens)


@benchmark()
def test_speed_vs_triton(shape, workload):
    query_lens, kv_lens = SERVING[workload]
    case = make_case(query_lens, kv_lens, shape)
    triton = partial(ua.unified_attention, **case, backend="triton")
    # The oracle is too slow at serving sizes, so Triton is the reference.
    want = triton().clone()
    candidates = {"triton": triton, "flydsl": flydsl_candidate(case, kv_lens)}
    ret = measure(
        candidates,
        case,
        want,
        query_lens,
        kv_lens,
        atol=2 * TOLERANCE * want.abs().max().item(),
    )
    ret[RATIO] = ret["flydsl us"] / ret["triton us"]
    return ret


def with_geomean(df):
    """Append the geometric-mean time ratio per shape and over every row."""

    def geomean(ratios):
        return math.exp(ratios.map(math.log).mean())

    means = [
        {"shape": shape, "workload": "geomean", RATIO: geomean(rows[RATIO])}
        for shape, rows in df.groupby("shape", sort=False)
    ]
    means.append({"shape": "all", "workload": "geomean", RATIO: geomean(df[RATIO])})
    df = pd.concat([df, pd.DataFrame(means)], ignore_index=True)
    # None, unlike NaN, prints as an empty cell.
    return df.astype(object).where(df.notna(), None)


def main():
    arch = get_gfx_runtime()
    if get_gfx() != "gfx942" or arch != "gfx942":
        aiter.logger.warning(
            "FlyDSL unified attention requires gfx942; skipping (build=%s, attached=%s)",
            get_gfx(),
            arch,
        )
        return
    if importlib.util.find_spec("flydsl") is None:
        aiter.logger.warning("FlyDSL unified attention requires FlyDSL; skipping")
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FP8 unified attention correctness and warm-buffer timing sweep",
    )
    parser.add_argument(
        "--shape", choices=list(SHAPES), nargs="+", default=list(SHAPES)
    )
    parser.add_argument(
        "--page", type=int, choices=[32, 64, 128], nargs="+", default=[32, 64, 128]
    )
    parser.add_argument(
        "--layout", choices=["plain", "vllm"], nargs="+", default=["plain", "vllm"]
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 5, 16, 64])
    parser.add_argument("--boundary-gib", type=int, nargs="+", default=[2, 4])
    parser.add_argument(
        "--perf",
        action="store_true",
        help="also time FlyDSL against Triton on serving-size batches",
    )
    args = parser.parse_args()
    sweeps = [
        ("prefill", test_prefill, itertools.product(args.shape, args.page)),
        (
            "decode",
            test_decode,
            itertools.product(args.shape, args.page, args.layout),
        ),
        (
            "mixed batch",
            test_mixed_batch,
            itertools.product(args.shape, args.page, ["1/sqrt(d)", "1 (Gemma-4)"]),
        ),
        (
            "cross-attention mask",
            test_cross_attention_mask,
            itertools.product(args.shape, [320], [1024, 2000]),
        ),
        (
            "split-K combine",
            test_splitk_combine,
            itertools.product(args.shape, args.splits),
        ),
        (
            "paged addressing",
            test_paged_addressing,
            itertools.product(args.shape, args.boundary_gib, args.layout),
        ),
        (
            "routing/backend gate",
            test_routing_backend_gate,
            itertools.product(["supported", "softcap", "sinks", "ceded decode"]),
        ),
        (
            "warm-cache run-only (not full AOT)",
            test_warm_cache_run_only,
            itertools.product(args.shape, ["prefill", "decode"]),
        ),
    ]
    if args.perf:
        sweeps.append(
            (
                "speed vs Triton",
                test_speed_vs_triton,
                itertools.product(args.shape, SERVING),
            )
        )
    for name, fn, parameters in sweeps:
        rows = [fn(*values) for values in parameters]
        df = pd.DataFrame(rows)
        if RATIO in df:
            df = with_geomean(df)
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            df.to_markdown(index=False, missingval=""),
        )
    aiter.logger.info(
        "PASS: all unified-attention test groups; all candidate timings non-zero"
    )


if __name__ == "__main__":
    main()
