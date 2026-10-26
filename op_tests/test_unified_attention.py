# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 unified attention: public dispatch and focused direct-launch regressions.

Covers the FlyDSL gfx942 backend on Gemma-4's two layer shapes against the
Triton suite's fp32 paged-attention oracle. Routing and support-gate checks
run on every architecture with stubbed kernels and CPU tensors. The AOT
matrix runs first and loads every prebuilt kernel with JIT disabled, so it
needs the cache aiter's build writes, or one from
``python -m aiter.aot.flydsl.unified_attention`` in ``FLYDSL_RUNTIME_CACHE_DIR``.
Timings use warm, fixed buffers; TB/s is logical traffic, not measured HBM
bandwidth. Performance against Triton is
op_tests/op_benchmarks/flydsl/bench_unified_attention.py.
"""

import argparse
import importlib
import itertools
import os
import sys
from functools import partial
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import torch

# Plain-script CI invocation must select this checkout, not another editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aiter
import aiter.ops.unified_attention as ua
from aiter import dtypes
from aiter.aot.flydsl.common import run_only_env
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

PAGE = 64
# Gemma-4-31B layers: (head dim, query heads, KV heads, sliding window in keys).
SHAPES = {"full": (512, 32, 4, None), "sliding": (256, 32, 16, 1024)}
# Max |err| against the oracle: TOLERANCE of the batch's max |ref|, and
# ROW_TOLERANCE of the max |ref| of the element's own sequence and head, so a
# row of small outputs, such as a long-context average, cannot drift or vanish.
# Rounding P to FNUZ (half-step 2**-4 relative) costs a single row about 5%.
TOLERANCE = 0.04
ROW_TOLERANCE = 0.1
# One-token sequences at the decode boundaries, plus a (0, 0) padding entry.
DECODE_QUERY_LENS = [1] * 8 + [0]
DECODE_KV_LENS = [1, 65, 1023, 1024, 1025, 2047, 4097, 8192, 0]


def q8(x):
    dtype = dtypes.get_dtype_fp8()
    s = x.abs().amax().clamp(min=1e-4) / torch.finfo(dtype).max
    return (x / s).to(dtype), s.reshape(1).float().to(x.device)


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
    """The Triton suite's fp32 oracle. A query whose position is below 0 sees
    no key; the oracle leaves its row NaN and the kernel writes zeros."""
    # Imported here: the Triton suite reads the GPU architecture at import.
    from op_tests.triton_tests.attention.test_unified_attention import (
        ref_paged_attn,
    )

    want = ref_paged_attn(
        case["q"],
        case["k"],
        case["v"],
        query_lens,
        kv_lens,
        case["block_table"],
        case["softmax_scale"],
        torch.float32,
        sliding_window=window_of(case),
        soft_cap=case["softcap"],
        q_descale=case["q_descale"],
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        causal=case["causal"],
    )
    start = 0
    for q_len, kv_len in zip(query_lens, kv_lens):
        want[start : start + max(0, q_len - kv_len)] = 0
        start += q_len
    return want


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


def direct_candidate(case, kv_lens, splits=None, prefill_splits=None):
    """Launch past the support gate and dispatch policy; splits forces the
    decode split count, prefill_splits a lone sequence's KV split count."""
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
            num_prefill_splits=prefill_splits,
        )

    return launch


def flydsl_candidate(case, kv_lens):
    """The public wrapper, or a direct launch where dispatch cedes to Triton."""
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    if adapter._cede_to_triton(
        case["q"].shape[-1],
        case["max_seqlen_q"],
        len(kv_lens),
        case["max_seqlen_k"],
        case["k"].shape[1],
        window_of(case),
    ):
        return direct_candidate(case, kv_lens)
    return partial(ua.unified_attention, **case, backend="flydsl")


def attended_pairs(query_lens, kv_lens, window):
    pairs = 0
    for q_len, kv_len in zip(query_lens, kv_lens):
        for pos in range(max(0, kv_len - q_len), kv_len):
            pairs += pos + 1 if window is None else min(pos + 1, window)
    return pairs


def row_max(ref, query_lens):
    """Each element's (sequence, head) max |ref|."""
    ref = ref.float().abs()
    peak = torch.zeros_like(ref)
    start = 0
    for q_len in query_lens:
        if q_len:
            rows = ref[start : start + q_len]
            peak[start : start + q_len] = rows.amax(dim=(0, 2), keepdim=True)
        start += q_len
    return peak


def allowed_err(want, query_lens):
    """The largest error allowed at each element. A row whose reference is
    zero, such as a query that sees no key, must be zero."""
    cap = TOLERANCE * want.float().abs().max().item()
    allowed = (ROW_TOLERANCE * row_max(want, query_lens)).clamp(max=cap)
    return allowed.clamp(min=torch.finfo(torch.float32).tiny)


def err_ratio(want, got, query_lens):
    """max |got - want| as a fraction of the error allowed there."""
    allowed = allowed_err(want, query_lens)
    return ((got.float() - want.float()).abs() / allowed).max().item()


def compare(want, got, query_lens, name):
    allowed = allowed_err(want, query_lens)
    err = checkAllclose(
        want.float() / allowed,
        got.float() / allowed,
        rtol=0,
        atol=1,
        tol_err_ratio=0,
        msg=name,
    )
    assert err == 0, f"{name}: {err:.3%} elements exceed the allowed error"
    return err


def assert_same(a, b, name):
    err = checkAllclose(a.float(), b.float(), rtol=0, atol=0, tol_err_ratio=0, msg=name)
    assert err == 0, f"{name}: {err:.3%} elements differ"


def measure(candidates, case, want, query_lens, kv_lens):
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
        err = compare(want, got, query_lens, name)
        ret[f"{name} max err / allowed"] = err_ratio(want, got, query_lens)
        _, us = run_perftest(fn, num_rotate_args=1)
        assert us > 0, f"{name}: empty timing"
        compare(want, got, query_lens, f"{name} after timing")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_aot_matrix(shape, page, layout, path):
    """Every prebuilt attention and combine kernel loads with JIT disabled.

    Runs before the other cases: a launcher this process already compiled
    returns before FlyDSL's run-only cache check.
    """
    if path == "prefill":
        # A prefix chunk: the unified build, and its KV splits' combine.
        query_lens, kv_lens = [64], [1024]
    else:
        query_lens, kv_lens = [1] * 4, [1024] * 4
    case = make_case(query_lens, kv_lens, shape, page=page, seed=23)
    want = reference(case, query_lens, kv_lens)
    if layout == "vllm":
        case["k"], case["v"] = interleave_kv(case["k"], case["v"])
    call = direct_candidate(case, kv_lens)
    with run_only_env():
        return measure({"aot_run_only": call}, case, want, query_lens, kv_lens)


@benchmark()
def test_prefill(shape, page):
    query_lens, kv_lens = [1024, 77], [1024, 77]
    case = make_case(query_lens, kv_lens, shape, page)
    want = reference(case, query_lens, kv_lens)
    candidates = {"flydsl": flydsl_candidate(case, kv_lens)}
    return measure(candidates, case, want, query_lens, kv_lens)


@benchmark()
def test_decode(shape, page, layout):
    case = make_case(DECODE_QUERY_LENS, DECODE_KV_LENS, shape, page)
    want = reference(case, DECODE_QUERY_LENS, DECODE_KV_LENS)
    if layout == "vllm":
        case["k"], case["v"] = interleave_kv(case["k"], case["v"])
    candidates = {"flydsl": flydsl_candidate(case, DECODE_KV_LENS)}
    ret = measure(candidates, case, want, DECODE_QUERY_LENS, DECODE_KV_LENS)
    # The 4097- and 8192-key rows are small next to the one-key rows; losing
    # them must still fail the comparison.
    lost = case["out"].clone()
    lost[6:8] = 0
    assert (
        err_ratio(want, lost, DECODE_QUERY_LENS) > 1
    ), "the comparison accepted zeroed long-context rows"
    return ret


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
    candidates = {"flydsl": flydsl_candidate(case, kv_lens)}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    assert torch.isnan(
        storage[sum(query_lens) :]
    ).all(), "kernel wrote beyond packed output"
    return ret


@benchmark()
def test_no_visible_key(shape, path):
    """Queries at positions below 0 see no key and write zeros: a prefill
    longer than its KV, or a decode with no KV. The blind sequence comes
    first, behind a block-table entry and unused entries that name an
    out-of-range page, so K/V loaded through column -1 faults."""
    if path == "prefill":
        query_lens, kv_lens = [64, 64], [1, 1024]
    else:
        query_lens, kv_lens = [1, 1], [0, 77]
    case = make_case(query_lens, kv_lens, shape)
    want = reference(case, query_lens, kv_lens)
    table = case["block_table"]
    storage = torch.full(
        (table.numel() + 1,), 1 << 30, device=table.device, dtype=table.dtype
    )
    case["block_table"] = storage[1:].view(table.shape)
    page = case["k"].shape[1]
    for i, kv_len in enumerate(kv_lens):
        used = -(-kv_len // page)
        case["block_table"][i, :used] = table[i, :used]
    candidates = {"flydsl": flydsl_candidate(case, kv_lens)}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    blind = query_lens[0] - kv_lens[0]
    assert (case["out"][:blind] == 0).all(), "a query that sees no key is not zero"
    return ret


@benchmark()
def test_overestimated_max_seqlen_q(shape):
    """A graph-capture upper bound must not misclassify a lone decode as a
    prefill and leave its output row unwritten."""
    query_lens, kv_lens = [1], [4096]
    case = make_case(query_lens, kv_lens, shape)
    case["max_seqlen_q"] = 16
    # Singleton rank is not part of the public per-tensor descale contract.
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = case[name].reshape(1, 1)
    want = reference(case, query_lens, kv_lens)
    candidates = {
        "flydsl_overestimated_q": partial(
            ua.unified_attention, **case, backend="flydsl"
        )
    }
    ret = measure(candidates, case, want, query_lens, kv_lens)

    # Exercise _launch's unified-build fallback too. Public vLLM calls slice Q
    # to real tokens, but padded storage must not resurrect the unwritten-row
    # bug for another caller.
    padding = torch.zeros(
        (3,) + tuple(case["q"].shape[1:]),
        device=case["q"].device,
        dtype=case["q"].dtype,
    )
    case["q"] = torch.cat((case["q"], padding))
    case["out"] = torch.full(
        case["q"].shape, float("nan"), device="cuda", dtype=dtypes.bf16
    )
    got = direct_candidate(case, kv_lens)()
    compare(want, got[:1], query_lens, "padded direct launch")
    assert torch.isnan(got[1:]).all(), "padded query rows were written"
    ret["padded direct max err / allowed"] = err_ratio(want, got[:1], query_lens)
    return ret


def nan_workspace(device, numel):
    """Split partials pre-filled with NaN: one the kernel never writes poisons
    the combine."""
    return torch.full((numel,), float("nan"), device=device, dtype=torch.float32)


def check_on_nan_workspace(launch, case, want, query_lens, name):
    """Every split must write its partial before a combine reads it."""
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    with mock.patch.object(adapter, "_workspace", nan_workspace):
        compare(want, launch(), query_lens, name)
    case["out"].fill_(float("nan"))


@benchmark()
def test_splitk_combine(shape, splits):
    case = make_case(DECODE_QUERY_LENS, DECODE_KV_LENS, shape)
    want = reference(case, DECODE_QUERY_LENS, DECODE_KV_LENS)
    launch = direct_candidate(case, DECODE_KV_LENS, splits)
    check_on_nan_workspace(
        launch, case, want, DECODE_QUERY_LENS, f"{splits} splits on a NaN workspace"
    )
    ret = measure({"splitk": launch}, case, want, DECODE_QUERY_LENS, DECODE_KV_LENS)
    ret["splits run"] = splits
    return ret


@benchmark()
def test_prefill_splits(shape, page, chunk, splits):
    """A lone prefix chunk whose KV range splits across workgroups, merged by
    the prefill combine; None takes the planner's count."""
    query_lens, kv_lens = [chunk[0]], [chunk[1]]
    case = make_case(query_lens, kv_lens, shape, page)
    case["softmax_scale"] = 1.0
    want = reference(case, query_lens, kv_lens)
    case["k"], case["v"] = interleave_kv(case["k"], case["v"])
    launch = direct_candidate(case, kv_lens, prefill_splits=splits)
    check_on_nan_workspace(
        launch, case, want, query_lens, "prefill splits on a NaN workspace"
    )
    ret = measure({"prefill_splits": launch}, case, want, query_lens, kv_lens)
    ret["splits run"] = splits or "planned"
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
    assert_same(
        cases["identity"]["out"],
        cases["reversed"]["out"],
        "page permutation invariance",
    )
    ret["K pool GiB"] = cases["identity"]["k"].numel() / 2**30
    return ret


def cache_with_token_stride(dense, token_stride):
    """Same logical cache, with `token_stride` between tokens.

    A 2 GiB poison prefix sits in front of the view. An i32 product that
    sign-extends negative lands there (FNUZ NaN) instead of faulting.
    """
    blocks, page, heads, dim = dense.shape
    guard = 1 << 31
    stride = (token_stride, token_stride, dim, 1)
    span = (blocks - 1) * stride[0] + (page - 1) * stride[1] + heads * dim
    storage = torch.empty(guard + span, device=dense.device, dtype=dense.dtype)
    storage.view(torch.uint8).fill_(0x80)
    view = torch.as_strided(storage, dense.shape, stride, storage_offset=guard)
    for token in range(page):
        view[:, token].copy_(dense[:, token])
    return view


@benchmark()
def test_token_stride_overflow(shape):
    """Decode key 16 with token stride 2**27. The product is 2**31."""
    token_stride = 1 << 27
    assert 16 * token_stride == 1 << 31
    query_lens, kv_lens = [1], [32]
    case = make_case(query_lens, kv_lens, shape, page=32)
    want = reference(case, query_lens, kv_lens)
    dense_k, dense_v = case["k"], case["v"]
    case["k"] = cache_with_token_stride(dense_k, token_stride)
    case["v"] = cache_with_token_stride(dense_v, token_stride)
    assert torch.equal(case["k"], dense_k) and torch.equal(case["v"], dense_v)
    from aiter.ops.flydsl import unified_attention_kernels as adapter

    assert adapter._strides_ok(
        case["q"],
        case["k"],
        case["v"],
        case["out"],
        case["cu_seqlens_q"],
        case["seqused_k"],
        case["block_table"],
    ), "stride 2**27 passes the alignment gate"
    ret = measure(
        {"token_stride": flydsl_candidate(case, kv_lens)},
        case,
        want,
        query_lens,
        kv_lens,
    )
    ret["token stride"] = token_stride
    return ret


def check_window_boundary():
    """The key at pos - window is excluded and pos - window + 1 included, in
    decode and through prefill KV splits."""
    for name, key, expected in (
        ("excluded", 0, 0.0),
        ("included", 1, 1.0),
    ):
        query_lens, kv_lens = [1], [1025]
        case = make_case(query_lens, kv_lens, "sliding")
        case["q"].fill_(4)
        case["k"].zero_()
        case["v"].zero_()
        case["q_descale"].fill_(1)
        case["k_descale"].fill_(1)
        case["v_descale"].fill_(1)
        case["softmax_scale"] = 1.0
        page = case["k"].shape[1]
        physical_page = int(case["block_table"][0, key // page])
        case["k"][physical_page, key % page].fill_(4)
        case["v"][physical_page, key % page].fill_(1)
        want = reference(case, query_lens, kv_lens)
        got = ua.unified_attention(**case, backend="flydsl")
        compare(want, got, query_lens, f"window boundary {name}")
        observed = got.float().mean().item()
        assert abs(observed - expected) < 0.01, (name, observed)

    # Prefill positions 1024..1027 have left edges 1..4. A dominant key at 1
    # must affect only the first query, including through prefill KV splits.
    query_lens, kv_lens = [4], [1028]
    case = make_case(query_lens, kv_lens, "sliding")
    case["q"].fill_(4)
    case["k"].zero_()
    case["v"].zero_()
    case["q_descale"].fill_(1)
    case["k_descale"].fill_(1)
    case["v_descale"].fill_(1)
    case["softmax_scale"] = 1.0
    page = case["k"].shape[1]
    physical_page = int(case["block_table"][0, 1 // page])
    case["k"][physical_page, 1 % page].fill_(4)
    case["v"][physical_page, 1 % page].fill_(1)
    want = reference(case, query_lens, kv_lens)
    got = ua.unified_attention(**case, backend="flydsl")
    compare(want, got, query_lens, "window boundary prefill")
    first = got[0].float().mean().item()
    rest = got[1:].float().abs().max().item()
    assert abs(first - 1.0) < 0.01 and rest < 0.01, (first, rest)
    aiter.logger.info("window boundary checks: passed")


def router_case(
    shape="full", page=PAGE, query_lens=(1, 1), kv_lens=(4096, 4096), **overrides
):
    """A supported call on CPU tensors; only the gate and the cede policy read it."""
    head_dim, num_heads, num_kv_heads, window = SHAPES[shape]
    tokens, blocks, fp8 = sum(query_lens), 4, torch.float8_e4m3fnuz
    case = {
        "q": torch.zeros(tokens, num_heads, head_dim, dtype=fp8),
        "k": torch.zeros(blocks, page, num_kv_heads, head_dim, dtype=fp8),
        "v": torch.zeros(blocks, page, num_kv_heads, head_dim, dtype=fp8),
        "out": torch.zeros(tokens, num_heads, head_dim, dtype=torch.bfloat16),
        "cu_seqlens_q": torch.tensor(
            [0, *itertools.accumulate(query_lens)], dtype=torch.int32
        ),
        "max_seqlen_q": max(query_lens),
        "seqused_k": torch.tensor(kv_lens, dtype=torch.int32),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": 1.0,
        "causal": True,
        "window_size": (-1, -1) if window is None else (window - 1, 0),
        "block_table": torch.zeros(len(query_lens), blocks, dtype=torch.int32),
        "softcap": 0,
        "q_descale": torch.ones(1),
        "k_descale": torch.ones(1),
        "v_descale": torch.ones(1),
    }
    case.update(overrides)
    return case


def check_routing():
    """Backend precedence, the support gate, the cede policy and the AOT
    layouts, with stubbed kernels and CPU tensors: no GPU is needed."""
    from aiter.aot.flydsl.unified_attention import DEFAULT_SHAPES, _fake_kv
    from aiter.ops.flydsl import unified_attention_kernels as adapter
    from aiter.ops.flydsl.kernels.unified_attention_fp8_gfx942 import (
        plan_prefill_splits,
    )

    calls = []

    def stub(name):
        def run(q, k, v, out, *args, **kwargs):
            calls.append(name)
            return out

        return run

    def routed(arch="gfx942", env="auto", **overrides):
        calls.clear()
        with (
            mock.patch.object(adapter, "get_gfx_runtime", return_value=arch),
            mock.patch.object(ua, "_BACKEND_OVERRIDE", env),
        ):
            ua.unified_attention(**router_case(**overrides))
        return calls

    def raises(error, **kwargs):
        try:
            routed(**kwargs)
        except error:
            return True
        return False

    def kv(layout="plain", page=PAGE, pad=0):
        """Full-layer K and V in a cache layout; pad widens each plain row."""
        head_dim, _, num_kv_heads, _ = SHAPES["full"]
        fp8 = torch.float8_e4m3fnuz
        if layout == "plain":
            cache = torch.zeros(4, page, num_kv_heads, head_dim + pad, dtype=fp8)
            return {"k": cache[..., :head_dim], "v": cache[..., :head_dim]}
        cache = torch.zeros(4, num_kv_heads, page, 2 * head_dim, dtype=fp8)
        cache = cache.transpose(1, 2)
        return {"k": cache[..., :head_dim], "v": cache[..., head_dim:]}

    cpu = router_case()
    assert not adapter._devices_ok(cpu["q"], cpu["k"]), "CPU tensors are served"
    triton = SimpleNamespace(unified_attention=stub("triton"))
    fp8 = torch.float8_e4m3fnuz
    misaligned = torch.zeros(2 * 32 * 512 + 4, dtype=torch.bfloat16)[4:]
    declined = {
        "bf16 query": {"q": torch.zeros(2, 32, 512, dtype=torch.bfloat16)},
        "per-token query descale": {"q_descale": torch.ones(2)},
        "fp16 key descale": {"k_descale": torch.ones(1, dtype=torch.float16)},
        "non-unit head-dim stride": {
            "k": torch.zeros(4, PAGE, 4, 1024, dtype=fp8)[..., ::2]
        },
        "shuffled cache": {"shuffled_kv_cache": True},
        "softcap": {"softcap": 30.0},
        "sinks": {"sinks": torch.zeros(32)},
        "8-byte aligned output": {"out": misaligned.view(2, 32, 512)},
        "empty block table": {"block_table": torch.zeros(2, 0, dtype=torch.int32)},
    }
    with (
        mock.patch.dict(
            sys.modules, {"aiter.ops.triton.attention.unified_attention": triton}
        ),
        mock.patch.object(adapter, "_devices_ok", return_value=True),
        mock.patch.object(adapter, "_launch", side_effect=stub("flydsl")),
    ):
        # An explicit backend beats the environment, and only a call that
        # reads an invalid environment value raises: importing does not.
        with mock.patch.dict(os.environ, {"AITER_UNIFIED_ATTENTION_BACKEND": "bogus"}):
            importlib.reload(ua)
        importlib.reload(ua)
        assert routed() == ["flydsl"]
        assert routed(env="triton") == ["triton"]
        assert routed(env="flydsl", backend="triton") == ["triton"]
        assert routed(env="bogus", backend="gluon") == ["triton"]
        assert raises(ValueError, env="bogus")
        assert raises(ValueError, backend="bogus")
        # FlyDSL is built for gfx942 only.
        assert routed(arch="gfx950") == ["triton"]
        assert raises(RuntimeError, arch="gfx950", backend="flydsl")
        for name, overrides in declined.items():
            assert routed(**overrides) == ["triton"], name
            assert raises(RuntimeError, backend="flydsl", **overrides), name
        # Run-only mode serves only the prebuilt layouts; a padded cache
        # (strides 67584, 2112, 528 at page 32) would miss the cache.
        padded = kv(page=32, pad=16)
        assert routed(**padded) == ["flydsl"]
        with run_only_env():
            assert routed(**padded) == ["triton"]
            assert routed(**kv(page=32)) == ["flydsl"]
            assert routed(**kv("vllm", page=32)) == ["flydsl"]
        # One boundary per cede rule: tiny head-512 decode, page-32 sliding
        # decode past 38 sequences, a short fresh prefill, and a mixed
        # batch's short chunk.
        assert routed(kv_lens=(1024, 1024)) == ["triton"]
        sliding = {"shape": "sliding", "page": 32}
        assert routed(**sliding, query_lens=(1,) * 38, kv_lens=(1024,) * 38) == [
            "flydsl"
        ]
        assert routed(**sliding, query_lens=(1,) * 39, kv_lens=(1024,) * 39) == [
            "triton"
        ]
        assert routed(query_lens=(256,), kv_lens=(256,)) == ["triton"]
        assert routed(query_lens=(384,), kv_lens=(384,)) == ["flydsl"]
        assert routed(query_lens=(128, 1), kv_lens=(16384, 4096)) == ["triton"]
        assert routed(query_lens=(256, 1), kv_lens=(16384, 4096)) == ["flydsl"]
    # A fresh prefill never splits its KV range; a short chunk over a long
    # prefix does.
    assert plan_prefill_splits(4096, 4096, 4096, 32, 4, None, 512, 304) == 1
    assert plan_prefill_splits(16, 4096, 16, 32, 4, None, 512, 304) > 1
    # The gate's prebuilt strides are the layouts AOT compiles.
    for entries in DEFAULT_SHAPES.values():
        for _, num_kv_heads, head_dim, _ in entries:
            for page in adapter._PAGE_SIZES:
                for layout in adapter.KV_LAYOUTS:
                    k, v = _fake_kv(layout, page, num_kv_heads, head_dim)
                    want = adapter.kv_layout_strides(
                        layout, page, num_kv_heads, head_dim
                    )
                    assert k.stride()[:3] == v.stride()[:3] == want, (layout, page)
    aiter.logger.info("routing and support-gate checks: passed")


def main():
    check_routing()
    arch = get_gfx_runtime()
    if get_gfx() != "gfx942" or arch != "gfx942":
        aiter.logger.warning(
            "FlyDSL unified attention requires gfx942; skipping kernel cases "
            "(build=%s, attached=%s)",
            get_gfx(),
            arch,
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FP8 unified attention correctness and warm-buffer timing sweep",
    )
    parser.add_argument(
        "--shape", choices=list(SHAPES), nargs="+", default=list(SHAPES)
    )
    parser.add_argument(
        "--page", type=int, choices=[32, 64], nargs="+", default=[32, 64]
    )
    parser.add_argument(
        "--layout", choices=["plain", "vllm"], nargs="+", default=["plain", "vllm"]
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 5, 16, 64])
    parser.add_argument("--boundary-gib", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    sweeps = [
        (
            "TP1 AOT matrix, run-only",
            test_aot_matrix,
            itertools.product(
                args.shape, args.page, args.layout, ["prefill", "decode"]
            ),
        ),
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
            "no visible key",
            test_no_visible_key,
            itertools.product(args.shape, ["prefill", "decode"]),
        ),
        (
            "overestimated max query length",
            test_overestimated_max_seqlen_q,
            itertools.product(args.shape),
        ),
        (
            "split-K combine",
            test_splitk_combine,
            itertools.product(args.shape, args.splits),
        ),
        (
            "prefill KV splits",
            test_prefill_splits,
            itertools.product(
                args.shape,
                args.page,
                [(16, 1500), (37, 4096), (300, 3000), (320, 1024)],
                [None, 3, 8],
            ),
        ),
        (
            "paged addressing",
            test_paged_addressing,
            itertools.product(args.shape, args.boundary_gib, args.layout),
        ),
        (
            "token stride overflow",
            test_token_stride_overflow,
            itertools.product(args.shape),
        ),
    ]
    for name, fn, parameters in sweeps:
        rows = [fn(*values) for values in parameters]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "%s summary (markdown):\n%s",
            name,
            df.to_markdown(index=False, missingval=""),
        )
    check_window_boundary()
    aiter.logger.info(
        "PASS: all unified-attention test groups; all candidate timings non-zero"
    )


if __name__ == "__main__":
    main()
