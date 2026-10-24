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

PAGE, H, HKV, D = 64, 64, 4, 128
FP8_REL_ERR = {dtypes.bf16: 7.7e-2, dtypes.fp16: 7.7e-2}


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype = torch.float32,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    sinks: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    causal: int = 1,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    v_head_size = value_cache.shape[-1]
    outputs: list[torch.Tensor] = []
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
        q = q * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_size)[:kv_len]
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


def shuffle_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """
    dtype = key_cache.dtype
    assert value_cache.dtype == dtype
    assert dtype in (torch.bfloat16, dtypes.fp8)

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    k_width = 16 // key_cache.element_size()
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 3, 1)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        head_size // k_width,
        k_width,
        block_size,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // k_width,
        k_width,
        head_size,
    )
    value_cache_shuffled = value_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    return key_cache_shuffled, value_cache_shuffled


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


def compare(want, got, atol, name):
    want, got = want.float(), got.float()
    err = checkAllclose(want, got, rtol=0, atol=atol, tol_err_ratio=0, msg=name)
    assert err == 0, f"{name}: {err:.3%} elements exceed atol={atol}"
    return err


def measure(candidates, case, want, query_lens, kv_lens, atol=None):
    # Output-dtype bound on the global max rather than an element-relative rtol.
    if atol is None:
        atol = FP8_REL_ERR[case["out"].dtype] * want.abs().max().item()
    heads = case["q"].shape[1]
    # Bottom-right-aligned causal: query i sees kv_len - q_len + i + 1 keys.
    pairs = sum(
        (
            sum(max(0, min(k, k - q + i + 1)) for i in range(q))
            if case["causal"]
            else q * k
        )
        for q, k in zip(query_lens, kv_lens)
    )
    flops = 4 * heads * D * pairs
    nbytes = sum(query_lens) * heads * D * (1 + case["out"].element_size())
    nbytes += 2 * sum(kv_lens) * HKV * D
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        got = fn()
        err = compare(want, got, atol, name)
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
    # Uniform short prefill exercises graph capture without host synchronization.
    "small_sq": ([64, 64], [2048, 2048], 16),
    # Ragged pure prefill: M=970 > (B-1)*S+1=961, so no row can have query length 1.
    "varlen": ([320, 300, 200, 150], [2048, 5000, 8192, 3000], 16),
    # Ragged causal prefill with partial KV pages.
    "varlen_causal": ([93, 80, 71, 60], [157, 100, 90, 70], 4),
    # Long group-2 paged prefill: 4096 KV tokens = 64 tiles > 48 selects the BM256
    # tile; group 2 is not a packed-BN64 group. Single sequence also passes the
    # causal no-decode proof, which the ragged rows cannot.
    "long_g2": ([256], [4096], 2),
    **{f"ragged_g{g}": ([93, 17, 41, 9], [157, 0, 70, 1], g) for g in (1, 2, 4, 8, 16)},
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
        case["k"], case["v"] = shuffle_kv_cache(case["k"], case["v"])
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
        compare(
            want,
            case["out"],
            FP8_REL_ERR[case["out"].dtype] * want.abs().max().item(),
            "small-Sq graph",
        )
    return ret


_LAYER_GRAPH_CHECKED = set()


@benchmark()
def test_decode(dtype, depth, layout):
    query_lens = [1] * 8
    # Only the FlyDSL layout defines empty-KV output (zero); the Triton fallback gives NaN.
    empty_kv = 0 if layout == "vectorized" else 1
    kv_lens = [depth, max(1, depth // 2 + 3), depth, 65, depth, 1, depth, empty_kv]
    case = make_case(query_lens, kv_lens, dtype)
    want = reference(case, query_lens, kv_lens)
    if layout == "vectorized":
        case["k"], case["v"] = shuffle_kv_cache(case["k"], case["v"])
    case["shuffled_kv_cache"] = layout == "vectorized"
    backend = "flydsl" if layout == "vectorized" else None
    name = "flydsl" if layout == "vectorized" else "triton_fallback"
    candidates = {name: partial(ua.unified_attention, **case, backend=backend)}
    ret = measure(candidates, case, want, query_lens, kv_lens)
    if layout == "vectorized":
        compare(torch.zeros_like(want[-1]), case["out"][-1], 0, "empty KV")
        # Layer-graph coverage depends on dtype only, so run it once per dtype.
        if dtype not in _LAYER_GRAPH_CHECKED:
            _LAYER_GRAPH_CHECKED.add(dtype)
            check_decode_layer_graphs(dtype)
            check_decode_workspace_growth(dtype)
    if layout == "linear":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            candidates[name]()
        case["out"].fill_(float("nan"))
        graph.replay()
        compare(
            want,
            case["out"],
            FP8_REL_ERR[case["out"].dtype] * want.abs().max().item(),
            "decode graph",
        )
    return ret


def check_decode_layer_graphs(dtype):
    query_lens, capacity_lens = [1] * 3, [1024] * 3
    layers = [
        make_case(query_lens, capacity_lens, dtype, seed=seed) for seed in (7, 19)
    ]
    lengths = layers[0]["seqused_k"]
    table = layers[0]["block_table"]
    calls, linear_layers = [], []
    for layer in layers:
        layer["seqused_k"], layer["block_table"] = lengths, table
        linear_layers.append(dict(layer))
        layer["k"], layer["v"] = shuffle_kv_cache(layer["k"], layer["v"])
        layer["shuffled_kv_cache"] = True
        calls.append(partial(ua.unified_attention, **layer, backend="flydsl"))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for call in calls:
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for call in calls:
            call()

    def replay_and_check(graph, cases, context_lengths, label):
        for case in cases:
            case["out"].fill_(float("nan"))
        graph.replay()
        for i, case in enumerate(cases):
            want = reference(case, query_lens, context_lengths)
            compare(
                want,
                case["out"],
                FP8_REL_ERR[case["out"].dtype] * want.abs().max().item(),
                f"{label} layer {i}",
            )

    for context_lengths in ([65, 513, 0], [1024, 1, 257]):
        lengths.copy_(torch.tensor(context_lengths, device="cuda", dtype=torch.int32))
        table.copy_(table.roll(1, dims=1))
        replay_and_check(graph, linear_layers, context_lengths, "shared decode graph")

    # Each captured call's planner refresh must observe lengths changed between captures.
    second_lengths = torch.tensor([769, 129, 3], device="cuda", dtype=torch.int32)
    second_cases = [dict(case, seqused_k=second_lengths) for case in linear_layers]
    second_calls = [
        partial(
            ua.unified_attention,
            **dict(layer, seqused_k=second_lengths),
            backend="flydsl",
        )
        for layer in layers
    ]
    second_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(second_graph, stream=stream):
        for call in second_calls:
            call()
    replay_and_check(second_graph, second_cases, [769, 129, 3], "second decode graph")
    second_lengths.copy_(torch.tensor([0, 1024, 65], device="cuda", dtype=torch.int32))
    table.copy_(table.roll(2, dims=1))
    replay_and_check(second_graph, second_cases, [0, 1024, 65], "second decode replay")
    replay_and_check(graph, linear_layers, [1024, 1, 257], "first graph after second")

    second_graph.reset()
    with torch.cuda.graph(second_graph, stream=stream):
        for call in second_calls:
            call()
    replay_and_check(
        second_graph, second_cases, [0, 1024, 65], "recaptured decode graph"
    )


def check_decode_workspace_growth(dtype):
    from aiter.ops.flydsl import unified_attention_kernels as kernels

    buffers = ("work_info", "reduce_info", "exp_sums", "max_logits", "temporary_output")

    def pointers(workspace):
        return tuple(workspace[name].data_ptr() for name in buffers)

    def prepare(batch, depth=64):
        lengths = [depth] * batch
        linear = make_case([1] * batch, lengths, dtype)
        case = dict(linear)
        case["k"], case["v"] = shuffle_kv_cache(case["k"], case["v"])
        case["shuffled_kv_cache"] = True
        call = partial(ua.unified_attention, **case, backend="flydsl")
        return linear, call

    def check(case, label):
        lengths = case["seqused_k"].tolist()
        want = reference(case, [1] * len(lengths), lengths)
        compare(
            want,
            case["out"],
            FP8_REL_ERR[dtype] * want.abs().max().item(),
            label,
        )

    def key_for(stream):
        return (torch.cuda.current_device(), stream.cuda_stream, HKV, H, D, dtype)

    eager_stream = torch.cuda.Stream()
    eager_stream.wait_stream(torch.cuda.current_stream())
    key = key_for(eager_stream)
    assert key not in kernels._PA_DECODE_WORKSPACES
    with torch.cuda.stream(eager_stream):
        high_water = 0
        # Cross each power-of-two boundary, then revisit smaller logical views.
        for batch in list(range(1, 34, 2)) + list(range(32, 0, -2)):
            case, call = prepare(batch)
            call()
            high_water = max(high_water, batch)
            state = kernels._PA_DECODE_WORKSPACES[key]
            assert sum(k == key for k in kernels._PA_DECODE_WORKSPACES) == 1
            assert state["current"]["captured"] is False
            assert state["pinned"] == []
            assert state["current"]["capacity"] == 1 << (high_water - 1).bit_length()
            check(case, f"eager workspace batch {batch}")
        workspace = state["current"]
        for depth in (513, 1024):
            case, call = prepare(3, depth)
            assert (
                kernels._pa_decode_num_partitions(
                    3, HKV, depth, kernels._num_cu(case["q"].device)
                )
                > 1
            )
            call()
            assert state["current"] is workspace and state["pinned"] == []
            check(case, f"partitioned workspace depth {depth}")
    eager_stream.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    key = key_for(stream)
    assert key not in kernels._PA_DECODE_WORKSPACES
    with torch.cuda.stream(stream):
        small, small_call = prepare(3)
        small_call()
        state = kernels._PA_DECODE_WORKSPACES[key]
        first = state["current"]
        assert first["capacity"] == 4
        first_pointers = pointers(first)
        small_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(small_graph, stream=stream):
            small_call()
        assert first["captured"] is True and state["current"] is first

        medium, medium_call = prepare(17)
        medium_call()
        shared = state["current"]
        assert shared["capacity"] == 32
        assert len(state["pinned"]) == 1 and state["pinned"][0] is first
        assert pointers(first) == first_pointers
        small["seqused_k"].copy_(
            torch.tensor([0, 1, 63], device="cuda", dtype=torch.int32)
        )
        small_graph.replay()
        check(small, "small graph after growth")

        middle, middle_call = prepare(5)
        middle_call()
        medium_call()
        shared_pointers = pointers(shared)
        graphs = [(small_graph, small)]
        for case, call in ((middle, middle_call), (medium, medium_call)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
            graphs.append((graph, case))
            assert state["current"] is shared and pointers(shared) == shared_pointers
            assert shared["captured"] is True and len(state["pinned"]) == 1
        for iteration in range(2):
            for graph, case in reversed(graphs):
                batch = case["q"].shape[0]
                lengths = [(i * 17 + iteration * 7) % 65 for i in range(batch)]
                case["seqused_k"].copy_(
                    torch.tensor(lengths, device="cuda", dtype=torch.int32)
                )
                case["out"].fill_(float("nan"))
                graph.replay()
                check(case, f"shared workspace replay {batch}/{iteration}")

        large, large_call = prepare(33)
        large_call()
        current = state["current"]
        assert current["capacity"] == 64 and current["captured"] is False
        assert len(state["pinned"]) == 2
        assert state["pinned"][0] is first and state["pinned"][1] is shared
        assert pointers(first) == first_pointers and pointers(shared) == shared_pointers
        check(large, "eager growth to 64")
        for graph, case in graphs:
            case["seqused_k"].zero_()
            case["seqused_k"][-1] = 64
            case["out"].fill_(float("nan"))
            graph.replay()
            check(case, "older graph after second growth")

        # Explicit FlyDSL declines before any planner or workspace allocation.
        _cold, cold_call = prepare(65)
        current_pointers = pointers(current)
        cold_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cold_graph, stream=stream):
            try:
                cold_call()
            except RuntimeError as error:
                assert "does not support this configuration" in str(error)
            else:
                raise AssertionError("insufficient captured workspace did not decline")
        assert state["current"] is current and pointers(current) == current_pointers
        assert current["capacity"] == 64 and current["captured"] is False
        assert len(state["pinned"]) == 2
        assert key in kernels._PA_DECODE_COLD_CAPTURE_WARNED
    stream.synchronize()
    aiter.logger.info("PASS: decode workspace growth and retained graphs (%s)", dtype)


@benchmark()
def test_paged_addressing(pool_blocks, layout):
    # Shuffling temporarily holds both linear pools and both shuffled pools.
    required_bytes = (4 if layout == "vectorized" else 2) * pool_blocks * PAGE * HKV * D
    free_bytes, _ = torch.cuda.mem_get_info()
    if required_bytes + 256 * 1024**2 > free_bytes:
        aiter.logger.warning(
            "paged addressing skipped: pool_blocks=%s layout=%s requires %s bytes plus 256 MiB; free=%s",
            pool_blocks,
            layout,
            required_bytes,
            free_bytes,
        )
        return {"gfx": get_gfx_runtime(), "skipped": "insufficient device memory"}
    query_lens, kv_lens = [128], [256]
    original = make_case(query_lens, kv_lens, dtypes.bf16)
    want = reference(original, query_lens, kv_lens)
    count = original["k"].shape[0]
    case = dict(original, out=torch.empty_like(original["out"]))
    ids = [pool_blocks - 1 - i for i in range(count)]
    for key in ("k", "v"):
        pool = torch.zeros(
            pool_blocks, PAGE, HKV, D, device="cuda", dtype=original[key].dtype
        )
        for logical, pid in enumerate(ids):
            pool[pid] = original[key][int(original["block_table"][0, logical])]
        case[key] = pool
    case["block_table"] = torch.tensor([ids], device="cuda", dtype=torch.int32)
    if layout == "vectorized":
        case["k"], case["v"] = shuffle_kv_cache(case["k"], case["v"])
    case["shuffled_kv_cache"] = layout == "vectorized"
    candidates = {"reversed": partial(ua.unified_attention, **case, backend="flydsl")}
    ret = measure(candidates, case, want, query_lens, kv_lens, atol=0.055)
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

    if not is_flydsl_available(torch.cuda.current_device()):
        aiter.logger.warning(
            "FlyDSL unified attention requires FlyDSL on gfx950; skipping"
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
                # These ragged query lengths do not satisfy the host-only no-decode
                # proof; exercise them noncausally on FlyDSL.
                if not (c and k.startswith("ragged"))
            ],
        ),
        (
            "decode",
            test_decode,
            itertools.product(args.dtype, args.decode_depth, args.layout),
        ),
        # The 65537-page pool puts page 65536 at byte offset 2**31.
        (
            "paged addressing",
            test_paged_addressing,
            itertools.product([65535, 65537], args.layout),
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
