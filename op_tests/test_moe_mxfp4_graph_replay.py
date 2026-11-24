# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP-graph replay safety of the MXFP4 (W4A4) MoE scratch buffers (ROCm/aiter#3632).

A captured graph replays on the same private-pool addresses and the pool is not
re-zeroed between replays, so every byte of a scratch buffer that a consumer
reads must be defined by the captured work itself. These checks capture the op,
overwrite everything the graph allocated with a poison pattern between replays,
and require the replay to be defined:

  * fused quant+sort (activation fp4 data and the swizzled E8M0 scale),
  * the sort's token-id and expert-id tails,
  * the whole ``fused_moe`` A4W4 call.

On a mismatch the failing tensor is named, which is the information needed to
pin the buffer responsible (sort workspace, atomic output or quant scale).

Run on gfx950:
    python3 op_tests/test_moe_mxfp4_graph_replay.py
    pytest op_tests/test_moe_mxfp4_graph_replay.py -q
"""

import ctypes

import pytest
import torch

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.jit.utils.hip_runtime import load_hip_runtime
from aiter.ops.quant import (
    fused_dynamic_mx_quant_moe_sort,
    fused_dynamic_mx_quant_moe_sort_hip,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

pytestmark = pytest.mark.skipif(
    get_gfx_runtime() != "gfx950", reason="MXFP4 W4A4 MoE requires gfx950"
)

BLOCK_M = 32


def _poison(pool_tensors):
    for t in pool_tensors:
        if t.numel():
            t.view(torch.uint8).fill_(0xFF)


def _poison_graph_pool(graph):
    """Overwrite every byte of the graph's private pool, including freed blocks.

    Replay does not clear the pool. A scratch buffer that died at the end of
    capture still has its writes in the graph, and a later capture can place a
    live tensor at that address. Filling only the output misses that scratch.
    """
    hip = load_hip_runtime()
    pool = graph.pool()
    for seg in torch.cuda.memory._snapshot()["segments"]:
        if seg.get("segment_pool_id") != pool:
            continue
        err = hip.hipMemset(
            ctypes.c_void_p(seg["address"]),
            0xFF,
            ctypes.c_size_t(seg["total_size"]),
        )
        if err != 0:
            raise RuntimeError(f"hipMemset of graph pool failed with error {err}")
    torch.cuda.synchronize()


def _zeros_like(t):
    """Zero-filled tensor of ``t``'s shape and dtype.

    Packed fp4 / e8m0 have no fill kernel, so zero the raw bytes and view them
    back. A byte the quant kernel does not write stays 0, which is what a
    captured replay records with its memset.
    """
    return torch.zeros(t.shape, dtype=torch.uint8, device=t.device).view(t.dtype)


def _capture(fn):
    """Warm up, capture ``fn`` and return (graph, static outputs)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


@pytest.mark.parametrize("token_num", [1, 4, 33])
def test_quant_sort_replay(token_num, model_dim=1024, num_experts=16, topk=4):
    torch.manual_seed(0)
    x = torch.randn((token_num, model_dim), dtype=dtypes.bf16, device="cuda")
    score = torch.randn((token_num, num_experts), dtype=dtypes.bf16, device="cuda")
    topk_w, topk_ids = fused_topk(x, score, topk, True)

    def run():
        sorted_ids, sorted_w, _, num_valid, _ = moe_sorting(
            topk_ids, topk_w, num_experts, model_dim, dtypes.bf16, BLOCK_M
        )
        quant, scale = fused_dynamic_mx_quant_moe_sort(
            x,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid,
            token_num=token_num,
            topk=topk,
            block_size=BLOCK_M,
            sorted_weights=sorted_w,
        )
        return quant, scale, sorted_ids, sorted_w, num_valid

    graph, (q, s, sorted_ids, sorted_w, num_valid) = _capture(run)
    for i in range(3):
        _poison([q, s])
        graph.replay()
        torch.cuda.synchronize()
        # The kernel does not cover every allocated scale byte (swizzle holes
        # and the tile padding). Eager ``torch.empty`` leaves those as allocator
        # leftovers, so a whole-buffer compare with eager is not a defined
        # result. Replay records a memset of the whole buffer and then the
        # kernel; the reference is that same kernel launched into zeroed
        # storage, using the routes this replay produced.
        ref_q, ref_s = _zeros_like(q), _zeros_like(s)
        fused_dynamic_mx_quant_moe_sort_hip(
            ref_q,
            ref_s,
            x,
            sorted_ids,
            num_valid,
            token_num,
            BLOCK_M,
            token_num * topk,
            32,
            sorted_w,
        )
        torch.testing.assert_close(
            s.view(torch.uint8),
            ref_s.view(torch.uint8),
            rtol=0,
            atol=0,
            msg=lambda m, i=i: f"replay {i}: scale diverged\n{m}",
        )
        torch.testing.assert_close(
            q.view(torch.uint8),
            ref_q.view(torch.uint8),
            rtol=0,
            atol=0,
            msg=lambda m, i=i: f"replay {i}: quantized activation diverged\n{m}",
        )


def test_sort_tail_defined_on_replay():
    """Routes past the occupied prefix must not survive into the next replay.

    The sort writes only the occupied prefix. Poison the captured buffers and
    require the tail to come back as the padding sentinels (token id == M,
    expert id == -1), not as the poison.
    """
    M, E, topk, model = 4, 32, 4, 128
    torch.manual_seed(0)
    ids = torch.randint(0, E, (M, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(M, topk, device="cuda")
    moe_sorting(ids, weights, E, model, torch.bfloat16, BLOCK_M)
    torch.cuda.synchronize()

    def run():
        return moe_sorting(ids, weights, E, model, torch.bfloat16, BLOCK_M)

    graph, (sorted_ids, _sorted_w, sorted_experts, num_valid, _buf) = _capture(run)
    poison = 0x5A5A5A5A
    sorted_ids.fill_(poison)
    sorted_experts.fill_(poison)
    graph.replay()
    torch.cuda.synchronize()
    occupied = int(num_valid[0])
    assert torch.all(
        sorted_ids[occupied:] == M
    ), "sorted token-id tail kept the previous replay's bytes"
    blocks = (occupied + BLOCK_M - 1) // BLOCK_M
    assert torch.all(
        sorted_experts[blocks:] == -1
    ), "sorted expert-id tail kept the previous replay's bytes"


@pytest.mark.parametrize("token_num", [1, 4, 33])
def test_fused_moe_a4w4_replay(
    token_num, model_dim=1024, inter_dim=512, num_experts=16, topk=4
):
    dtype = dtypes.bf16
    torch.manual_seed(0)
    x = torch.randn((token_num, model_dim), dtype=dtype, device="cuda")
    w1 = torch.randn(
        (num_experts, inter_dim * 2, model_dim), dtype=dtype, device="cuda"
    )
    w2 = torch.randn((num_experts, model_dim, inter_dim), dtype=dtype, device="cuda")
    score = torch.randn((token_num, num_experts), dtype=dtype, device="cuda")
    topk_w, topk_ids = fused_topk(x, score, topk, True)

    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1_qt, w1_s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2_qt, w2_s = tq(w2, quant_dtype=dtypes.fp4x2)
    w1_qt = shuffle_weight(
        w1_qt.view(num_experts, inter_dim * 2, model_dim // 2), (16, 16)
    )
    w2_qt = shuffle_weight(w2_qt.view(num_experts, model_dim, inter_dim // 2), (16, 16))
    w1_s = e8m0_shuffle(w1_s)
    w2_s = e8m0_shuffle(w2_s)

    def run():
        return fused_moe(
            x,
            w1_qt,
            w2_qt,
            topk_w,
            topk_ids,
            w1_scale=w1_s,
            w2_scale=w2_s,
            quant_type=QuantType.per_1x32,
        )

    eager = run().float().clone()
    # Eager itself is not bitwise repeatable (atomic accumulation order, bf16
    # rounding): measured eager-vs-eager spread is 0.4-0.8% of the output max.
    # Tolerate 2% of the max; a corrupted replay is off by O(max).
    tol = 0.02 * float(eager.abs().max())
    graph, out = _capture(run)
    for i in range(3):
        _poison_graph_pool(graph)
        graph.replay()
        torch.cuda.synchronize()
        err = float((out.float() - eager).abs().max())
        assert err <= tol, (
            f"replay {i}: fused_moe output diverged from eager: "
            f"max_abs={err:.1f} > tol={tol:.1f} (output max {tol / 0.02:.1f})"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
