# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance coverage for the FlyDSL QSA sparse decode (gfx950).

Qwen sparse attention (QSA) decode as SGLang's ``QwenSparseAttnBackend`` calls it:
every query row attends to its own top-k token positions, read straight from the
KV pool through ``req_to_token``.

    q              [rows, Hq, 256]          bf16, contiguous
    k/v pool       [num_slots, Hkv, 256]    bf16 or fp8 e4m3fn
    req_to_token   [num_reqs, max_ctx]      int32
    row_req        [rows]                   int64 (the batch's req_pool_indices)
    indices        [rows, topk]             int32 logical positions, -1 padded
    seq_lens       [rows]                   int32
    out            [rows, Hq, 256]          bf16

The backend sizes one workspace for every captured batch with
``workspace_partials`` and passes ``recommended_num_splits`` per call; both are
reproduced here. Production shapes are Qwen3.8 topk 2051 with (Hq, Hkv) = (24, 2)
at TP1 and (3, 1) at TP8.

Run:
    python op_tests/test_flydsl_qsa_sparse_decode.py
    python op_tests/test_flydsl_qsa_sparse_decode.py -b 1 64 -n 24,2 -d bf16
    pytest op_tests/test_flydsl_qsa_sparse_decode.py -q
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.qsa_sparse_decode import (
    can_use_flydsl_qsa_sparse_decode,
    flydsl_qsa_sparse_decode,
    recommended_num_splits,
    workspace_partials,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# CDNA4 MFMA shapes and ds_read_b64_tr_b16 only.
SUPPORTED_GFX = ["gfx950"]
pytestmark = pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX, reason="FlyDSL QSA sparse decode requires gfx950"
)

HEAD_DIM = 256
# Qwen3.8-Flash-Next: 512 compressed blocks x ratio 4 + 3.
TOPK = 2051
MAX_CTX = 8192
NUM_SLOTS = 40000
ATOL = 1e-2
RTOL = 1e-2


def make_inputs(
    rows,
    num_q_heads,
    num_kv_heads,
    topk=TOPK,
    kv_dtype=dtypes.bf16,
    num_slots=NUM_SLOTS,
    mixed=False,
    seed=0,
):
    """Pool, request map and per-row selections for one decode batch.

    By default every row selects ``topk`` distinct positions of a longer context,
    as in long-context decode. ``mixed`` adds every masking case the kernel must
    honor: valid prefixes of varying length, ``-1`` padding, an entry at
    ``seq_len`` (masked although not ``-1``), and an empty row 0. Slot 0 is never
    referenced by a valid token and holds NaN, the way CUDA-graph padding leaves
    it, so any masked lane that reads it must not leak into the output.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(rows, num_q_heads, HEAD_DIM, generator=gen)
    k = torch.randn(num_slots, num_kv_heads, HEAD_DIM, generator=gen)
    v = torch.randn(num_slots, num_kv_heads, HEAD_DIM, generator=gen)
    k[0] = float("nan")
    v[0] = float("nan")
    req_to_token = torch.randint(
        1, num_slots, (rows + 4, MAX_CTX), generator=gen, dtype=dtypes.i32
    )
    row_req = torch.randperm(rows + 4, generator=gen)[:rows]
    low = topk // 2 if mixed else topk
    seq_lens = torch.randint(low, MAX_CTX, (rows,), generator=gen, dtype=dtypes.i32)
    indices = torch.full((rows, topk), -1, dtype=dtypes.i32)
    for r in range(rows):
        seq_len = int(seq_lens[r])
        count = topk
        if mixed:
            count = 0 if (r == 0 and rows > 1) else (r * 977) % topk + 1
        count = min(count, seq_len)
        picked = torch.randperm(seq_len, generator=gen)[:count]
        indices[r, :count] = picked.sort().values.to(dtypes.i32)
        if count < topk:
            indices[r, count] = seq_len
    return (
        q.to(dtypes.bf16),
        k.to(kv_dtype),
        v.to(kv_dtype),
        req_to_token,
        row_req,
        indices,
        seq_lens,
    )


def make_workspace(max_rows, num_q_heads, num_kv_heads, topk=TOPK):
    """One workspace for every batch up to ``max_rows``, as the backend sizes it."""
    partials = workspace_partials(
        max_rows, num_q_heads, num_kv_heads, topk, torch.device("cuda")
    )
    return (
        torch.empty(partials, dtype=dtypes.fp32),
        torch.empty(partials, dtype=dtypes.fp32),
        torch.empty(partials * HEAD_DIM, dtype=dtypes.fp32),
    )


def valid_mask(indices, seq_lens):
    return (indices >= 0) & (indices < seq_lens[:, None].long())


def run_torch(q, k, v, req_to_token, row_req, indices, seq_lens, scale):
    """Reference: fp32 softmax over each row's valid selections; empty rows are 0."""
    group = q.shape[1] // k.shape[1]
    valid = valid_mask(indices, seq_lens)
    positions = torch.where(valid, indices, 0).long()
    slots = req_to_token[row_req.long()[:, None], positions].long()
    k_sel = k[slots].to(dtypes.fp32).repeat_interleave(group, dim=2)
    v_sel = v[slots].to(dtypes.fp32).repeat_interleave(group, dim=2)
    k_sel = torch.where(valid[:, :, None, None], k_sel, 0.0)
    v_sel = torch.where(valid[:, :, None, None], v_sel, 0.0)
    scores = torch.einsum("rhd,rthd->rht", q.to(dtypes.fp32), k_sel) * scale
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("rht,rthd->rhd", probs, v_sel)


def check(inputs, out, scale, msg):
    return checkAllclose(
        run_torch(*inputs, scale),
        out.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg=msg,
    )


@benchmark()
def test_qsa_sparse_decode(rows, num_q_heads, num_kv_heads, topk, kv_dtype, splits):
    """``splits`` 0 takes ``recommended_num_splits``, as the backend does."""
    inputs = make_inputs(rows, num_q_heads, num_kv_heads, topk, kv_dtype)
    q, k, _, req_to_token, _, indices, seq_lens = inputs
    scale = HEAD_DIM**-0.5
    num_splits = splits or recommended_num_splits(rows, num_kv_heads, topk, q.device)
    workspace = make_workspace(rows, num_q_heads, num_kv_heads, topk)

    candidates = {"flydsl": flydsl_qsa_sparse_decode}

    num_valid = int(valid_mask(indices, seq_lens).sum())
    # QK and PV: 2 * D multiply-adds each per (valid token, query head).
    flops = 4 * HEAD_DIM * num_q_heads * num_valid
    nbytes = (
        2 * num_valid * num_kv_heads * HEAD_DIM * k.element_size()  # K + V
        + 2 * q.numel() * q.element_size()  # q + out
        + indices.numel() * indices.element_size()
        + num_valid * req_to_token.element_size()  # slot lookups
    )

    ret = {"gfx": get_gfx(), "num_splits": num_splits}
    for name, fn in candidates.items():
        # Positional tensors so run_perftest rotates the pool: selected KV arrives
        # cold from HBM, as it does across a model's layers.
        out, us = run_perftest(
            fn, *inputs, scale, num_splits=num_splits, workspace=workspace
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = check(
            inputs,
            out,
            scale,
            msg=f"{name}: rows={rows} heads={num_q_heads},{num_kv_heads} {kv_dtype}",
        )
    return ret


test_qsa_sparse_decode.__test__ = False


@pytest.mark.parametrize(
    "rows, num_q_heads, num_kv_heads, kv_dtype, num_splits",
    [
        (1, 24, 2, dtypes.bf16, None),  # TP1
        (7, 3, 1, dtypes.bf16, None),  # TP8
        (64, 24, 2, dtypes.bf16, None),
        (129, 3, 1, dtypes.bf16, 1),  # single split writes the output directly
        (5, 24, 2, dtypes.fp8, None),
        (3, 3, 1, dtypes.fp8, 33),  # one wave round per split
        (4, 16, 1, dtypes.bf16, None),  # largest supported GQA group
    ],
)
def test_masking(rows, num_q_heads, num_kv_heads, kv_dtype, num_splits):
    inputs = make_inputs(rows, num_q_heads, num_kv_heads, kv_dtype=kv_dtype, mixed=True)
    scale = HEAD_DIM**-0.5
    if num_splits is None:
        num_splits = recommended_num_splits(rows, num_kv_heads, TOPK, inputs[0].device)
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=num_splits)
    assert check(inputs, out, scale, msg="masking") == 0
    if rows > 1:
        assert torch.count_nonzero(out[0]) == 0, "an empty row must produce zeros"


@pytest.mark.parametrize("topk", [1, 64, 65, 512])
def test_small_topk(topk):
    inputs = make_inputs(6, 24, 2, topk=topk, mixed=True)
    scale = HEAD_DIM**-0.5
    num_splits = recommended_num_splits(6, 2, topk, inputs[0].device)
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=num_splits)
    assert check(inputs, out, scale, msg=f"topk={topk}") == 0


def test_int32_index_tensors():
    # The kernel is specialized on the request-map and length dtypes.
    q, k, v, req_to_token, row_req, indices, seq_lens = make_inputs(5, 24, 2)
    inputs = (q, k, v, req_to_token, row_req.int(), indices, seq_lens.long())
    scale = HEAD_DIM**-0.5
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=4)
    assert check(inputs, out, scale, msg="int32 row_req, int64 seq_lens") == 0


def test_slot_offsets_beyond_int32():
    """Slot * Hkv * D overflows int32 once a pool holds > 4M tokens at Hkv=2."""
    rows, num_kv_heads = 2, 2
    num_slots = (1 << 31) // (num_kv_heads * HEAD_DIM) + 4096
    q, _, _, req_to_token, row_req, indices, seq_lens = make_inputs(
        rows, 24, num_kv_heads, num_slots=64
    )
    k = torch.zeros(num_slots, num_kv_heads, HEAD_DIM, dtype=dtypes.fp8)
    v = torch.zeros_like(k)
    req_to_token = req_to_token + (num_slots - 64)
    gen = torch.Generator(device="cuda").manual_seed(1)
    tail = (64, num_kv_heads, HEAD_DIM)
    k[-64:] = torch.randn(tail, generator=gen).to(k.dtype)
    v[-64:] = torch.randn(tail, generator=gen).to(v.dtype)
    inputs = (q, k, v, req_to_token, row_req, indices, seq_lens)
    scale = HEAD_DIM**-0.5
    out = flydsl_qsa_sparse_decode(*inputs, scale, num_splits=4)
    assert check(inputs, out, scale, msg="pool offsets beyond int32") == 0


def test_cuda_graph_replay_tracks_new_inputs():
    """A captured launch must re-read indices, lengths and the request map."""
    rows, num_q_heads, num_kv_heads = 8, 24, 2
    first = make_inputs(rows, num_q_heads, num_kv_heads, mixed=True, seed=0)
    second = make_inputs(rows, num_q_heads, num_kv_heads, mixed=True, seed=1)
    live = tuple(t.clone() for t in first)
    scale = HEAD_DIM**-0.5
    num_splits = recommended_num_splits(rows, num_kv_heads, TOPK, live[0].device)
    workspace = make_workspace(rows, num_q_heads, num_kv_heads)

    def run():
        return flydsl_qsa_sparse_decode(
            *live, scale, num_splits=num_splits, workspace=workspace
        )

    run()  # compile outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for dst, src in zip(live, second):
        dst.copy_(src)
    graph.replay()
    torch.cuda.synchronize()
    assert check(second, out, scale, msg="graph replay") == 0


@pytest.mark.parametrize("num_q_heads, num_kv_heads", [(24, 2), (3, 1)])
def test_workspace_covers_every_captured_batch(num_q_heads, num_kv_heads):
    # The backend sizes one workspace at the captured maximum and reuses it.
    device = torch.device("cuda")
    max_rows = 160
    capacity = workspace_partials(max_rows, num_q_heads, num_kv_heads, TOPK, device)
    for rows in range(1, max_rows + 1):
        splits = recommended_num_splits(rows, num_kv_heads, TOPK, device)
        assert rows * num_q_heads * splits <= capacity


def test_supported_operands():
    def kv_pool(num_kv_heads, head_dim=HEAD_DIM, dtype=dtypes.bf16):
        return torch.empty(16, num_kv_heads, head_dim, dtype=dtype)

    supported = can_use_flydsl_qsa_sparse_decode
    q = torch.empty(2, 24, HEAD_DIM, dtype=dtypes.bf16)
    kv, kv_fp8 = kv_pool(2), kv_pool(2, dtype=dtypes.fp8)
    assert supported(q, kv, kv)
    assert supported(q, kv_fp8, kv_fp8)
    assert not supported(q.half(), kv, kv)
    assert not supported(q, kv.half(), kv.half())
    assert not supported(q, kv, kv_fp8)  # mixed K/V dtypes
    q_128, kv_128 = q[..., :128].contiguous(), kv_pool(2, head_dim=128)
    assert not supported(q_128, kv_128, kv_128)
    assert not supported(q, kv_pool(1), kv_pool(1))  # GQA group 24 > 16
    assert not supported(q, kv_pool(5), kv_pool(5))  # 24 heads over 5 KV heads
    strided = torch.empty(32, 2, HEAD_DIM, dtype=dtypes.bf16)[::2]
    assert not supported(q, strided, strided)


def test_reject_invalid_arguments():
    q, k, v, req_to_token, row_req, indices, seq_lens = make_inputs(4, 24, 2)
    scale = HEAD_DIM**-0.5

    def call(**overrides):
        args = {
            "q": q,
            "k_cache": k,
            "v_cache": v,
            "req_to_token": req_to_token,
            "row_req_pool_indices": row_req,
            "indices": indices,
            "seq_lens": seq_lens,
            "softmax_scale": scale,
            "num_splits": 4,
        }
        args.update(overrides)
        return flydsl_qsa_sparse_decode(**args)

    for overrides in [
        {"q": q.half()},
        {"q": q.transpose(0, 1).contiguous().transpose(0, 1)},
        {"indices": indices.long()},
        {"indices": indices[:3]},
        {"indices": indices.t().contiguous().t()},
        {"req_to_token": req_to_token.long()},
        {"row_req_pool_indices": row_req[:3]},
        {"row_req_pool_indices": row_req.float()},
        {"seq_lens": seq_lens[:3]},
        {
            "workspace": (
                torch.empty(1, dtype=dtypes.fp32),
                torch.empty(1, dtype=dtypes.fp32),
                torch.empty(HEAD_DIM, dtype=dtypes.fp32),
            )
        },
    ]:
        with pytest.raises(ValueError):
            call(**overrides)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl qsa sparse decode unsupported on %s; skipping", get_gfx()
        )
        return 0

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL QSA sparse decode correctness and perf",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16, dtypes.fp8],
        help="""KV pool dtype (bf16 or fp8).
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 4, 16, 64, 128],
        help="""Query rows (decode requests x speculative tokens).
    e.g.: -b 1 64""",
    )
    parser.add_argument(
        "-n",
        "--heads",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(24, 2), (3, 1)],
        help="""(num_q_heads, num_kv_heads); Qwen3.8 TP1 is 24,2 and TP8 is 3,1.
    e.g.: -n 24,2""",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[TOPK],
        help="""Selected positions per row.
    e.g.: -k 2051""",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="*",
        default=[0],
        help="""KV splits per row; 0 takes recommended_num_splits.
    e.g.: --splits 0 1""",
    )
    args = parser.parse_args()

    aiter.logger.info("qsa sparse decode: running correctness checks...")
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    if status != pytest.ExitCode.OK:
        aiter.logger.error(
            "qsa sparse decode: checks failed (pytest exit %s); skipping perf sweep",
            int(status),
        )
        return 1

    for kv_dtype in args.dtype:
        df = []
        for (hq, hkv), topk, splits, rows in itertools.product(
            args.heads, args.topk, args.splits, args.batch
        ):
            df.append(test_qsa_sparse_decode(rows, hq, hkv, topk, kv_dtype, splits))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "flydsl qsa sparse decode summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
