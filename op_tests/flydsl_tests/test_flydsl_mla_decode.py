# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and perf tests for the FlyDSL gfx1250 MLA (Multi-head Latent
Attention) decode kernel.

Run under pytest for the parametrized correctness matrix, or as a script for
the correctness + perf sweep that ends in a markdown summary table:

    pytest op_tests/flydsl_tests/test_flydsl_mla_decode.py
    python3 op_tests/flydsl_tests/test_flydsl_mla_decode.py -q 16 128
"""

import argparse
import itertools
import math
import random

import pandas as pd
import pytest
import torch

pytest.importorskip("flydsl")
import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.flydsl.mla_decode import flydsl_mla_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)

SUPPORTED_GFX = ["gfx1250"]


def _gfx() -> str:
    try:
        return get_gfx()
    except Exception:  # noqa: BLE001 - no GPU / detection failure means "unsupported"
        return ""


pytestmark = pytest.mark.skipif(
    _gfx() not in SUPPORTED_GFX,
    reason="flydsl_mla_decode is gfx1250 only",
)


# need to update it for fp8 dtype
def shuffle_kv_buffer(
    kv_buffer: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    kv_lora_rank: int,
) -> torch.Tensor:
    """Shuffle KV cache layout for optimized WMMA-fragment loads.

    layout: (num_lanes, num_elements_per_thread) = (16, 8) for bf16/fp16 on gfx1250.
    WMMA instruction shape (bf16/fp16): 16x16x32.

    Returns a contiguous tensor shaped
    ``[num_blocks, num_kv_heads, block_size, head_size]`` with the bytes
    reordered within each tile.
    """
    dtype = kv_buffer.dtype
    assert dtype in (torch.bfloat16, torch.float16), f"unsupported dtype {dtype}"

    # 16-bit dtypes use a (16, 8) lane layout on gfx1250.
    num_lanes, num_elements_per_thread = (16, 8)

    _num_blocks, block_size, num_kv_heads, head_size = kv_buffer.shape
    assert block_size >= 16
    assert block_size % num_lanes == 0

    def shuffle(kvb, h):
        kvb = kvb.view(
            -1,
            num_kv_heads,
            block_size // num_lanes,
            num_lanes,
            h // (2 * num_elements_per_thread),
            2,  # 2 thread groups: t0..t15 and t16..t31
            num_elements_per_thread,
        )
        kvb = kvb.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
        kvb = kvb.view(-1, num_kv_heads, block_size // 16, h * 16)
        return kvb

    kv_shuffled = kv_buffer.view(-1, block_size, num_kv_heads, head_size).permute(
        0, 2, 1, 3
    )
    lora = shuffle(kv_shuffled[..., :kv_lora_rank], kv_lora_rank)
    rope = shuffle(kv_shuffled[..., kv_lora_rank:], head_size - kv_lora_rank)
    lora = lora.view(-1, num_kv_heads, block_size * kv_lora_rank)
    rope = rope.view(-1, num_kv_heads, block_size * (head_size - kv_lora_rank))
    kv_shuffled = torch.cat([lora, rope], dim=-1).contiguous()
    kv_shuffled = kv_shuffled.view(-1, num_kv_heads, block_size, head_size)
    return kv_shuffled


# need to update for fp8 dtype
def _ref_masked_attention(
    q: torch.Tensor,  # [1, num_q_heads, head_size]   (query_len == 1: decode)
    k: torch.Tensor,  # [kv_len, num_kv_heads, head_size]
    v: torch.Tensor,  # [kv_len, num_kv_heads, kv_lora_rank]
    scale: float,
) -> torch.Tensor:

    if q.shape[1] != k.shape[1]:  # GQA / MQA expand kv heads up to query heads
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    k = k.to(q.dtype)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # [num_q_heads, 1, kv_len]
    attn = attn * scale
    attn = torch.softmax(attn, dim=-1).to(q.dtype)
    v = v.to(q.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # [1, num_q_heads, kv_lora_rank]
    return out


def _torch_mla_decode_ref(
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_size]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens: torch.Tensor,  # [num_seqs]
    scale: float,
    kv_lora_rank: int,
) -> torch.Tensor:
    """MLA decode golden. Returns [num_seqs, num_q_heads, kv_lora_rank]."""
    num_seqs, num_q_heads, head_size = query.shape
    _, block_size, num_kv_heads, qk_head_dim = kv_cache.shape
    assert head_size == qk_head_dim
    device = query.device

    outputs = []
    for i in range(num_seqs):
        kv_len = int(seq_lens[i].item())
        if kv_len <= 0:
            outputs.append(
                torch.zeros(
                    (1, num_q_heads, kv_lora_rank), dtype=query.dtype, device=device
                )
            )
            continue

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = kv_cache[block_indices].view(-1, num_kv_heads, qk_head_dim)[:kv_len]
        v = k[..., :kv_lora_rank]

        q = query[i : i + 1]  # [1, num_q_heads, head_size]
        outputs.append(_ref_masked_attention(q, k, v, scale))

    return torch.cat(outputs, dim=0)  # [num_seqs, num_q_heads, kv_lora_rank]


def _generate_inputs(
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    ctx_len: int,
    dtype: torch.dtype,
    varlen: bool = False,
    num_blocks: int | None = None,
    seed: int = 42,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    random.seed(seed)

    qk_head_dim = kv_lora_rank + qk_rope_head_dim

    if varlen:
        lens = [
            int(max(random.normalvariate(ctx_len, ctx_len / 2), ctx_len))
            for _ in range(num_seqs)
        ]
        seq_lens = torch.tensor(lens, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.full((num_seqs,), ctx_len, dtype=torch.int32, device=device)

    # Block-table width derived from the realized max
    max_seqlen = int(seq_lens.max().item())
    max_num_blocks_per_seq = (max_seqlen + block_size - 1) // block_size

    if num_blocks is None:
        num_blocks = max_num_blocks_per_seq * num_seqs + 16

    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    kv_cache = torch.randn(
        (num_blocks, block_size, num_kv_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)
    query = torch.randn(
        (num_seqs, num_query_heads, qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtype)

    return query, kv_cache, block_tables, seq_lens


_KV_LORA_RANK = 512
_QK_ROPE_HEAD_DIM = 64


def _build_case(*, num_seqs, ctx_len, dtype, varlen, block_size, num_query_heads):
    """Build kernel inputs + torch reference for one decode case.

    Reproduces the real model call: pre-shuffled KV cache and a caller-owned
    preallocated output buffer.
    """
    query, kv_cache, block_tables, seq_lens = _generate_inputs(
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=1,  # MLA: single shared latent KV head
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        block_size=block_size,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=varlen,
    )
    attn_scale = 1.0 / math.sqrt(_KV_LORA_RANK + _QK_ROPE_HEAD_DIM)

    ref = _torch_mla_decode_ref(
        query, kv_cache, block_tables, seq_lens, attn_scale, _KV_LORA_RANK
    )

    kernel_kv_cache = shuffle_kv_buffer(kv_cache, _KV_LORA_RANK)
    del kv_cache

    output = torch.zeros(
        (num_seqs, num_query_heads, _KV_LORA_RANK), dtype=dtype, device=query.device
    )
    max_seqlen = int(seq_lens.max().item())
    return (
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        ref,
        output,
        attn_scale,
        max_seqlen,
    )


def _run_decode_case(
    *,
    num_seqs,
    ctx_len,
    dtype,
    varlen,
    block_size,
    num_query_heads,
):
    (
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        ref,
        output,
        attn_scale,
        max_seqlen,
    ) = _build_case(
        num_seqs=num_seqs,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=varlen,
        block_size=block_size,
        num_query_heads=num_query_heads,
    )
    flydsl_mla_decode(
        output,
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        attn_scale,
        max_seqlen=max_seqlen,
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
    )

    assert not torch.isnan(output).any(), "output contains NaN"
    torch.testing.assert_close(output, ref, atol=1.5e-2, rtol=1e-2)


# (num_seqs, ctx_len)
_CASES = [
    (1, 200),
    (1, 600),
    (1, 256),
    (2, 400),
    (8, 1024),
]
_BLOCK_SIZES = [16, 64, 128]

_NUM_Q_HEADS = [16, 32, 48, 64, 128]


@pytest.mark.parametrize("num_seqs,ctx_len", _CASES)
@pytest.mark.parametrize("num_q_heads", _NUM_Q_HEADS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("varlen", [True, False])
@pytest.mark.parametrize("block_size", _BLOCK_SIZES)
def test_flydsl_mla_decode(num_seqs, ctx_len, num_q_heads, dtype, varlen, block_size):
    _run_decode_case(
        num_seqs=num_seqs,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=varlen,
        block_size=block_size,
        num_query_heads=num_q_heads,
    )


_LARGE_CASES = [
    (1024, 8192),
    (1024, 16384),
    (1024, 32768),
]


@pytest.mark.parametrize("num_seqs,ctx_len", _LARGE_CASES)
@pytest.mark.parametrize("num_q_heads", _NUM_Q_HEADS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_flydsl_mla_decode_large(num_seqs, ctx_len, num_q_heads, dtype):
    _run_decode_case(
        num_seqs=num_seqs,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=True,
        block_size=64,
        num_query_heads=num_q_heads,
    )


@benchmark()
def test_mla_decode_perf(num_seqs, ctx_len, num_q_heads, block_size, dtype, varlen):
    (
        query,
        kernel_kv_cache,
        block_tables,
        seq_lens,
        ref,
        output,
        attn_scale,
        max_seqlen,
    ) = _build_case(
        num_seqs=num_seqs,
        ctx_len=ctx_len,
        dtype=dtype,
        varlen=varlen,
        block_size=block_size,
        num_query_heads=num_q_heads,
    )

    candidates = {
        "flydsl": lambda: flydsl_mla_decode(
            output,
            query,
            kernel_kv_cache,
            block_tables,
            seq_lens,
            attn_scale,
            max_seqlen=max_seqlen,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        ),
    }

    # Roofline: QK over (d_c + d_rope) plus PV over d_c, per (seq, q-head).
    # BW counts UNIQUE KV bytes only — excludes Q, partials, output, re-reads.
    sum_seq_lens = int(seq_lens.sum().item())
    qk_head_dim = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
    flops = 2 * num_q_heads * sum_seq_lens * (qk_head_dim + _KV_LORA_RANK)
    nbytes = sum_seq_lens * qk_head_dim * query.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1.5e-2,
            msg=f"{name}: mla_decode",
            printLog=False,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# pytest collects the parametrized tests above; main() drives this sweep.
test_mla_decode_perf.__test__ = False


def main():
    # Arch-gate here, not in the @benchmark fn (an in-fn return still emits a NaN row).
    if _gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_mla_decode unsupported on %s; skipping", _gfx() or "<no gpu>"
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL gfx1250 MLA decode correctness + perf sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="kernel dtypes to sweep (bf16 f16)",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8, 1024), (1024, 16384)],
        help="(num_seqs, ctx_len) pairs to sweep",
    )
    parser.add_argument(
        "-q",
        "--num-q-heads",
        type=int,
        nargs="*",
        default=_NUM_Q_HEADS,
        help="query head counts to sweep",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        nargs="*",
        default=_BLOCK_SIZES,
        help="paged-KV page sizes to sweep",
    )
    parser.add_argument(
        "--varlen",
        type=int,
        nargs="*",
        default=[0],
        choices=[0, 1],
        help="0: uniform seq_lens == ctx_len, 1: random varlen (>= ctx_len)",
    )
    args = parser.parse_args()

    df = []
    for dtype, (num_seqs, ctx_len), nqh, bs, vl in itertools.product(
        args.dtype, args.shapes, args.num_q_heads, args.block_size, args.varlen
    ):
        df.append(test_mla_decode_perf(num_seqs, ctx_len, nqh, bs, dtype, bool(vl)))
    df = pd.DataFrame(df)
    aiter.logger.info("mla_decode summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
