# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness sweep for the FlyDSL paged FP8 MQA-logits (decode) kernel.

Mirrors the aiter-op-test standard used by ``test_flydsl_fp8_mqa_logits.py`` and
the paged benchmark ``op_tests/op_benchmarks/triton/bench_deepgemm_attention.py``.
The correctness reference is a torch port of vLLM's ``fp8_paged_mqa_logits_torch``
(dequantizes the co-packed fp8 cache and applies the causal mask); it is copied
here so the test does not depend on vLLM. Correctness gate: exact ``-inf``-mask
match + ``calc_diff < 1e-3`` (tolerances are NOT widened).
"""

import argparse
import itertools
import random

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

torch.set_default_device("cuda")

# gfx942 is the primary target; gfx950 (native fp8) is also runnable.
SUPPORTED_GFX = ["gfx942", "gfx950"]
# "fnuz" == the arch-native e4m3 (fnuz on gfx942, fn on gfx950); "fn" forces the
# OCP e4m3fn operand (exercises the gfx942 FN->FNUZ patch). Mirrors the dense
# test's dtype aliasing.
_E4M3_NATIVE = get_fp8_e4m3_dtype()
DTYPE_MAP = {"fnuz": _E4M3_NATIVE, "fn": torch.float8_e4m3fn}

try:
    from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
except ImportError:
    flydsl_fp8_paged_mqa_logits = None


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return 1 - 2 * (x * y).sum() / denominator


def kv_cache_cast_to_fp8(x, fp8_dtype):
    """Co-pack a bf16 KV cache into the fp8+scale byte layout (KVBlockSize>=1).

    Layout per block-row: KVBlockSize*head_dim fp8 bytes, then KVBlockSize f32
    (4-byte) per-token scales. No 16B padding (Phase 1). Mirrors the paged
    benchmark's builder.
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(fp8_dtype)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)


def ref_fp8_paged_mqa_logits(
    q,
    kv_cache_fp8,
    weights,
    context_lens,
    block_tables,
    max_model_len,
    fp8_dtype,
    block_size=1,
):
    """Torch reference (vectorized port of vLLM ``fp8_paged_mqa_logits_torch``).

    Dequantizes the **block-flat** co-packed fp8 cache and computes:
        logits[b*next_n+n, p] = sum_h ReLU(<q[b,n,h,:], K_deq(p)>) * weights[.., h]
    with the causal mask ``p <= context_len - next_n + n``. Logical position ``p``
    resolves to physical ``(block, tok) = (block_tables[b, p//KVB], p%KVB)`` and
    the block-flat block groups all ``KVB`` fp8 key rows first, then all ``KVB``
    f32 scales. ``block_size == 1`` degenerates to the per-token page table.
    """
    batch_size, next_n, heads, dim = q.size()
    num_blocks = kv_cache_fp8.shape[0]
    index_dim = kv_cache_fp8.shape[-1]
    # Reconstruct the block-flat block: [KVB keys (dim bytes each)][KVB f32 scales].
    flat = kv_cache_fp8.reshape(num_blocks, block_size * index_dim)
    keys = (
        flat[:, : block_size * dim]
        .contiguous()
        .view(fp8_dtype)
        .float()
        .view(num_blocks, block_size, dim)
    )
    scales = (
        flat[:, block_size * dim : block_size * dim + 4 * block_size]
        .contiguous()
        .view(torch.float32)
        .view(num_blocks, block_size, 1)
    )
    kvf = keys * scales  # [num_blocks, block_size, dim] dequantized K
    qf = q.float()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = int(context_lens[i].item())
        if context_len == 0:
            continue
        pos = torch.arange(context_len, device=q.device)
        blk = block_tables[i, pos // block_size]  # [ctx] physical block ids
        tok = pos % block_size  # [ctx] token-in-block
        kx = kvf[blk, tok]  # [ctx, dim] dequantized K
        s = torch.einsum("nhd,pd->nhp", qf[i], kx)  # [next_n, heads, ctx]
        s = torch.relu(s)
        wl = weights[i * next_n : (i + 1) * next_n, :]  # [next_n, heads]
        s = (s * wl[:, :, None]).sum(dim=1)  # [next_n, ctx]
        q_lim = (
            context_len - next_n + torch.arange(next_n, device=q.device)
        ).unsqueeze(
            1
        )  # [next_n, 1]
        s = torch.where(pos[None, :] <= q_lim, s, float("-inf"))
        logits[i * next_n : (i + 1) * next_n, :context_len] = s
    return logits


def _build_inputs(
    batch_size, next_n, heads, head_dim, avg_kv_length, q_dtype, block_size=1, seed=0
):
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * avg_kv_length
    num_blocks = (max_model_len + block_size - 1) // block_size

    lo = max(1, int((1 - 0.5) * avg_kv_length))
    hi = int((1 + 0.5) * avg_kv_length) + 1
    context_lens = torch.randint(lo, hi, (batch_size,)).cuda().to(torch.int32)
    # decode with MTP needs at least next_n tokens of context.
    context_lens = torch.clamp(context_lens, min=next_n)

    q = torch.randn((batch_size, next_n, heads, head_dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, block_size, 1, head_dim), dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads), dtype=torch.float32)

    # block table: one entry per KVBlockSize-token block, ceil(ctx / block_size).
    max_block_len = (int(context_lens.max().item()) + block_size - 1) // block_size
    block_tables = torch.zeros(
        (batch_size, max_block_len), device="cuda", dtype=torch.int32
    )
    pool = list(range(num_blocks))
    random.shuffle(pool)
    counter = 0
    for i in range(batch_size):
        n_blk = (int(context_lens[i].item()) + block_size - 1) // block_size
        for j in range(n_blk):
            block_tables[i][j] = pool[counter % num_blocks]
            counter += 1

    q_fp8 = q.to(q_dtype)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, fp8_dtype)
    return (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    )


@benchmark()
def test_fp8_paged_mqa_logits(
    batch_size,
    next_n,
    heads,
    head_dim,
    avg_kv_length,
    q_dtype,
    split_kv=0,
    block_size=1,
):
    # split_kv == 0 -> auto (production host formula); else an explicit override
    # (1 disables splitting). Both must be correctness-identical.
    _split_kv = None if split_kv == 0 else split_kv
    (
        q,
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    ) = _build_inputs(
        batch_size,
        next_n,
        heads,
        head_dim,
        avg_kv_length,
        DTYPE_MAP[q_dtype],
        block_size=block_size,
    )

    with torch.inference_mode():
        ref = ref_fp8_paged_mqa_logits(
            q,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            fp8_dtype,
            block_size=block_size,
        )
    ref_mask = ref == float("-inf")

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    with torch.inference_mode():
        got = flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            KVBlockSize=block_size,
            SplitKV=_split_kv,
        )

    got_mask = got == float("-inf")
    assert torch.equal(got_mask, ref_mask), "flydsl paged: -inf mask mismatch"

    err = 0.0
    if not ref_mask.all():
        diff = calc_diff(got.masked_fill(got_mask, 0), ref.masked_fill(ref_mask, 0))
        assert diff < 1e-3, f"flydsl paged calc_diff={diff}"
        err = diff.item()
        checkAllclose(
            ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
            got.masked_fill(got_mask, 0).to(dtypes.fp32),
            rtol=1e-2,
            atol=5.0,
            msg="flydsl paged fp8_mqa_logits",
            printLog=False,
        )

    return {
        "gfx": get_gfx(),
        "kvb": block_size,
        "split_kv": "auto" if split_kv == 0 else split_kv,
        "flydsl err": err,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_logits unsupported on %s; skipping", get_gfx()
        )
        return
    if flydsl_fp8_paged_mqa_logits is None:
        aiter.logger.warning("flydsl package not installed; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL paged fp8_mqa_logits correctness sweep",
    )
    parser.add_argument(
        "-b",
        "--batch-next-n",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (8, 1)],
        help="(batch_size, next_n) pairs",
    )
    parser.add_argument("--num-heads", type=int, nargs="*", default=[64, 128])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[64, 128])
    parser.add_argument(
        "--avg-kv-length", type=int, nargs="*", default=[128, 1024, 8192]
    )
    parser.add_argument(
        "--q-dtype",
        type=str,
        nargs="*",
        default=["fnuz", "fn"],
        choices=["fnuz", "fn"],
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=[0, 1, 4],
        help="0 == auto (production formula); else explicit SplitKV (1 disables)",
    )
    parser.add_argument(
        "--kv-block-size",
        type=int,
        nargs="*",
        default=[1, 64],
        help="KVBlockSize (paged block-flat page size); 1 == per-token slots",
    )
    args = parser.parse_args()

    df = []
    for (bs, nn), nh, hd, kv, qd, sk, kvb in itertools.product(
        args.batch_next_n,
        args.num_heads,
        args.head_dim,
        args.avg_kv_length,
        args.q_dtype,
        args.split_kv,
        args.kv_block_size,
    ):
        df.append(test_fp8_paged_mqa_logits(bs, nn, nh, hd, kv, qd, sk, block_size=kvb))

    df = pd.DataFrame(df)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        # `tabulate` (optional) not installed -> plain-text table.
        summary = df.to_string(index=False)
    aiter.logger.info("fp8_paged_mqa_logits summary:\n%s", summary)


if __name__ == "__main__":
    main()
