# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness sweep for the FlyDSL paged FP8 MQA-logits (decode) kernel.

The reference is a torch port of vLLM's ``fp8_paged_mqa_logits_torch``, copied
here so the test does not depend on vLLM. Gate: exact ``-inf``-mask match plus
``calc_diff < 1e-3``, tolerances NOT widened.
"""

import argparse
import itertools
import random
from typing import NamedTuple

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
    """Co-pack a bf16 KV cache into the fp8+scale byte layout: per block-row,
    KVBlockSize*head_dim fp8 bytes then KVBlockSize f32 scales, no padding.
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


def preshuffle_kv_data(kv_cache_fp8, head_dim):
    """Apply ``shuffle_weight(layout=(16,16))`` to the per-block fp8 key bytes,
    leaving the co-packed f32 scale tail alone -- the production Preshuffle
    layout. Only the kernel gets this copy; the oracle reads the unshuffled
    cache, since the reference is layout-agnostic.
    """
    from aiter.ops.shuffle import shuffle_weight

    num_blocks, block_size, one, index_dim = kv_cache_fp8.shape
    assert block_size % 16 == 0, "preshuffle requires KVBlockSize % 16 == 0"
    flat = kv_cache_fp8.reshape(num_blocks, block_size * index_dim).clone()
    data = (
        flat[:, : block_size * head_dim]
        .contiguous()
        .view(num_blocks, block_size, head_dim)
    )
    shuffled = shuffle_weight(data, layout=(16, 16)).reshape(
        num_blocks, block_size * head_dim
    )
    flat[:, : block_size * head_dim] = shuffled
    return flat.view(num_blocks, block_size, one, index_dim)


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
    """Torch reference, vectorized port of vLLM ``fp8_paged_mqa_logits_torch``::

        logits[b*next_n+n, p] = sum_h ReLU(<q[b,n,h,:], K_deq(p)>) * weights[.., h]

    masked by ``p <= context_len - next_n + n``. Position ``p`` resolves to
    ``(block_tables[b, p//KVB], p%KVB)`` in the block-flat co-packed cache.
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


class Inputs(NamedTuple):
    q: torch.Tensor  # bf16, for the oracle
    q_fp8: torch.Tensor  # quantized, for the kernel
    kv_cache_fp8: torch.Tensor  # co-packed, unshuffled
    weights: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    max_model_len: int
    fp8_dtype: torch.dtype


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

    # Block table: one entry per KVBlockSize-token block, handed out of a
    # shuffled pool in sequence order. The pool is sized from max_model_len, so
    # sequences share blocks once the batch needs more than it holds -- harmless
    # here (and extra coverage of block reuse), unlike in the benchmarks where
    # it would shrink the measured working set into cache.
    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    pool = list(range(num_blocks))
    random.shuffle(pool)
    pool_t = torch.tensor(pool, device="cuda", dtype=torch.int32)
    col = torch.arange(max_block_len, device="cuda", dtype=torch.int64)
    starts = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
    block_tables = torch.where(
        col[None, :] < blocks_per_seq[:, None],
        pool_t[(starts[:, None] + col[None, :]) % num_blocks],
        torch.zeros((), device="cuda", dtype=torch.int32),
    ).to(torch.int32)

    q_fp8 = q.to(q_dtype)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, fp8_dtype)
    return Inputs(
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
    preshuffle=False,
    variant=None,
    chunk_k=128,
):
    # split_kv == 0 -> auto; else an explicit override (1 disables splitting).
    # Both must be correctness-identical.
    _split_kv = None if split_kv == 0 else split_kv
    inp = _build_inputs(
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
            inp.q,
            inp.kv_cache_fp8,
            inp.weights,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            inp.fp8_dtype,
            block_size=block_size,
        )
    ref_mask = ref == float("-inf")

    out = torch.full(
        (batch_size * next_n, inp.max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    # Same numbers as the oracle's cache, in the production data layout.
    kv_cache_kernel = (
        preshuffle_kv_data(inp.kv_cache_fp8, head_dim)
        if preshuffle
        else inp.kv_cache_fp8
    )

    with torch.inference_mode():
        got = flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            Preshuffle=preshuffle,
            KVBlockSize=block_size,
            SplitKV=_split_kv,
            ChunkK=chunk_k,
            variant=variant,
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
        "preshuffle": preshuffle,
        "split_kv": "auto" if split_kv == 0 else split_kv,
        "variant": variant or "default",
        "chunk_k": chunk_k,
        "flydsl err": err,
    }


# The kernel specialises at compile time on (heads, head_dim, KVBlockSize,
# Preshuffle); batch, next_n, context length, SplitKV and the Q dtype are
# runtime values. Those axes do not interact -- nothing couples head_dim to
# SplitKV, or the dtype to KVBlockSize -- so crossing all seven produced 864
# cases that compiled only 12 distinct kernels and re-ran each about 72 times.
#
# Instead: cross the compile-time axes, vary each runtime axis alone from a
# base, and add back the few combinations where the tile-range arithmetic
# genuinely couples. `--exhaustive` still runs the full product.
_BASE = {
    "batch_size": 2,
    "next_n": 2,
    "heads": 64,
    "head_dim": 128,
    "avg_kv_length": 1024,
    "q_dtype": "fnuz",
}
_SHAPES = [(64, 64), (64, 128), (128, 64), (128, 128)]


_AXES = (
    "batch_size",
    "next_n",
    "heads",
    "head_dim",
    "avg_kv_length",
    "q_dtype",
    "split_kv",
    "block_size",
)


def _c(**kw):
    """A case: the base config with KVBlockSize=64, plus the overrides given."""
    return {**_BASE, "block_size": 64, **kw}


def default_cases():
    cases = []
    # Compile-time cross: one case per distinct compiled kernel. Preshuffle
    # needs KVBlockSize % 16 == 0.
    for heads, head_dim in _SHAPES:
        for kvb in (1, 64):
            cases.append(_c(heads=heads, head_dim=head_dim, block_size=kvb))
        for kvb in (16, 64):
            cases.append(
                _c(heads=heads, head_dim=head_dim, block_size=kvb, preshuffle=True)
            )
    # Runtime axes, one at a time off the base.
    cases += [_c(batch_size=b, next_n=n) for b, n in ((1, 1), (1, 2), (4, 2), (8, 1))]
    cases += [_c(avg_kv_length=kv) for kv in (128, 8192)]  # 128 < one ChunkK tile
    cases += [_c(split_kv=sk) for sk in (1, 4)]  # 1 disables splitting
    cases.append(_c(q_dtype="fn"))  # gfx942 FN->FNUZ Q patch
    # Interactions: SplitKV against the context length it partitions, and
    # against the preshuffled layout, which the old matrix never crossed.
    cases.append(_c(avg_kv_length=128, split_kv=4))
    cases.append(_c(avg_kv_length=8192, split_kv=1))
    cases.append(_c(split_kv=4, preshuffle=True))
    cases.append(_c(split_kv=1, block_size=16, preshuffle=True))
    # Wave split and KV tile width, neither covered before: a bug in the
    # wave-to-column-tile assignment is correct by construction at one wave per
    # CTA. ChunkK/32 must be divisible by the variant's wave count.
    cases += [_c(variant=v) for v in ("paged_w2", "paged_w4")]
    cases.append(_c(chunk_k=64, variant="paged_w2"))
    cases.append(_c(chunk_k=256, variant="paged_w4"))
    cases.append(_c(chunk_k=256))
    cases.append(_c(preshuffle=True, variant="paged_w4"))
    return cases


def exhaustive_cases():
    """The original 7-way cartesian product (870 cases), for release validation."""
    prod = itertools.product(
        [(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (8, 1)],
        [64, 128],
        [64, 128],
        [128, 1024, 8192],
        ["fnuz", "fn"],
        [0, 1, 4],
        [1, 64],
    )
    cases = [
        dict(zip(_AXES, (bs, nn, nh, hd, kv, qd, sk, kvb)))
        for (bs, nn), nh, hd, kv, qd, sk, kvb in prod
    ]
    # Preshuffle was covered by a hand-picked set rather than a full cross.
    return cases + [
        {**dict(zip(_AXES, (bs, nn, nh, hd, kv, qd, 0, kvb))), "preshuffle": True}
        for bs, nn, nh, hd, kv, qd, kvb in [
            (1, 1, 64, 128, 1024, "fnuz", 16),
            (1, 2, 64, 128, 1024, "fnuz", 64),
            (2, 1, 128, 128, 8192, "fnuz", 64),
            (1, 1, 64, 64, 1024, "fnuz", 16),
            (1, 1, 64, 128, 1024, "fn", 64),
            (4, 2, 64, 128, 8192, "fnuz", 64),
        ]
    ]


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
        description="FlyDSL paged fp8_mqa_logits correctness sweep"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="run the full 7-way cartesian product (870 cases) instead of the "
        "curated matrix; for release validation",
    )
    parser.add_argument(
        "--no-preshuffle", action="store_true", help="skip the preshuffle cases"
    )
    args = parser.parse_args()

    cases = exhaustive_cases() if args.exhaustive else default_cases()
    if args.no_preshuffle:
        cases = [c for c in cases if not c.get("preshuffle")]

    df = [test_fp8_paged_mqa_logits(**c) for c in cases]
    df = pd.DataFrame(df)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        # `tabulate` (optional) not installed -> plain-text table.
        summary = df.to_string(index=False)
    aiter.logger.info("fp8_paged_mqa_logits summary:\n%s", summary)


if __name__ == "__main__":
    main()
