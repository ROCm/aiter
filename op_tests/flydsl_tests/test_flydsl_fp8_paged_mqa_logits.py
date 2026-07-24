# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf sweep for the FlyDSL paged FP8 MQA-logits (decode) kernel.

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
from aiter.test_common import benchmark, checkAllclose, run_perftest
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
    q, kv_cache_fp8, weights, context_lens, block_tables, max_model_len, fp8_dtype
):
    """Torch reference (vectorized port of vLLM ``fp8_paged_mqa_logits_torch``).

    Dequantizes the co-packed fp8 cache (``fp8_bytes.float() * scale``) so it
    matches what the kernel computes (kv-scale folded into K), then:
        logits[b*next_n+n, p] = sum_h ReLU(<q[b,n,h,:], K_deq(p)>) * weights[.., h]
    with the causal mask ``p <= context_len - next_n + n``. The inner per-token
    loop of the original is replaced by a single gather + einsum per batch
    element (block_size == 1), which is orders of magnitude faster at long
    context while numerically identical.
    """
    batch_size, next_n, heads, dim = q.size()
    kvv, scale = kv_cache_fp8[..., :dim], kv_cache_fp8[..., dim:]
    scale = scale.contiguous().view(torch.float)
    qf = q.float()
    # dequantized K per physical block: [num_blocks, dim]  (block_size == 1)
    kvf = (kvv.view(fp8_dtype).float() * scale).view(kv_cache_fp8.shape[0], dim)
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
        pages = block_tables[i, :context_len]  # [ctx] physical token ids
        kx = kvf[pages]  # [ctx, dim] dequantized K
        s = torch.einsum("nhd,pd->nhp", qf[i], kx)  # [next_n, heads, ctx]
        s = torch.relu(s)
        wl = weights[i * next_n : (i + 1) * next_n, :]  # [next_n, heads]
        s = (s * wl[:, :, None]).sum(dim=1)  # [next_n, ctx]
        p = torch.arange(context_len, device=q.device)
        q_lim = (
            context_len - next_n + torch.arange(next_n, device=q.device)
        ).unsqueeze(
            1
        )  # [next_n, 1]
        s = torch.where(p[None, :] <= q_lim, s, float("-inf"))
        logits[i * next_n : (i + 1) * next_n, :context_len] = s
    return logits


def _build_inputs(batch_size, next_n, heads, head_dim, avg_kv_length, q_dtype, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * avg_kv_length
    num_blocks = max_model_len  # KVBlockSize == 1

    lo = max(1, int((1 - 0.5) * avg_kv_length))
    hi = int((1 + 0.5) * avg_kv_length) + 1
    context_lens = torch.randint(lo, hi, (batch_size,)).cuda().to(torch.int32)
    # decode with MTP needs at least next_n tokens of context.
    context_lens = torch.clamp(context_lens, min=next_n)

    q = torch.randn((batch_size, next_n, heads, head_dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, 1, 1, head_dim), dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads), dtype=torch.float32)

    max_block_len = int(context_lens.max().item())  # blocksize == 1
    block_tables = torch.zeros(
        (batch_size, max_block_len), device="cuda", dtype=torch.int32
    )
    pool = list(range(num_blocks))
    random.shuffle(pool)
    counter = 0
    for i in range(batch_size):
        for j in range(int(context_lens[i].item())):
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
    batch_size, next_n, heads, head_dim, avg_kv_length, q_dtype, split_kv=0
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
        batch_size, next_n, heads, head_dim, avg_kv_length, DTYPE_MAP[q_dtype]
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
        )
    ref_mask = ref == float("-inf")

    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    def fn():
        out.fill_(float("-inf"))
        return flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            SplitKV=_split_kv,
        )

    with torch.inference_mode():
        got, us = run_perftest(fn)

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

    total_flops = 2 * next_n * heads * head_dim * context_lens.float().sum().item()
    return {
        "gfx": get_gfx(),
        "split_kv": "auto" if split_kv == 0 else split_kv,
        "flydsl us": us,
        "flydsl TFLOPS": total_flops / us / 1e6 if us > 0 else 0,
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
        description="FlyDSL paged fp8_mqa_logits correctness + perf sweep",
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
    args = parser.parse_args()

    df = []
    for (bs, nn), nh, hd, kv, qd, sk in itertools.product(
        args.batch_next_n,
        args.num_heads,
        args.head_dim,
        args.avg_kv_length,
        args.q_dtype,
        args.split_kv,
    ):
        df.append(test_fp8_paged_mqa_logits(bs, nn, nh, hd, kv, qd, sk))

    df = pd.DataFrame(df)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        # `tabulate` (optional) not installed -> plain-text table.
        summary = df.to_string(index=False)
    aiter.logger.info("fp8_paged_mqa_logits summary:\n%s", summary)


if __name__ == "__main__":
    main()
