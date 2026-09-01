# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness sweep and benchmarks for the FlyDSL paged FP8 MQA-logits kernel.

The reference is a torch port of vLLM's ``fp8_paged_mqa_logits_torch``, copied
here so the test does not depend on vLLM. Gate: exact ``-inf``-mask match plus
``calc_diff < 1e-3``, tolerances NOT widened.

Performance timing and optional Gluon A/B comparison live in this file rather
than separate benchmark scripts.
"""

import argparse
import itertools
import math
import random
from typing import NamedTuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
_E4M3_NATIVE = get_fp8_e4m3_dtype()
DTYPE_MAP = {"fnuz": _E4M3_NATIVE, "fn": torch.float8_e4m3fn}

REF_CHUNK_K = 256
REF_WAVE_PER_EU = 2


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
    """Torch reference, vectorized port of vLLM ``fp8_paged_mqa_logits_torch``."""
    batch_size, next_n, _heads, dim = q.size()
    num_blocks = kv_cache_fp8.shape[0]
    index_dim = kv_cache_fp8.shape[-1]
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
    kvf = keys * scales
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
        blk = block_tables[i, pos // block_size]
        tok = pos % block_size
        kx = kvf[blk, tok]
        s = torch.einsum("nhd,pd->nhp", qf[i], kx)
        s = torch.relu(s)
        wl = weights[i * next_n : (i + 1) * next_n, :]
        s = (s * wl[:, :, None]).sum(dim=1)
        q_lim = (
            context_len - next_n + torch.arange(next_n, device=q.device)
        ).unsqueeze(1)
        s = torch.where(pos[None, :] <= q_lim, s, float("-inf"))
        logits[i * next_n : (i + 1) * next_n, :context_len] = s
    return logits


class Inputs(NamedTuple):
    q: torch.Tensor
    q_fp8: torch.Tensor
    kv_cache_fp8: torch.Tensor
    weights: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    max_model_len: int
    fp8_dtype: torch.dtype


def _build_inputs(
    batch_size,
    next_n,
    heads,
    head_dim,
    avg_kv_length,
    q_dtype,
    block_size=1,
    seed=0,
    var_ratio=0.0,
    pool_blocks=0,
):
    torch.manual_seed(seed)
    random.seed(seed)
    fp8_dtype = get_fp8_e4m3_dtype()

    max_model_len = 2 * avg_kv_length
    if var_ratio == 0.0:
        context_lens = torch.full(
            (batch_size,), avg_kv_length, device="cuda", dtype=torch.int32
        )
    else:
        lo = max(1, int((1 - var_ratio) * avg_kv_length))
        hi = int((1 + var_ratio) * avg_kv_length) + 1
        context_lens = torch.randint(lo, hi, (batch_size,)).cuda().to(torch.int32)
    context_lens = torch.clamp(context_lens, min=next_n)

    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    needed_blocks = int(blocks_per_seq.sum().item())
    num_blocks = needed_blocks if pool_blocks <= 0 else max(pool_blocks, max_block_len)

    q = torch.randn((batch_size, next_n, heads, head_dim), dtype=torch.bfloat16)
    kv_cache = torch.randn((num_blocks, block_size, 1, head_dim), dtype=torch.bfloat16)
    weights = torch.randn((batch_size * next_n, heads), dtype=torch.float32)

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


def _kernel_inputs(inp, batch_size, next_n, head_dim, preshuffle, block_size):
    kv_cache_kernel = (
        preshuffle_kv_data(inp.kv_cache_fp8, head_dim)
        if preshuffle
        else inp.kv_cache_fp8
    )
    out = torch.full(
        (batch_size * next_n, inp.max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    return kv_cache_kernel, out


def _verify_paged_mqa_logits(
    inp,
    batch_size,
    next_n,
    head_dim,
    split_kv,
    block_size,
    preshuffle,
    variant,
    chunk_k,
):
    _split_kv = None if split_kv == 0 else split_kv
    kv_cache_kernel, out = _kernel_inputs(
        inp, batch_size, next_n, head_dim, preshuffle, block_size
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
    return err, kv_cache_kernel, out


@benchmark()
def _bench_flydsl_paged_kernel(
    q_fp8,
    kv_cache_kernel,
    weights,
    out,
    context_lens,
    block_tables,
    max_model_len,
    *,
    preshuffle,
    block_size,
    split_kv,
    chunk_k,
    variant,
):
    _split_kv = None if split_kv == 0 else split_kv

    def fn():
        return flydsl_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_kernel,
            weights,
            out,
            context_lens,
            block_tables,
            max_model_len,
            Preshuffle=preshuffle,
            KVBlockSize=block_size,
            SplitKV=_split_kv,
            ChunkK=chunk_k,
            variant=variant,
        )

    with torch.inference_mode():
        _, us = run_perftest(fn)
    return {"flydsl us": float(us)}


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
    bench=False,
):
    inp = _build_inputs(
        batch_size,
        next_n,
        heads,
        head_dim,
        avg_kv_length,
        DTYPE_MAP[q_dtype],
        block_size=block_size,
    )
    err, kv_cache_kernel, out = _verify_paged_mqa_logits(
        inp,
        batch_size,
        next_n,
        head_dim,
        split_kv,
        block_size,
        preshuffle,
        variant,
        chunk_k,
    )
    ret = {
        "gfx": get_gfx(),
        "kvb": block_size,
        "preshuffle": preshuffle,
        "split_kv": "auto" if split_kv == 0 else split_kv,
        "variant": variant or "default",
        "chunk_k": chunk_k,
        "flydsl err": err,
    }
    if not bench:
        return ret

    ret.update(
        _bench_flydsl_paged_kernel(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            preshuffle=preshuffle,
            block_size=block_size,
            split_kv=split_kv,
            chunk_k=chunk_k,
            variant=variant,
        )
    )

    if block_size == 1 and not preshuffle:
        from aiter.ops.triton.attention.pa_mqa_logits import (
            deepgemm_fp8_paged_mqa_logits,
        )

        out_ref = torch.full(
            (batch_size * next_n, inp.max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )

        def fn_ref():
            return deepgemm_fp8_paged_mqa_logits(
                inp.q_fp8,
                inp.kv_cache_fp8,
                inp.weights,
                out_ref,
                inp.context_lens,
                inp.block_tables,
                inp.max_model_len,
                ChunkK=REF_CHUNK_K,
                Preshuffle=False,
                KVBlockSize=1,
                WavePerEU=REF_WAVE_PER_EU,
            )

        with torch.inference_mode():
            _, ref_us = run_perftest(fn_ref)
        ret["gluon us"] = float(ref_us)

    return ret


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
    return {**_BASE, "block_size": 64, **kw}


def default_cases():
    cases = []
    for heads, head_dim in _SHAPES:
        for kvb in (1, 64):
            cases.append(_c(heads=heads, head_dim=head_dim, block_size=kvb))
        for kvb in (16, 64):
            cases.append(
                _c(heads=heads, head_dim=head_dim, block_size=kvb, preshuffle=True)
            )
    cases += [_c(batch_size=b, next_n=n) for b, n in ((1, 1), (1, 2), (4, 2), (8, 1))]
    cases += [_c(avg_kv_length=kv) for kv in (128, 8192)]
    cases += [_c(split_kv=sk) for sk in (1, 4)]
    cases.append(_c(avg_kv_length=128, split_kv=4))
    cases.append(_c(avg_kv_length=8192, split_kv=1))
    cases.append(_c(split_kv=4, preshuffle=True))
    cases.append(_c(split_kv=1, block_size=16, preshuffle=True))
    cases += [_c(variant=v) for v in ("paged_w2", "paged_w4")]
    cases.append(_c(chunk_k=64, variant="paged_w2"))
    cases.append(_c(chunk_k=256, variant="paged_w4"))
    cases.append(_c(chunk_k=256))
    cases.append(_c(preshuffle=True, variant="paged_w4"))
    return cases


def exhaustive_cases():
    prod = itertools.product(
        [(1, 1), (1, 2), (2, 1), (2, 2), (4, 2), (8, 1)],
        [64, 128],
        [64, 128],
        [128, 1024, 8192],
        ["fnuz"],
        [0, 1, 4],
        [1, 64],
    )
    cases = [
        dict(zip(_AXES, (bs, nn, nh, hd, kv, qd, sk, kvb)))
        for (bs, nn), nh, hd, kv, qd, sk, kvb in prod
    ]
    return cases + [
        {**dict(zip(_AXES, (bs, nn, nh, hd, kv, qd, 0, kvb))), "preshuffle": True}
        for bs, nn, nh, hd, kv, qd, kvb in [
            (1, 1, 64, 128, 1024, "fnuz", 16),
            (1, 2, 64, 128, 1024, "fnuz", 64),
            (2, 1, 128, 128, 8192, "fnuz", 64),
            (1, 1, 64, 64, 1024, "fnuz", 16),
            (4, 2, 64, 128, 8192, "fnuz", 64),
        ]
    ]


def gluon_ab_shapes():
    H, D = 64, 128
    shapes = []
    for B in (1, 4, 16, 64, 128):
        for avg_kv in (16384, 32768, 65536):
            shapes.append((B, 2, H, D, avg_kv))
    for B in (1, 16, 128):
        shapes.append((B, 1, H, D, 32768))
    return shapes


def run_profile(args):
    inp = _build_inputs(
        args.batch,
        args.next_n,
        args.heads,
        args.head_dim,
        args.kv_len,
        DTYPE_MAP[args.q_dtype],
        block_size=args.kv_block_size,
        seed=args.seed,
        var_ratio=args.var_ratio,
        pool_blocks=args.pool_blocks,
    )
    err, kv_cache_kernel, out = _verify_paged_mqa_logits(
        inp,
        args.batch,
        args.next_n,
        args.head_dim,
        args.split_kv,
        args.kv_block_size,
        args.preshuffle,
        args.variant,
        args.chunk_k,
    )
    _split_kv = None if args.split_kv == 0 else args.split_kv

    def run():
        flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            Preshuffle=args.preshuffle,
            KVBlockSize=args.kv_block_size,
            ChunkK=args.chunk_k,
            SplitKV=_split_kv,
            WavePerEU=args.wave_per_eu,
            variant=args.variant,
        )

    run()
    torch.cuda.synchronize()
    total_ctx = int(inp.context_lens.sum().item())
    kv_bytes = args.next_n * total_ctx * (args.head_dim + 4)
    print(f"# profile correctness err={err:.3e}")
    print(
        f"# gfx={get_gfx()} B={args.batch} nn={args.next_n} H={args.heads} "
        f"D={args.head_dim} kv_len={args.kv_len} kvb={args.kv_block_size} "
        f"preshuffle={args.preshuffle} chunk_k={args.chunk_k} iters={args.iters}"
    )
    print(f"# KV requested: {kv_bytes / 1e6:.1f} MB")
    torch.cuda.synchronize()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
    if args.time:
        _, us = run_perftest(run, num_iters=args.num_iters)
        flops = 2 * args.heads * args.head_dim * args.next_n * total_ctx
        print(
            f"time: {us:.2f} us | {flops / us / 1e6:.1f} TFLOP/s | "
            f"{kv_bytes / us / 1e3:.1f} GB/s requested"
        )


def run_gluon_ab(args):
    rows = []
    for B, nn, H, D, kv_len in gluon_ab_shapes():
        ret = test_fp8_paged_mqa_logits(
            batch_size=B,
            next_n=nn,
            heads=H,
            head_dim=D,
            avg_kv_length=kv_len,
            q_dtype="fnuz",
            block_size=1,
            preshuffle=False,
            bench=True,
        )
        fly_us = ret["flydsl us"]
        ref_us = ret.get("gluon us", float("nan"))
        rows.append(
            {
                **ret,
                "B": B,
                "nn": nn,
                "avg_kv": kv_len,
                "fly/ref": (
                    fly_us / ref_us
                    if not math.isnan(ref_us) and ref_us > 0
                    else float("nan")
                ),
            }
        )
        print(
            f"B={B} nn={nn} kv={kv_len} fly={fly_us:.1f}us gluon={ref_us:.1f}us "
            f"fly/ref={rows[-1]['fly/ref']:.2f}x err={ret['flydsl err']:.2e}",
            flush=True,
        )
    df = pd.DataFrame(rows)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        summary = df.to_string(index=False)
    aiter.logger.info("gluon A/B summary:\n%s", summary)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_logits unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        description="FlyDSL paged fp8_mqa_logits correctness + benchmarks"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="run the full cartesian product instead of the curated matrix",
    )
    parser.add_argument(
        "--no-preshuffle", action="store_true", help="skip preshuffle cases"
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="also run run_perftest timing in the correctness sweep",
    )
    parser.add_argument(
        "--compare-gluon",
        action="store_true",
        help="run the production Gluon A/B decode shape sweep (KVBlockSize=1)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run a single-config timed loop (for rocprofv3 / isolation profiling)",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--next-n", type=int, default=2)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--kv-block-size", type=int, default=1)
    parser.add_argument("--preshuffle", action="store_true")
    parser.add_argument("--q-dtype", type=str, default="fnuz", choices=["fnuz", "fn"])
    parser.add_argument("--split-kv", type=int, default=0)
    parser.add_argument("--wave-per-eu", type=int, default=2)
    parser.add_argument("--chunk-k", type=int, default=128)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--var-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pool-blocks", type=int, default=0)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--num-iters", type=int, default=101)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    if args.profile:
        if args.preshuffle and args.kv_block_size % 16 != 0:
            raise SystemExit("--preshuffle requires --kv-block-size divisible by 16")
        run_profile(args)
        return
    if args.compare_gluon:
        run_gluon_ab(args)
        return

    cases = exhaustive_cases() if args.exhaustive else default_cases()
    if args.no_preshuffle:
        cases = [c for c in cases if not c.get("preshuffle")]

    df = [test_fp8_paged_mqa_logits(bench=args.bench, **c) for c in cases]
    df = pd.DataFrame(df)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        summary = df.to_string(index=False)
    aiter.logger.info("fp8_paged_mqa_logits summary:\n%s", summary)


if __name__ == "__main__":
    main()
