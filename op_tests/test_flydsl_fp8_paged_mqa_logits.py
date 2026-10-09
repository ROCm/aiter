# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL paged FP8 MQA logits (decode) vs a torch reference.

The kernel is the gfx950 H=32 / D=128 / KVB=64 preshuffled path the indexer
calls: compact Q ``[B, next_n, 32, 128]``, preallocated ``out_logits``, optional
``next_n_lens``. Torch is the reference only (not timed). Gluon is a candidate
on the uniform compact path.

Shape mapping (decode indexer):
- ``batch`` / ``nq`` = sequences
- ``next_n`` = MTP rows per sequence (compact ``Q.shape[1]`` / ragged pad)
- ``kv_len`` = tokens per sequence (multiple of kvb=64 preferred)

``test_fp8_paged_mqa_logits_probe1079`` is the production decode contract from
the probe_1079 bs64 histogram: 44362-block KV pool, scattered 8 KiB pages,
``max_model_len=258048``, mean Nq=117. The 212-call histogram sweep stays in
the WaveScope driver; this table is the mean-Nq next_n × ctx grid.
"""

from __future__ import annotations

import argparse
import itertools
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
HEADS = 32
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
INDEX_DIM = HEAD_DIM + 4
MAX_NN = 8
WIDE_MAX_MODEL_LEN = 1 << 20
# probe_1079 decode (bs64): 44362 × 64-token pages, output stride 258048.
PROBE_NUM_BLOCKS = 44362
PROBE_MAX_MODEL_LEN = 258048
PROBE_MEAN_NQ = 117
PROBE_SEED = 1079
_E4M3_NATIVE = get_fp8_e4m3_dtype()
_Q_DTYPE = {"fn": torch.float8_e4m3fn}
_PROBE_KV = None

try:
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )
except ImportError:
    deepgemm_fp8_paged_mqa_logits = None


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    return 1 - 2 * (x * y).sum() / denominator


def kv_cache_cast_to_fp8(x, fp8_dtype):
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


def run_torch(
    q,
    kv_cache_fp8,
    weights,
    context_lens,
    block_tables,
    max_model_len,
    fp8_dtype,
    block_size=KV_BLOCK_SIZE,
):
    """Torch reference. Not timed, not in the table."""
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


ref_fp8_paged_mqa_logits = run_torch


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
    block_size=KV_BLOCK_SIZE,
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


def scattered_block_tables(
    nq, pages, num_blocks=PROBE_NUM_BLOCKS, max_model_len=PROBE_MAX_MODEL_LEN
):
    """Random physical pages; consecutive logical pages are not adjacent in HBM."""
    used = nq * pages
    max_blocks_seq = max_model_len // KV_BLOCK_SIZE
    if used <= num_blocks:
        ids = torch.randperm(num_blocks, device="cuda")[:used]
    else:
        reps = (used + num_blocks - 1) // num_blocks
        ids = torch.cat(
            [torch.randperm(num_blocks, device="cuda") for _ in range(reps)]
        )[:used]
    tables = torch.zeros((nq, max_blocks_seq), dtype=torch.int32, device="cuda")
    tables[:, :pages] = ids.reshape(nq, pages).to(torch.int32)
    return tables


def _probe_kv_pool():
    global _PROBE_KV
    if _PROBE_KV is None:
        fp8 = get_fp8_e4m3_dtype()
        kv = torch.randn(
            (PROBE_NUM_BLOCKS, KV_BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16
        )
        raw = kv_cache_cast_to_fp8(kv, fp8)
        _PROBE_KV = (raw, preshuffle_kv_data(raw, HEAD_DIM), fp8)
    return _PROBE_KV


def _build_probe_inputs(nq, next_n, kv_len, q_dtype):
    """Compact decode tensors matching probe_1079 / the indexer call."""
    if kv_len % KV_BLOCK_SIZE:
        raise ValueError(f"kv_len {kv_len} must be a multiple of kvb={KV_BLOCK_SIZE}")
    if kv_len > PROBE_MAX_MODEL_LEN:
        raise ValueError(
            f"kv_len {kv_len} exceeds probe max_model_len {PROBE_MAX_MODEL_LEN}"
        )
    torch.manual_seed(PROBE_SEED)
    random.seed(PROBE_SEED)
    pages = kv_len // KV_BLOCK_SIZE
    kv_raw, kv_kernel, fp8_dtype = _probe_kv_pool()
    q = torch.randn((nq, next_n, HEADS, HEAD_DIM), dtype=torch.bfloat16)
    weights = torch.randn((nq * next_n, HEADS), dtype=torch.float32)
    context_lens = torch.full((nq,), kv_len, device="cuda", dtype=torch.int32)
    tables = scattered_block_tables(nq, pages)
    out = torch.full(
        (nq * next_n, PROBE_MAX_MODEL_LEN),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    return (
        Inputs(
            q,
            q.to(q_dtype),
            kv_raw,
            weights,
            context_lens,
            tables,
            PROBE_MAX_MODEL_LEN,
            fp8_dtype,
        ),
        kv_kernel,
        out,
    )


def _kernel_inputs(inp, batch_size, next_n, head_dim):
    kv_cache_kernel = preshuffle_kv_data(inp.kv_cache_fp8, head_dim)
    out = torch.full(
        (batch_size * next_n, inp.max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    return kv_cache_kernel, out


def sample_next_n_lens(nq: int, max_nn: int = MAX_NN, seed: int = 1079) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + nq)
    return torch.randint(
        1,
        max_nn + 1,
        (nq,),
        generator=g,
        dtype=torch.int32,
        device="cpu",
    )


def padded_logits(batch: int, max_nn: int, max_model_len: int) -> torch.Tensor:
    return torch.full(
        (batch * max_nn, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device="cuda",
    )


def scatter_seq_logits(out, seq, max_nn, row_logits, n_rows):
    dst = seq * max_nn
    out[dst : dst + n_rows].copy_(row_logits[:n_rows])


def ref_padded_ragged(
    q_pad,
    kv_cache_fp8,
    weights_pad,
    context_lens,
    block_tables,
    next_n_lens,
    max_model_len,
    fp8_dtype,
    *,
    max_nn=MAX_NN,
    block_size=KV_BLOCK_SIZE,
):
    batch = q_pad.shape[0]
    out = padded_logits(batch, max_nn, max_model_len)
    for b, n in enumerate(next_n_lens.tolist()):
        n = int(n)
        compact = run_torch(
            q_pad[b : b + 1, :n],
            kv_cache_fp8,
            weights_pad[b * max_nn : b * max_nn + n],
            context_lens[b : b + 1],
            block_tables[b : b + 1],
            max_model_len,
            fp8_dtype,
            block_size=block_size,
        )
        scatter_seq_logits(out, b, max_nn, compact, n)
    return out


def live_row_mask(next_n_lens, max_nn, max_model_len):
    batch = next_n_lens.shape[0]
    rows = torch.arange(max_nn, device=next_n_lens.device)
    live = rows[None, :] < next_n_lens[:, None]
    return live.reshape(batch * max_nn, 1).expand(-1, max_model_len)


def _split_arg(split_kv):
    return None if int(split_kv) <= 0 else int(split_kv)


def _grade(ref, got, msg):
    ref_mask = ref == float("-inf")
    got_mask = got == float("-inf")
    assert torch.equal(got_mask, ref_mask), f"{msg}: -inf mask mismatch"
    live = ~ref_mask
    if not live.any():
        return 0.0
    diff = calc_diff(got.masked_fill(got_mask, 0), ref.masked_fill(ref_mask, 0))
    assert diff < 1e-3, f"{msg} calc_diff={diff}"
    return checkAllclose(
        ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
        got.masked_fill(got_mask, 0).to(dtypes.fp32),
        rtol=1e-2,
        atol=5.0,
        msg=msg,
        printLog=False,
    )


def _roofline(batch, next_n, kv_len, context_lens, q_fp8, weights, out):
    total_ctx = int(context_lens.sum().item())
    flops = 2 * HEADS * HEAD_DIM * next_n * total_ctx
    nbytes = (
        batch * next_n * HEADS * HEAD_DIM * q_fp8.element_size()
        + total_ctx * INDEX_DIM
        + batch * next_n * HEADS * weights.element_size()
        + batch * next_n * kv_len * out.element_size()
    )
    return flops, nbytes


def _time_candidates(candidates, ref, flops, nbytes, msg):
    ret = {"gfx": get_gfx()}
    for name, (fn, buf) in candidates.items():
        with torch.inference_mode():
            _, us = run_perftest(fn)
        err = _grade(ref, buf, f"{name}: {msg}")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_fp8_paged_mqa_logits(batch, next_n, kv_len, split_kv, dtype):
    q_dtype = _Q_DTYPE[dtype]
    inp = _build_inputs(
        batch, next_n, HEADS, HEAD_DIM, kv_len, q_dtype, block_size=KV_BLOCK_SIZE
    )
    kv_cache_kernel, out = _kernel_inputs(inp, batch, next_n, HEAD_DIM)
    ref = run_torch(
        inp.q,
        inp.kv_cache_fp8,
        inp.weights,
        inp.context_lens,
        inp.block_tables,
        inp.max_model_len,
        inp.fp8_dtype,
        block_size=KV_BLOCK_SIZE,
    )
    split = _split_arg(split_kv)

    def flydsl():
        out.fill_(float("-inf"))
        return flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
            SplitKV=split,
        )

    candidates = {"flydsl": (flydsl, out)}
    # Gluon is a real kernel under test, not the torch reference. It only
    # accepts a launch-wide next_n = Q.shape[1], so it is compact-only.
    if deepgemm_fp8_paged_mqa_logits is not None:
        out_g = out.clone()

        def gluon():
            out_g.fill_(float("-inf"))
            return deepgemm_fp8_paged_mqa_logits(
                inp.q_fp8,
                kv_cache_kernel,
                inp.weights,
                out_g,
                inp.context_lens,
                inp.block_tables,
                inp.max_model_len,
                ChunkK=256,
                Preshuffle=True,
                KVBlockSize=KV_BLOCK_SIZE,
                WavePerEU=2,
            )

        candidates["gluon"] = (gluon, out_g)

    flops, nbytes = _roofline(
        batch, next_n, kv_len, inp.context_lens, inp.q_fp8, inp.weights, out
    )
    return _time_candidates(candidates, ref, flops, nbytes, "paged fp8_mqa_logits")


@benchmark()
def test_fp8_paged_mqa_logits_probe1079(nq, next_n, kv_len, dtype):
    """Mean-Nq decode: scattered 44362-block pool, production output stride."""
    q_dtype = _Q_DTYPE[dtype]
    inp, kv_cache_kernel, out = _build_probe_inputs(nq, next_n, kv_len, q_dtype)
    scored = out[:, :kv_len]
    ref = run_torch(
        inp.q,
        inp.kv_cache_fp8,
        inp.weights,
        inp.context_lens,
        inp.block_tables,
        kv_len,
        inp.fp8_dtype,
        block_size=KV_BLOCK_SIZE,
    )

    def flydsl():
        out.fill_(float("-inf"))
        flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            PROBE_MAX_MODEL_LEN,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
        )
        return scored

    candidates = {"flydsl": (flydsl, scored)}
    if deepgemm_fp8_paged_mqa_logits is not None:
        out_g = out.clone()
        scored_g = out_g[:, :kv_len]

        def gluon():
            out_g.fill_(float("-inf"))
            deepgemm_fp8_paged_mqa_logits(
                inp.q_fp8,
                kv_cache_kernel,
                inp.weights,
                out_g,
                inp.context_lens,
                inp.block_tables,
                PROBE_MAX_MODEL_LEN,
                ChunkK=256,
                Preshuffle=True,
                KVBlockSize=KV_BLOCK_SIZE,
                WavePerEU=2,
            )
            return scored_g

        candidates["gluon"] = (gluon, scored_g)

    flops, nbytes = _roofline(
        nq, next_n, kv_len, inp.context_lens, inp.q_fp8, inp.weights, scored
    )
    return _time_candidates(candidates, ref, flops, nbytes, "paged probe1079")


@benchmark()
def test_fp8_paged_mqa_logits_ragged(batch, next_n, kv_len, split_kv):
    inp = _build_inputs(
        batch, next_n, HEADS, HEAD_DIM, kv_len, _E4M3_NATIVE, block_size=KV_BLOCK_SIZE
    )
    kv_cache_kernel, out = _kernel_inputs(inp, batch, next_n, HEAD_DIM)
    next_n_lens = sample_next_n_lens(batch, next_n).cuda()
    ref = ref_padded_ragged(
        inp.q,
        inp.kv_cache_fp8,
        inp.weights,
        inp.context_lens,
        inp.block_tables,
        next_n_lens,
        inp.max_model_len,
        inp.fp8_dtype,
        max_nn=next_n,
        block_size=KV_BLOCK_SIZE,
    )
    split = _split_arg(split_kv)
    live = int(next_n_lens.sum().item())

    def flydsl():
        out.fill_(float("-inf"))
        return flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            out,
            inp.context_lens,
            inp.block_tables,
            inp.max_model_len,
            next_n_lens=next_n_lens,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
            SplitKV=split,
        )

    # Gluon omitted: it has no next_n_lens and would score padded rows.
    flops, nbytes = _roofline(
        batch,
        live / max(batch, 1),
        kv_len,
        inp.context_lens,
        inp.q_fp8,
        inp.weights,
        out,
    )
    ret = _time_candidates(
        {"flydsl": (flydsl, out)}, ref, flops, nbytes, "paged ragged"
    )
    unused = ~live_row_mask(next_n_lens, next_n, inp.max_model_len)[:, 0]
    if unused.any():
        assert torch.all(ref[unused] == float("-inf"))
        assert torch.all(out[unused] == float("-inf"))
    return ret


@benchmark()
def test_fp8_paged_mqa_logits_wide_out(batch, kv_len):
    """Padded output row 512 crosses the signed i32 byte-offset boundary."""
    next_n = MAX_NN
    inp = _build_inputs(
        batch, next_n, HEADS, HEAD_DIM, kv_len, _E4M3_NATIVE, block_size=KV_BLOCK_SIZE
    )
    kv_cache_kernel, _compact = _kernel_inputs(inp, batch, next_n, HEAD_DIM)
    next_n_lens = torch.ones(batch, dtype=torch.int32, device="cuda")
    wide = torch.full(
        (batch * next_n, WIDE_MAX_MODEL_LEN),
        float("-inf"),
        dtype=torch.float32,
    )
    ref = ref_padded_ragged(
        inp.q,
        inp.kv_cache_fp8,
        inp.weights,
        inp.context_lens,
        inp.block_tables,
        next_n_lens,
        inp.max_model_len,
        inp.fp8_dtype,
        max_nn=next_n,
        block_size=KV_BLOCK_SIZE,
    )
    split = 1

    scored = wide[:, : inp.max_model_len]

    def flydsl():
        wide.fill_(float("-inf"))
        flydsl_fp8_paged_mqa_logits(
            inp.q_fp8,
            kv_cache_kernel,
            inp.weights,
            wide,
            inp.context_lens,
            inp.block_tables,
            WIDE_MAX_MODEL_LEN,
            next_n_lens=next_n_lens,
            Preshuffle=True,
            KVBlockSize=KV_BLOCK_SIZE,
            SplitKV=split,
        )
        return scored

    flops, nbytes = _roofline(
        batch, 1, kv_len, inp.context_lens, inp.q_fp8, inp.weights, wide
    )
    return _time_candidates(
        {"flydsl": (flydsl, scored)}, ref, flops, nbytes, "paged wide_out"
    )


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)
    aiter.logger.info("%s summary (markdown):\n%s", name, table)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_logits unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["fn"],
        choices=list(_Q_DTYPE),
        help="Q dtype tag. gfx950 uses native E4M3 FN.\n    e.g.: -d fn",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8],
        help="Batch (sequences).\n    e.g.: -b 4",
    )
    parser.add_argument(
        "--next-n",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8],
        help="Compact Q dim-1 / ragged pad.\n    e.g.: --next-n 1 8",
    )
    parser.add_argument(
        "--kv-len",
        type=int,
        nargs="*",
        default=[1024],
        help="Tokens per sequence (multiple of kvb=64 preferred).\n    e.g.: --kv-len 128 8192",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=[0],
        help="SplitKV; 0 = auto occupancy mapping.\n    e.g.: --split-kv 0 1 4",
    )
    parser.add_argument(
        "--ragged-batch",
        type=int,
        nargs="*",
        default=[1, 8, 24],
        help="Batch for the ragged table.",
    )
    parser.add_argument(
        "--ragged-next-n",
        type=int,
        nargs="*",
        default=[8],
        help="Ragged pad (max next_n).",
    )
    parser.add_argument(
        "--ragged-kv-len",
        type=int,
        nargs="*",
        default=[128, 192, 256, 448],
        help="Context for the ragged table (2/3/4/7 pages of kvb=64).",
    )
    parser.add_argument(
        "--ragged-split-kv",
        type=int,
        nargs="*",
        default=[0, 3],
        help="SplitKV for the ragged table.",
    )
    parser.add_argument(
        "--long-batch",
        type=int,
        nargs="*",
        default=[16],
        help="Batch for the long-context ragged table.",
    )
    parser.add_argument(
        "--long-kv-len",
        type=int,
        nargs="*",
        default=[32768],
        help="Context for the long-context ragged table.",
    )
    parser.add_argument(
        "--wide-batch",
        type=int,
        nargs="*",
        default=[65],
        help="Batch for the wide-output i32-offset table.",
    )
    parser.add_argument(
        "--hist-nq",
        type=int,
        nargs="*",
        default=[PROBE_MEAN_NQ],
        help="Nq for the probe_1079 table (mean of the 212-call histogram is 117).\n"
        "    e.g.: --hist-nq 16 117 256",
    )
    parser.add_argument(
        "--hist-next-n",
        type=int,
        nargs="*",
        default=list(range(1, MAX_NN + 1)),
        help="Compact next_n for the probe_1079 table.\n    e.g.: --hist-next-n 1 8",
    )
    parser.add_argument(
        "--hist-kv-len",
        type=int,
        nargs="*",
        default=[8192],
        help="Tokens/seq for the probe_1079 table (scattered 8 KiB pages).\n"
        "    e.g.: --hist-kv-len 8192 131072",
    )
    args = parser.parse_args()

    compact = [
        test_fp8_paged_mqa_logits(batch, next_n, kv_len, split_kv, dtype)
        for dtype, batch, next_n, kv_len, split_kv in itertools.product(
            args.dtype, args.batch, args.next_n, args.kv_len, args.split_kv
        )
    ]
    if compact:
        _summarize("fp8_paged_mqa_logits", compact)

    ragged = [
        test_fp8_paged_mqa_logits_ragged(batch, next_n, kv_len, split_kv)
        for batch, next_n, kv_len, split_kv in itertools.product(
            args.ragged_batch,
            args.ragged_next_n,
            args.ragged_kv_len,
            args.ragged_split_kv,
        )
    ]
    if ragged:
        _summarize("fp8_paged_mqa_logits_ragged", ragged)

    long_ctx = [
        test_fp8_paged_mqa_logits_ragged(batch, next_n, kv_len, split_kv)
        for batch, next_n, kv_len, split_kv in itertools.product(
            args.long_batch, args.ragged_next_n, args.long_kv_len, args.split_kv
        )
    ]
    if long_ctx:
        _summarize("fp8_paged_mqa_logits_ragged_long", long_ctx)

    wide = [
        test_fp8_paged_mqa_logits_wide_out(batch, KV_BLOCK_SIZE)
        for batch in args.wide_batch
    ]
    if wide:
        _summarize("fp8_paged_mqa_logits_wide_out", wide)

    hist = [
        test_fp8_paged_mqa_logits_probe1079(nq, next_n, kv_len, dtype)
        for dtype, nq, next_n, kv_len in itertools.product(
            args.dtype, args.hist_nq, args.hist_next_n, args.hist_kv_len
        )
    ]
    if hist:
        _summarize("fp8_paged_mqa_logits_probe1079", hist)


if __name__ == "__main__":
    main()
