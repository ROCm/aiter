# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf sweep for the decode FP8 paged MQA indexer logits kernels.

Candidates are the production Triton/Gluon kernel
(`deepgemm_fp8_paged_mqa_logits`) and the hand-written HIP one
(`aiter.ops.fp8_paged_mqa_logits`), graded against the same fp32 torch reference
on identical inputs.

The HIP kernel only covers nh=32/hd=128 on gfx950 and KVBlockSize in {1, 64}, so
it is added per case rather than unconditionally; its cells stay NaN elsewhere.
"""

import argparse
import itertools
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]

# What the production Triton host uses for this op.
REF_CHUNK_K = 256
REF_WAVE_PER_EU = 2

try:
    from aiter.ops.fp8_paged_mqa_logits import fp8_paged_mqa_logits as hip_paged_logits
    from aiter.ops.fp8_paged_mqa_logits import is_supported as hip_supported
except ImportError:
    hip_paged_logits = None

    def hip_supported(num_heads, head_dim, kv_block_size=1):
        return False


def calc_diff(x, y):
    x, y = x.double(), y.double()
    return 1 - 2 * (x * y).sum() / (x * x + y * y).sum()


def kv_cache_cast_to_fp8(x, fp8_dtype):
    """Co-pack a bf16 KV cache into the fp8+scale byte layout: per block-row,
    block_size*head_dim fp8 bytes then block_size f32 scales, no padding.
    """
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(fp8_dtype)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)), device=x.device, dtype=torch.uint8
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
    layout. Only the kernel gets this copy; the reference reads the unshuffled
    cache, since it is layout-agnostic.
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
    flat[:, : block_size * head_dim] = shuffle_weight(data, layout=(16, 16)).reshape(
        num_blocks, block_size * head_dim
    )
    return flat.view(num_blocks, block_size, one, index_dim)


def run_torch(
    q,
    kv_cache_fp8,
    weights,
    context_lens,
    block_tables,
    max_model_len,
    fp8_dtype,
    block_size,
):
    """Reference only: a torch port of vLLM's ``fp8_paged_mqa_logits_torch``.
    Not timed, not in the table.

    Dequantizes per sequence rather than up front. Materializing the whole cache
    in fp32 costs total_ctx * head_dim * 4 twice over (keys and keys*scales),
    which is several GiB at the long-context shapes; this way the fp32 peak is one
    sequence's worth.
    """
    batch_size, next_n, _heads, dim = q.size()
    num_blocks = kv_cache_fp8.shape[0]
    index_dim = kv_cache_fp8.shape[-1]
    flat = kv_cache_fp8.reshape(num_blocks, block_size * index_dim)
    keys_u8 = flat[:, : block_size * dim].contiguous().view(num_blocks, block_size, dim)
    scales = (
        flat[:, block_size * dim : block_size * dim + 4 * block_size]
        .contiguous()
        .view(dtypes.fp32)
        .view(num_blocks, block_size)
    )
    qf = q.float()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=dtypes.fp32,
    )
    for i in range(batch_size):
        context_len = int(context_lens[i].item())
        if context_len == 0:
            continue
        pos = torch.arange(context_len, device=q.device)
        blk = block_tables[i, pos // block_size]
        tok = pos % block_size
        kx = keys_u8[blk, tok].view(fp8_dtype).float() * scales[blk, tok][:, None]
        s = torch.relu(torch.einsum("nhd,pd->nhp", qf[i], kx))
        wl = weights[i * next_n : (i + 1) * next_n, :]
        s = torch.einsum("nhp,nh->np", s, wl)
        causal = context_len - next_n + torch.arange(next_n, device=q.device)
        s = s.masked_fill(pos[None, :] > causal[:, None], float("-inf"))
        logits[i * next_n : (i + 1) * next_n, :context_len] = s
        del kx, s
    return logits


def _build_inputs(
    batch_size,
    next_n,
    heads,
    head_dim,
    avg_kv_length,
    block_size,
    var_ratio=0.0,
    seed=0,
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
        # Ragged: sequences of different lengths in one batch, which is the normal
        # serving case. It gives every sequence a different tail tile and a
        # different causal boundary, so a kernel that derives its bounds from one
        # sequence -- or from max_model_len -- is caught.
        lo = max(1, int((1 - var_ratio) * avg_kv_length))
        hi = int((1 + var_ratio) * avg_kv_length) + 1
        context_lens = torch.randint(lo, hi, (batch_size,)).cuda().to(torch.int32)
    context_lens = context_lens.clamp(min=next_n)

    blocks_per_seq = (context_lens.to(torch.int64) + block_size - 1) // block_size
    max_block_len = int(blocks_per_seq.max().item())
    num_blocks = int(blocks_per_seq.sum().item())

    q = torch.randn((batch_size, next_n, heads, head_dim), dtype=dtypes.bf16)
    kv_cache = torch.randn((num_blocks, block_size, 1, head_dim), dtype=dtypes.bf16)
    weights = torch.randn((batch_size * next_n, heads), dtype=dtypes.fp32)

    # A shuffled block pool, so a kernel that ignores the block table and walks the
    # cache linearly is caught rather than accidentally right.
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

    return (
        q,
        q.to(fp8_dtype),
        kv_cache_cast_to_fp8(kv_cache, fp8_dtype),
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
    )


@benchmark()
def test_fp8_paged_mqa_logits(
    batch_size, next_n, heads, head_dim, avg_kv_length, block_size, var_ratio
):
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
        block_size,
        var_ratio=var_ratio,
    )

    ref = run_torch(
        q,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        fp8_dtype,
        block_size,
    )
    ref_mask = ref == float("-inf")

    rows = batch_size * next_n
    # Preshuffle is the production pairing: KVBlockSize=1 plain, >1 shuffled. The
    # reference stays on the plain cache either way.
    preshuffle = block_size > 1
    kv_kernel = (
        preshuffle_kv_data(kv_cache_fp8, head_dim) if preshuffle else kv_cache_fp8
    )

    # Both kernels write only the causal window and leave the -inf outside it to the
    # caller, and the Triton launcher writes in place rather than returning, so each
    # candidate owns a pre-filled buffer that is graded after the run. Filling once
    # is enough: every repeat writes the same window.
    outs = {
        name: torch.full((rows, max_model_len), float("-inf"), dtype=dtypes.fp32)
        for name in ("triton", "hip")
    }

    candidates = {
        "triton": lambda: deepgemm_fp8_paged_mqa_logits(
            q_fp8,
            kv_kernel,
            weights,
            outs["triton"],
            context_lens,
            block_tables,
            max_model_len,
            ChunkK=REF_CHUNK_K,
            Preshuffle=preshuffle,
            KVBlockSize=block_size,
            WavePerEU=REF_WAVE_PER_EU,
        ),
    }
    # gfx950-only, nh=32/hd=128-only, KVBlockSize in {1, 64}; NaN cells elsewhere
    # rather than a wrong row.
    if hip_paged_logits is not None and hip_supported(heads, head_dim, block_size):
        candidates["hip"] = lambda: hip_paged_logits(
            q_fp8,
            kv_kernel,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            out=outs["hip"],
        )

    total_ctx = int(context_lens.sum().item())
    flops = 2 * heads * head_dim * next_n * total_ctx
    nbytes = (
        rows * heads * head_dim * q_fp8.element_size()  # Q
        + total_ctx * (head_dim + 4)  # K bytes + co-packed f32 scales
        + rows * heads * 4  # weights
        + rows * max_model_len * 4  # output
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        out = outs[name]
        out_mask = out == float("-inf")
        assert torch.equal(out_mask, ref_mask), (
            f"{name}: -inf mask mismatch, "
            f"{int((out_mask != ref_mask).sum())} of {ref.numel()} positions"
        )
        err = checkAllclose(
            ref.masked_fill(ref_mask, 0).to(dtypes.fp32),
            out.masked_fill(out_mask, 0).to(dtypes.fp32),
            rtol=1e-2,
            atol=5.0,
            msg=f"{name}: fp8_paged_mqa_logits",
            printLog=False,
        )
        diff = calc_diff(out.masked_fill(out_mask, 0), ref.masked_fill(ref_mask, 0))
        assert diff < 1e-3, f"{name}: calc_diff={diff.item():.3e}"

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


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
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1, 4, 16, 64])
    # next_n drives ROWS_PER_BLOCK, which the host clamps to next_n; 4 and 6 are what
    # reach the R=3 instantiation that MTP decode actually runs on.
    parser.add_argument("--next-n", type=int, nargs="*", default=[1, 2, 4, 6])
    parser.add_argument("--heads", type=int, nargs="*", default=[32, 64])
    parser.add_argument("--head-dim", type=int, nargs="*", default=[128])
    parser.add_argument(
        "--kv-len",
        type=int,
        nargs="*",
        default=[1024, 8192, 32768, 131072],
        help="average context length per sequence",
    )
    parser.add_argument("--kv-block-size", type=int, nargs="*", default=[1, 64])
    parser.add_argument(
        "--var-ratio",
        type=float,
        nargs="*",
        default=[0.0, 0.3],
        help="context-length spread within a batch: 0 is uniform, 0.3 draws each\n"
        "sequence from +/-30%% of --kv-len (the normal serving case)",
    )
    args = parser.parse_args()

    df = []
    skipped = []
    for b, nn, h, hd, kv, kvb, vr in itertools.product(
        args.batch,
        args.next_n,
        args.heads,
        args.head_dim,
        args.kv_len,
        args.kv_block_size,
        args.var_ratio,
    ):
        try:
            df.append(test_fp8_paged_mqa_logits(b, nn, h, hd, kv, kvb, vr))
        except torch.OutOfMemoryError:
            # The long-context corner needs several GiB for the cache plus one
            # output buffer per candidate. Record what was dropped rather than
            # letting a short table read as full coverage.
            skipped.append((b, nn, h, hd, kv, kvb, vr))
            torch.cuda.empty_cache()
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fp8_paged_mqa_logits summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if skipped:
        aiter.logger.warning(
            "fp8_paged_mqa_logits: %d of %d cases skipped, out of memory:\n%s",
            len(skipped),
            len(skipped) + len(df),
            "\n".join(
                f"  batch={b} next_n={nn} heads={h} head_dim={hd} kv_len={kv} "
                f"kv_block_size={kvb} var_ratio={vr}"
                for b, nn, h, hd, kv, kvb, vr in skipped
            ),
        )


if __name__ == "__main__":
    main()
