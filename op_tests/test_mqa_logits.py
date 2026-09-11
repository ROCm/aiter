# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MQA logits (sparse-attention lightning indexer) prefill + decode sweep.

The head geometry is fixed by the model: KV is a single head (the MQA in the
name), head_dim is 128, and Q has 32 or 64 heads. That leaves M (query rows) and
N (KV length) as the axes worth sweeping. Both phases compute

    logits[m, n] = sum_h weights[m, h] * ReLU(<q[m, h, :], k[n, :]>)

inside row m's window and -inf outside.

Prefill and decode are different kernels with different calling conventions, so
they get one table each:

  prefill  fp8_mqa_logits -- contiguous KV [N, 128], Q [M, H, 128], out [M, N].
           M is the chunked-prefill token count, N the context it attends to.
  decode   deepgemm_fp8_paged_mqa_logits -- paged KV cache, Q [B, next_n, H, 128],
           out [B*next_n, N]. Here M = B * next_n and N is the context length.

Q and KV arrive already quantized; neither kernel quantizes. Prefill takes a
plain fp8 cast of Q (its per-(token, head) scale folds into `weights`, which the
kernel multiplies per head) plus an explicit per-token KV scale. Decode reads the
KV scale out of the paged cache, where it is packed behind the data bytes.

The references dequantize the same fp8 bytes the kernels read, so `err` measures
the kernel, not the quantization -- a non-zero `err` column is a real bug.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# The full [H, M, N] fp32 score tensor a naive reference builds is ~69 TB at
# H=32/M=8192/N=65536, so prefill accuracy is always checked on a row sample.
# These bound that sample's [R, H, N] tensor instead.
_REF_SCORE_BYTES = 1 << 30
_REF_MAX_ROWS = 64

# Above this read-side footprint, run_perftest's automatic argument rotation
# (deep-copies of every input, to defeat L2) costs GiBs and buys nothing,
# because the working set already blows past a 4 MB L2.
_ROTATE_MAX_BYTES = 256 << 20


def _pertoken_cast_to_fp8(x, fp8_dtype):
    """Block-fp8 with block size (1, D): one fp32 scale per token row."""
    amax = x.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    sf = amax / torch.finfo(fp8_dtype).max
    return (x / sf).to(fp8_dtype), sf.squeeze(-1).float()


def _rotate_args(read_bytes):
    return 0 if read_bytes < _ROTATE_MAX_BYTES else 1


def _ref_rows(m, num_heads, n):
    """Query rows to check: grid edges and the BLOCK_M=2 block seam first (a
    row-indexing bug lands there), then an even spread over the rest."""
    budget = max(1, _REF_SCORE_BYTES // (num_heads * n * 4))
    want = max(1, min(m, _REF_MAX_ROWS, budget))
    spread = torch.linspace(0, m - 1, steps=want).round().long().tolist()
    rows = []
    for r in [0, 1, m // 2, m // 2 + 1, m - 2, m - 1] + spread:
        if 0 <= r < m and r not in rows:
            rows.append(r)
        if len(rows) >= want:
            break
    return torch.tensor(sorted(rows), dtype=torch.long)


def run_torch_prefill(q_rows, kv, w_rows, ks_rows, ke_rows, seq_len_kv):
    """fp32 reference for a subset of query rows. q_rows/kv are the dequantized
    fp8 values, so the kernel and the reference see identical numbers."""
    scores = torch.einsum("rhd,nd->rhn", q_rows, kv).relu()
    out = (scores * w_rows[:, :, None]).sum(dim=1)
    cols = torch.arange(seq_len_kv)
    inside = (cols[None, :] >= ks_rows[:, None]) & (cols[None, :] < ke_rows[:, None])
    return out, inside


def run_torch_decode(
    q, kv_q, kv_sf, weights, context_lens, block_tables, max_model_len
):
    """fp32 reference for the paged decode indexer, one sequence at a time.

    Dequantizes per sequence rather than materializing the whole cache in fp32:
    at batch=256/N=128K that single tensor would be 17 GiB.
    """
    batch, next_n, _heads, dim = q.shape
    block_size = kv_q.shape[1]
    out = torch.full((batch * next_n, max_model_len), -float("inf"), dtype=dtypes.fp32)
    inside = torch.zeros_like(out, dtype=torch.bool)
    for i in range(batch):
        ctx = int(context_lens[i])
        idx = block_tables[i, : (ctx + block_size - 1) // block_size].long()
        k = (kv_q[idx].float() * kv_sf[idx][..., None]).view(-1, dim)[:ctx]
        scores = torch.einsum("nhd,cd->nhc", q[i], k).relu()
        rows = slice(i * next_n, (i + 1) * next_n)
        row = (scores * weights[rows][:, :, None]).sum(dim=1)
        # the next_n speculative rows sit at absolute positions ctx-next_n .. ctx-1
        causal = torch.arange(ctx)[None, :] <= torch.arange(ctx - next_n, ctx)[:, None]
        out[rows, :ctx] = row.masked_fill(~causal, -float("inf"))
        inside[rows, :ctx] = causal
    return out, inside


def _pack_paged_kv(kv_q, kv_sf, preshuffle):
    """Pack fp8 data + fp32 scales into the paged layout the kernel reads: per
    block, block_size*D data bytes followed by block_size fp32 scales."""
    num_blocks, block_size, dim = kv_q.shape
    data = shuffle_weight(kv_q) if preshuffle else kv_q
    packed = torch.empty((num_blocks, block_size * (dim + 4)), dtype=torch.uint8)
    packed[:, : block_size * dim] = data.reshape(num_blocks, -1).view(torch.uint8)
    packed[:, block_size * dim :] = kv_sf.reshape(num_blocks, -1).view(torch.uint8)
    return packed.view(num_blocks, block_size, 1, dim + 4)


@benchmark()
def test_mqa_logits_prefill(m, n, num_heads, head_dim, clean_logits):
    torch.manual_seed(0)
    fp8_dtype = get_fp8_e4m3_dtype()

    q = torch.randn(m, num_heads, head_dim, dtype=dtypes.bf16)
    kv = torch.randn(n, head_dim, dtype=dtypes.bf16)
    q_fp8 = q.to(fp8_dtype)
    kv_fp8, kv_scales = _pertoken_cast_to_fp8(kv, fp8_dtype)
    weights = torch.randn(m, num_heads, dtype=dtypes.fp32)

    # Chunked prefill: row i is absolute position n-m+i and attends to [0, i].
    ks = torch.zeros(m, dtype=dtypes.i32)
    ke = torch.arange(m, dtype=dtypes.i32) + (n - m) + 1

    rows = _ref_rows(m, num_heads, n)
    ref, inside = run_torch_prefill(
        q_fp8[rows].float(),
        kv_fp8.float() * kv_scales[:, None],
        weights[rows],
        ks[rows],
        ke[rows],
        n,
    )

    candidates = {
        "triton": lambda: fp8_mqa_logits(
            q_fp8, kv_fp8, kv_scales, weights, ks, ke, clean_logits
        ),
    }

    valid = int((ke.clamp(max=n) - ks).clamp(min=0).sum().item())
    n_pad = (n + 255) // 256 * 256
    flops = 2.0 * num_heads * head_dim * valid
    read_bytes = q_fp8.nbytes + kv_fp8.nbytes + kv_scales.nbytes + weights.nbytes
    # Compulsory traffic only. KV is re-read per query row, so once N*128 stops
    # fitting in L2 the achieved bandwidth is well above this number.
    nbytes = read_bytes + (m * n_pad * 4 if clean_logits else valid * 4)

    ret = {"gfx": get_gfx(), "ref_rows": len(rows)}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_rotate_args=_rotate_args(read_bytes))
        got = out[rows]
        if clean_logits:
            assert torch.equal(got == -float("inf"), ~inside), (
                f"{name}: window mask mismatch (m={m}, n={n}, h={num_heads})"
            )
        # clean_logits=False leaves everything outside the window untouched, so
        # only in-window elements are defined either way.
        err = checkAllclose(
            ref[inside],
            got[inside].float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: prefill m={m} n={n} h={num_heads}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mqa_logits_decode(batch, n, num_heads, head_dim, next_n, kv_block):
    torch.manual_seed(0)
    fp8_dtype = get_fp8_e4m3_dtype()

    rows = batch * next_n
    blocks_per_seq = (n + kv_block - 1) // kv_block
    # Every sequence owns its blocks, so the sweep sees realistic cache pressure
    # rather than the L2 hit rate of a shared block pool.
    num_blocks = batch * blocks_per_seq

    q = torch.randn(batch, next_n, num_heads, head_dim, dtype=dtypes.bf16)
    q_fp8 = q.to(fp8_dtype)
    del q

    kv = torch.randn(num_blocks * kv_block, head_dim, dtype=dtypes.bf16)
    kv_q, kv_sf = _pertoken_cast_to_fp8(kv, fp8_dtype)
    del kv
    kv_q = kv_q.view(num_blocks, kv_block, head_dim)
    kv_sf = kv_sf.view(num_blocks, kv_block)

    weights = torch.randn(rows, num_heads, dtype=dtypes.fp32)
    context_lens = torch.full((batch,), n, dtype=dtypes.i32)
    # Blocks are handed out in random order: a real KV cache is fragmented, and a
    # sequential block table would hand the kernel a perfectly coalesced stream
    # it never sees in serving.
    block_tables = torch.randperm(num_blocks).to(dtypes.i32).view(batch, blocks_per_seq)

    ref, inside = run_torch_decode(
        q_fp8.float(), kv_q, kv_sf, weights, context_lens, block_tables, n
    )

    # ATOM's decode indexer calling convention: preshuffled 64-token KV blocks,
    # ChunkK=256, WavePerEU=2, no varctx schedule, caller-owned output buffer.
    kv_cache = _pack_paged_kv(kv_q, kv_sf, preshuffle=True)
    out_logits = torch.full((rows, n), -float("inf"), dtype=dtypes.fp32)

    def run_triton():
        deepgemm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache,
            weights,
            out_logits,
            context_lens,
            block_tables,
            n,
            Preshuffle=True,
            KVBlockSize=kv_block,
            ChunkK=256,
            WavePerEU=2,
        )
        return out_logits

    total_ctx = int(context_lens.sum().item())
    flops = 2.0 * num_heads * head_dim * next_n * total_ctx
    read_bytes = q_fp8.nbytes + total_ctx * (head_dim + 4) + weights.nbytes
    nbytes = read_bytes + rows * n * 4

    ret = {"gfx": get_gfx(), "m": rows}
    for name, fn in {"triton": run_triton}.items():
        out, us = run_perftest(fn, num_rotate_args=_rotate_args(read_bytes))
        err = checkAllclose(
            ref[inside],
            out[inside],
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: decode b={batch} n={n} h={num_heads} mtp={next_n}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _summarize(name, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mqa_logits unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1024, 2048, 4096, 8192, 16384],
        help="""Prefill query rows (chunked-prefill token count).
    e.g.: -m 4096 8192""",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[4096, 16384, 65664, 131072],
        help="""KV length. Shared by both phases; prefill skips n < m.
    e.g.: -n 8192 131072""",
    )
    parser.add_argument(
        "-hq",
        "--num_heads",
        type=int,
        nargs="*",
        default=[32, 64],
        help="""Q heads. 32 and 64 take different kernel configs.
    e.g.: -hq 64""",
    )
    parser.add_argument(
        "-dh",
        "--head_dim",
        type=int,
        nargs="*",
        default=[128],
        help="""Head dim.
    e.g.: -dh 128""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128, 256],
        help="""Decode batch size (concurrency); decode M = batch * next_n.
    e.g.: -b 128 256""",
    )
    parser.add_argument(
        "-mtp",
        "--next_n",
        type=int,
        nargs="*",
        default=[1],
        help="""Decode speculative rows per sequence.
    e.g.: -mtp 1 2""",
    )
    parser.add_argument(
        "-kb",
        "--kv_block",
        type=int,
        nargs="*",
        default=[64],
        help="""Paged KV block size (preshuffle needs a multiple of 16).
    e.g.: -kb 64""",
    )
    parser.add_argument(
        "-c",
        "--clean_logits",
        type=int,
        nargs="*",
        default=[1],
        help="""Prefill: fill the out-of-window logits with -inf in-kernel.
    e.g.: -c 0 1""",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        nargs="*",
        choices=["prefill", "decode"],
        default=["prefill", "decode"],
        help="""Which phases to sweep.
    e.g.: -p prefill""",
    )
    args = parser.parse_args()

    if "prefill" in args.phase:
        rows = []
        for num_heads, head_dim, clean, m, n in itertools.product(
            args.num_heads, args.head_dim, args.clean_logits, args.m, args.n
        ):
            if n < m:
                continue  # a prefill chunk always attends to at least itself
            rows.append(test_mqa_logits_prefill(m, n, num_heads, head_dim, bool(clean)))
            torch.cuda.empty_cache()
        _summarize("mqa_logits prefill", rows)

    if "decode" in args.phase:
        rows = []
        for num_heads, head_dim, next_n, kv_block, batch, n in itertools.product(
            args.num_heads,
            args.head_dim,
            args.next_n,
            args.kv_block,
            args.batch,
            args.n,
        ):
            if n <= next_n:
                continue
            rows.append(
                test_mqa_logits_decode(batch, n, num_heads, head_dim, next_n, kv_block)
            )
            torch.cuda.empty_cache()
        _summarize("mqa_logits decode", rows)


if __name__ == "__main__":
    main()
