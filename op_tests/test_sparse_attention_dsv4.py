# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch
import triton

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.ops.triton.attention.sparse_attention_dsv4 import (
    sparse_attn_prefill,
    sparse_attn_decode,
)

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# DSV4 sparse-MLA layout: head_dim = NoPE(448) + RoPE(64) = 512.
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM
# fp8_ds_mla paged-cache packing (decode): per-token row = 448 B fp8 NoPE +
# 128 B (64xbf16) RoPE = 576 B, then per-block tail of block_size*8 scale bytes.
NOPE_BLOCK = 512                 # next_power_of_2(448)
NOPE_BYTES = 576
SCALE_BYTES = 8
PER_BLOCK_ROW_BYTES = NOPE_BYTES + SCALE_BYTES  # 584


def build_dense_indices(num_q, max_slots, topk, device):
    """Dense `[num_q, topk]` index matrix (`-1`-padded) plus per-row valid
    lengths, each row keeps a random number of unique slots in [topk // 4, topk].
    This is the format the public `sparse_attn_prefill` / `sparse_attn_decode` 
    interface consumes."""
    lens = torch.randint(
        max(1, topk // 4), topk + 1, (num_q,), dtype=torch.int32, device=device
    )
    dense = torch.full((num_q, topk), -1, dtype=torch.int32, device=device)
    for i in range(num_q):
        L = int(lens[i].item())
        dense[i, :L] = torch.randperm(max_slots, device=device, dtype=torch.int32)[:L]
    return dense, lens


def ragged_from_dense(dense, lens):
    """CSR `(flat, indptr)` from a `-1`-padded dense matrix. The torch-side
    mirror of the wrapper's dense -> ragged conversion, used to build the
    reference output."""
    rows = [dense[i, : int(lens[i].item())] for i in range(dense.shape[0])]
    flat = (
        torch.cat(rows)
        if rows
        else torch.empty(0, dtype=torch.int32, device=dense.device)
    )
    indptr = torch.zeros(dense.shape[0] + 1, dtype=torch.int32, device=dense.device)
    torch.cumsum(lens.to(torch.int32), dim=0, out=indptr[1:])
    return flat, indptr


def ref_sparse_prefill(q, kv, indices, indptr, scale, attn_sink):
    """torch reference: per-query masked softmax attention over gathered KV."""
    kv = kv.squeeze(1)
    out = torch.zeros_like(q, dtype=torch.float32)
    num_q = q.shape[0]
    for t in range(num_q):
        s, e = int(indptr[t]), int(indptr[t + 1])
        if e <= s:
            continue
        slots = indices[s:e].long()
        slots = slots[(slots >= 0) & (slots < kv.shape[0])]
        if slots.numel() == 0:
            continue
        K = kv[slots].float()                       # [L, D]
        sc = (q[t].float() @ K.t()) * scale         # [H, L]
        if attn_sink is not None:
            m = torch.maximum(sc.max(-1).values, attn_sink.float())
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1) + torch.exp(attn_sink.float() - m)
        else:
            m = sc.max(-1).values
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1)
        out[t] = (p / denom[:, None]) @ K
    return out.to(q.dtype)


def run_prefill(q, kv, dense_indices, lens, attn_sink, scale, has_sink):
    out = torch.empty_like(q)
    sparse_attn_prefill(
        q,
        kv,
        dense_indices,
        lens,
        scale,
        HEAD_DIM,
        NOPE_DIM,
        ROPE_DIM,
        attn_sink if has_sink else None,
        out,
    )
    return out


@benchmark()
def test_sparse_prefill(num_queries, num_heads, num_kv, topk, has_sink):
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn(num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv = torch.randn(num_kv, 1, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dense_indices, lens = build_dense_indices(num_queries, num_kv, topk, device)
    indices, indptr = ragged_from_dense(dense_indices, lens)  # reference CSR
    nnz = int(indptr[-1].item())
    scale = 1.0 / (HEAD_DIM ** 0.5)
    if has_sink:
        attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=device)

    out_triton, us = run_perftest(
        run_prefill, q, kv, dense_indices, lens, attn_sink, scale, has_sink
    )
    out_ref = ref_sparse_prefill(
        q, kv, indices, indptr, scale, attn_sink if has_sink else None
    )

    # FLOPs: QK + PV = 4 * H * D per (query, kv-pair). Bytes: Q + gathered KV + Out.
    flops = 4.0 * num_heads * HEAD_DIM * nnz
    nbytes = q.numel() * 2 + nnz * HEAD_DIM * 2 + out_triton.numel() * 2

    err = checkAllclose(
        out_ref.float(),
        out_triton.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"prefill [torch vs triton]: {us:>8.2f} us, "
        f"{flops / us / 1e6:>7.1f} TFLOPS, {nbytes / us / 1e3:>7.1f} GB/s",
    )

    return {
        "nnz": nnz,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "GB/s": nbytes / us / 1e3,
        "err": err,
    }


def build_fp8_cache(num_blocks, block_size, device):
    """Encode bf16 NoPE+RoPE into the 584-byte/row fp8_ds_mla layout the decode
    kernel reads, and also return the dequantized K ([total, HEAD_DIM], fp32)
    exactly as the kernel reconstructs it (fp8 * 2^(enc-127), RoPE bf16)."""
    total = num_blocks * block_size
    nope_bf16 = torch.randn(total, NOPE_DIM, dtype=torch.bfloat16, device=device) * 0.3
    rope_bf16 = torch.randn(total, ROPE_DIM, dtype=torch.bfloat16, device=device) * 0.3

    grp = nope_bf16.float().view(total, NOPE_DIM // 64, 64)
    amax = grp.abs().amax(dim=-1).clamp(min=1e-8)
    exp_v = torch.ceil(torch.log2(amax / 224.0)).clamp(min=-126, max=128) + 127.0
    enc = exp_v.to(torch.uint8)
    sc_f32 = torch.pow(2.0, enc.float() - 127.0)                      # [total, 7]
    fp8 = (grp / sc_f32.unsqueeze(-1)).to(torch.float8_e4m3fn).view(total, NOPE_DIM)

    # Dequantized K as the kernel sees it (for the torch reference).
    deq_nope = (fp8.view(total, NOPE_DIM // 64, 64).float() * sc_f32.unsqueeze(-1)).view(total, NOPE_DIM)
    deq_k = torch.cat([deq_nope, rope_bf16.float()], dim=1)           # [total, HEAD_DIM]

    fp8_bytes = fp8.view(torch.uint8)
    rope_bytes = rope_bf16.view(torch.uint8).view(total, ROPE_DIM * 2)
    cache = torch.zeros(
        num_blocks, block_size, PER_BLOCK_ROW_BYTES, dtype=torch.uint8, device=device
    )
    flat = cache.view(num_blocks, block_size * PER_BLOCK_ROW_BYTES)
    nope_pb = fp8_bytes.view(num_blocks, block_size, NOPE_DIM)
    rope_pb = rope_bytes.view(num_blocks, block_size, ROPE_DIM * 2)
    enc_pb = enc.view(num_blocks, block_size, NOPE_DIM // 64)
    for pos in range(block_size):
        base = pos * NOPE_BYTES
        flat[:, base: base + NOPE_DIM] = nope_pb[:, pos]
        flat[:, base + NOPE_DIM: base + NOPE_BYTES] = rope_pb[:, pos]
    sb0 = block_size * NOPE_BYTES
    for pos in range(block_size):
        sb = sb0 + pos * SCALE_BYTES
        flat[:, sb: sb + (NOPE_DIM // 64)] = enc_pb[:, pos]
    return cache, deq_k


def ref_sparse_decode(
    q, deq_main, idx_main, indptr_main, deq_extra, idx_extra, indptr_extra,
    scale, attn_sink, has_extra,
):
    """torch reference: per-query masked softmax attention over the gathered,
    dequantized latent K (value == key); output dim = NoPE + RoPE = HEAD_DIM."""
    num_q = q.shape[0]
    out = torch.zeros_like(q, dtype=torch.float32)
    for t in range(num_q):
        parts = []
        s, e = int(indptr_main[t]), int(indptr_main[t + 1])
        sl = idx_main[s:e].long()
        sl = sl[(sl >= 0) & (sl < deq_main.shape[0])]
        if sl.numel():
            parts.append(deq_main[sl])
        if has_extra:
            s2, e2 = int(indptr_extra[t]), int(indptr_extra[t + 1])
            sl2 = idx_extra[s2:e2].long()
            sl2 = sl2[(sl2 >= 0) & (sl2 < deq_extra.shape[0])]
            if sl2.numel():
                parts.append(deq_extra[sl2])
        if not parts:
            continue
        K = torch.cat(parts, 0)                       # [L, HEAD_DIM]
        sc = (q[t].float() @ K.t()) * scale           # [H, L]
        if attn_sink is not None:
            m = torch.maximum(sc.max(-1).values, attn_sink.float())
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1) + torch.exp(attn_sink.float() - m)
        else:
            m = sc.max(-1).values
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1)
        out[t] = (p / denom[:, None]) @ K
    return out.to(q.dtype)


def run_decode(
    q, swa_cache, swa_dense, swa_lens, topk_cache, topk_dense, topk_lens,
    attn_sink, scale, has_extra, has_sink,
):
    # Public interface: dense [N, topk] indices + per-row lens for the SWA (main)
    # pass and the optional top-k (extra) pass. The wrapper builds the ragged CSR
    # and dispatches. swa_k_cache is the main pass; kv_cache is the extra pass.
    out = torch.empty_like(q)
    sparse_attn_decode(
        q=q,
        kv_cache=topk_cache if has_extra else None,
        swa_k_cache=swa_cache,
        swa_only=not has_extra,
        topk_indices=topk_dense if has_extra else None,
        topk_lens=topk_lens if has_extra else None,
        swa_indices=swa_dense,
        swa_lens=swa_lens,
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=attn_sink if has_sink else None,
        scale=scale,
        head_dim=HEAD_DIM,
        nope_head_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        output=out,
    )
    return out


@benchmark()
def test_sparse_decode(num_q, num_heads, block_size, num_blocks, topk, has_extra, has_sink):
    torch.manual_seed(0)
    device = "cuda"
    has_extra = bool(has_extra)
    q = torch.randn(num_q, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    main_cache, deq_main = build_fp8_cache(num_blocks, block_size, device)
    main_dense, main_lens = build_dense_indices(
        num_q, num_blocks * block_size, topk, device
    )
    main_idx, main_indptr = ragged_from_dense(main_dense, main_lens)  # reference CSR
    if has_extra:
        extra_cache, deq_extra = build_fp8_cache(num_blocks, block_size, device)
        extra_dense, extra_lens = build_dense_indices(
            num_q, num_blocks * block_size, topk, device
        )
        extra_idx, extra_indptr = ragged_from_dense(extra_dense, extra_lens)
    else:
        extra_cache, deq_extra = main_cache, deq_main
        extra_dense = extra_lens = None
        extra_idx = torch.empty(0, dtype=torch.int32, device=device)
        extra_indptr = torch.zeros(num_q + 1, dtype=torch.int32, device=device)
    scale = 1.0 / (HEAD_DIM ** 0.5)
    if has_sink:
        attn_sink = torch.randn(num_heads, dtype=torch.float32, device=device)
    else:
        attn_sink = torch.empty(1, dtype=torch.float32, device=device)

    out_triton, us = run_perftest(
        run_decode, q, main_cache, main_dense, main_lens,
        extra_cache, extra_dense, extra_lens, attn_sink, scale,
        has_extra, has_sink,
    )
    out_ref = ref_sparse_decode(
        q, deq_main, main_idx, main_indptr, deq_extra, extra_idx, extra_indptr,
        scale, attn_sink if has_sink else None, has_extra,
    )
    err = checkAllclose(
        out_ref.float(), out_triton.float(), atol=1e-2, rtol=1e-2,
        msg=f"decode [torch vs triton]: {us:>8.2f} us",
    )
    return {"us": us, "err": err}


def _parse_cfg(s):
    parts = tuple(int(x) for x in s.split(","))
    assert len(parts) == 4, (
        f"--cfgs expects 'num_queries,num_heads,num_kv,topk', got '{s}'"
    )
    return parts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="DSV4 Gluon sparse-MLA prefill test",
    )
    parser.add_argument(
        "--cfgs",
        type=_parse_cfg,
        nargs="*",
        default=[(4096, 128, 4096, 512)],
        metavar="Q,H,KV,TOPK",
        help="""configs as 'num_queries,num_heads,num_kv,topk'.
    e.g.: --cfgs 4096,128,4096,512""",
    )
    parser.add_argument(
        "--sink",
        action="store_true",
        help="enable the optional per-head attention sink. Default: False.",
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="also run the decode (fp8_ds_mla paged cache) correctness/perf test.",
    )
    parser.add_argument(
        "--decode_cfgs",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        nargs="*",
        default=[(256, 128, 64, 256, 64, 0), (256, 128, 64, 256, 64, 1)],
        metavar="Q,H,BLK,NBLK,TOPK,EXTRA",
        help="decode configs 'num_q,num_heads,block_size,num_blocks,topk,has_extra'.",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "Gluon sparse_attention_dsv4 prefill requires gfx950 (CDNA4); "
            f"current arch is {get_gfx()}. Skipping."
        )
        raise SystemExit(0)

    rows = []
    for num_queries, num_heads, num_kv, topk in args.cfgs:
        rows.append(
            test_sparse_prefill(num_queries, num_heads, num_kv, topk, args.sink)
        )
    df = pd.DataFrame(rows)
    try:
        summary = df.to_markdown(index=False)
    except ImportError:
        # `tabulate` (used by to_markdown) is optional; fall back to plain text.
        summary = df.to_string(index=False)
    aiter.logger.info("sparse_attention_dsv4 prefill summary:\n%s", summary)

    if args.decode:
        drows = []
        for num_q, num_heads, block_size, num_blocks, topk, has_extra in args.decode_cfgs:
            drows.append(
                test_sparse_decode(
                    num_q, num_heads, block_size, num_blocks, topk, has_extra, args.sink
                )
            )
        ddf = pd.DataFrame(drows)
        try:
            dsummary = ddf.to_markdown(index=False)
        except ImportError:
            dsummary = ddf.to_string(index=False)
        aiter.logger.info("sparse_attention_dsv4 decode summary:\n%s", dsummary)
