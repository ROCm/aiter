#!/usr/bin/env python3
# Clean fp8 ASM PA crash reproducer: exact sequence from stress driver
# (seed=1), no streams, no keepalive games. Every call holds refs locally
# in `history` so input tensors can't be freed under the kernel.
#
# Sequence (recorded from pa_asm_crash_repro.py --seed 1 --kv-dtype fp8):
#   iter 0  bs= 96  ctx=1540  qlen=3  total_qo=288
#   iter 1  bs= 64  ctx=6361  qlen=3  total_qo=192
#   iter 2  bs=128  ctx= 919  qlen=3  total_qo=384
#   iter 3  bs=128  ctx=6724  qlen=3  total_qo=384
#   iter 4  bs=128  ctx= 168  qlen=3  total_qo=384   <- HIP error surfaces here
#
# Run with:
#   AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 \
#       python pa_asm_fp8_seq_repro.py
#
# If `--repeat-only-bad` is set, the script instead calls the iter-3 shape
# (bs=128, ctx=6724, qlen=3) repeatedly to test whether a single bad shape is
# enough vs whether the *sequence* matters.

import argparse
import os
import sys
import torch
import aiter
from aiter import pertoken_quant


def build_call(rng_seed, batch_size, ctx_len, qlen, num_kv_heads, num_q_heads,
               head_size, block_size, num_blocks, device):
    torch.manual_seed(rng_seed)
    max_seq_len = 16384
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks_per_seq     = (ctx_len   + block_size - 1) // block_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_lens_qo = torch.full((batch_size,), qlen, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_qo = int(qo_indptr[-1].item())
    query = torch.empty(
        (total_qo, num_q_heads, head_size),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    seq_lens = torch.full((batch_size,), ctx_len, dtype=torch.int32, device=device)
    block_tables = torch.zeros(
        (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device,
    )
    for i in range(batch_size):
        idx = torch.randint(
            0, num_blocks, (num_blocks_per_seq,), dtype=torch.int32, device=device,
        )
        block_tables[i, :num_blocks_per_seq] = idx
    return dict(query=query, seq_lens=seq_lens, qo_indptr=qo_indptr,
                block_tables=block_tables, max_qlen=qlen,
                bs=batch_size, ctx=ctx_len, qlen=qlen)


def build_kv(num_blocks, num_kv_heads, head_size, block_size, device):
    x = 16 // 2
    k_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size // x, block_size, x),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    v_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size, block_size),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    k_perm = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_kv_heads, block_size, -1).contiguous()
    )
    v_perm = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_kv_heads, block_size, -1).contiguous()
    )
    k_q, k_scale_asm = pertoken_quant(k_perm, quant_dtype=aiter.dtypes.fp8)
    v_q, v_scale_asm = pertoken_quant(v_perm, quant_dtype=aiter.dtypes.fp8)
    quant_x = 16 // aiter.dtypes.fp8.itemsize
    k_quant = (
        k_q.view(num_blocks, num_kv_heads, block_size, head_size // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4).contiguous()
    )
    v_quant = (
        v_q.view(num_blocks, num_kv_heads, block_size, head_size)
        .permute(0, 1, 3, 2).contiguous()
    )
    qx = 16 // v_quant.element_size()
    v_quant_asm = (
        v_quant.view(num_blocks, num_kv_heads, head_size, block_size // qx, qx)
        .permute(0, 1, 3, 2, 4).contiguous()
    )
    return k_quant, v_quant_asm, k_scale_asm, v_scale_asm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat-only-bad", action="store_true",
                    help="Skip iters 0-2, repeat iter-3 shape N times.")
    ap.add_argument("--n-repeat", type=int, default=20)
    ap.add_argument("--num-blocks", type=int, default=8192)
    args = ap.parse_args()

    device = "cuda:0"
    torch.set_default_device(device)

    head_size, block_size, num_q_heads, num_kv_heads = 128, 16, 8, 1
    print(f"[seq-repro] KV pool: num_blocks={args.num_blocks}  GQA=8  "
          f"head_size={head_size}  block_size={block_size}")

    k_quant, v_quant_asm, k_scale, v_scale = build_kv(
        args.num_blocks, num_kv_heads, head_size, block_size, device,
    )
    torch.cuda.synchronize()

    if args.repeat_only_bad:
        seq = [(3 + i, 128, 6724, 3) for i in range(args.n_repeat)]
        print(f"[seq-repro] mode: repeat-only-bad, "
              f"{args.n_repeat}x (bs=128, ctx=6724, qlen=3)")
    else:
        seq = [
            (0,  96, 1540, 3),
            (1,  64, 6361, 3),
            (2, 128,  919, 3),
            (3, 128, 6724, 3),
            (4, 128,  168, 3),
            (5, 128, 1024, 3),  # extras to keep going
            (6,  96, 4096, 4),
        ]

    history = []  # holds all input tensors so they cannot be freed
    for idx, (seed_for_call, bs, ctx, qlen) in enumerate(seq):
        try:
            call = build_call(
                rng_seed=seed_for_call,
                batch_size=bs, ctx_len=ctx, qlen=qlen,
                num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
                head_size=head_size, block_size=block_size,
                num_blocks=args.num_blocks, device=device,
            )
            print(f"[seq-repro] iter={idx}  bs={call['bs']:>4}  "
                  f"ctx={call['ctx']:>5}  qlen={call['qlen']}  "
                  f"-> pa_fwd_asm", flush=True)
            out = aiter.pa_fwd_asm(
                call["query"],
                k_quant,
                v_quant_asm,
                call["block_tables"],
                call["seq_lens"],
                call["block_tables"].stride(0),
                max_qlen=call["max_qlen"],
                K_QScale=k_scale,
                V_QScale=v_scale,
                out_=None,
                qo_indptr=call["qo_indptr"],
                high_precision=0,
            )
            call["out"] = out
            history.append(call)
            torch.cuda.synchronize()
            print(f"[seq-repro]   ok  out={tuple(out.shape)}", flush=True)
        except Exception as e:
            print(f"[seq-repro] !! CRASH at iter={idx}  bs={bs} ctx={ctx} "
                  f"qlen={qlen}: {type(e).__name__}: {e}")
            return 1

    print(f"\n[seq-repro] ALL OK — {len(seq)} calls completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
