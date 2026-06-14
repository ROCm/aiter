#!/usr/bin/env python3
# Repeat one specific (bs, ctx, qlen) call N times and report when it crashes.
# All input tensors are kept in `history` so the caching allocator can't reuse
# their memory while async kernels are in flight.

import argparse
import sys
import torch
import aiter
from aiter import pertoken_quant


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


def build_call(bs, ctx, qlen, num_blocks, num_kv_heads, num_q_heads,
               head_size, block_size, device, seed=0):
    torch.manual_seed(seed)
    max_seq_len = 16384
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks_per_seq     = (ctx + block_size - 1) // block_size

    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(
        torch.full((bs,), qlen, dtype=torch.int32, device=device), dim=0)
    total_qo = int(qo_indptr[-1].item())
    query = torch.empty(
        (total_qo, num_q_heads, head_size),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
    block_tables = torch.zeros(
        (bs, max_num_blocks_per_seq), dtype=torch.int32, device=device,
    )
    for i in range(bs):
        idx = torch.randint(
            0, num_blocks, (num_blocks_per_seq,), dtype=torch.int32, device=device,
        )
        block_tables[i, :num_blocks_per_seq] = idx
    return query, seq_lens, qo_indptr, block_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=6724)
    ap.add_argument("--qlen", type=int, default=3)
    ap.add_argument("--n-repeat", type=int, default=5)
    ap.add_argument("--num-blocks", type=int, default=8192)
    ap.add_argument("--kv-dtype", choices=["fp8", "bf16"], default="fp8")
    args = ap.parse_args()

    device = "cuda:0"
    torch.set_default_device(device)
    head_size, block_size, num_q_heads, num_kv_heads = 128, 16, 8, 1

    if args.kv_dtype == "fp8":
        k_cache_for_call, v_cache_for_call, k_scale, v_scale = build_kv(
            args.num_blocks, num_kv_heads, head_size, block_size, device,
        )
    else:
        # bf16 path: no quant scales, V still needs ASM shuffle
        x = 16 // 2
        k_cache_for_call = torch.empty(
            (args.num_blocks, num_kv_heads, head_size // x, block_size, x),
            dtype=torch.bfloat16, device=device,
        ).uniform_(-1, 1)
        _v = torch.empty(
            (args.num_blocks, num_kv_heads, head_size, block_size),
            dtype=torch.bfloat16, device=device,
        ).uniform_(-1, 1)
        qx = 16 // _v.element_size()
        v_cache_for_call = (
            _v.view(args.num_blocks, num_kv_heads, head_size, block_size // qx, qx)
            .permute(0, 1, 3, 2, 4).contiguous()
        )
        k_scale = v_scale = None
    torch.cuda.synchronize()

    history = []
    for i in range(args.n_repeat):
        try:
            query, seq_lens, qo_indptr, block_tables = build_call(
                args.bs, args.ctx, args.qlen, args.num_blocks,
                num_kv_heads, num_q_heads, head_size, block_size, device,
                seed=i + 1,
            )
            out = aiter.pa_fwd_asm(
                query,
                k_cache_for_call,
                v_cache_for_call,
                block_tables,
                seq_lens,
                block_tables.stride(0),
                max_qlen=args.qlen,
                K_QScale=k_scale,
                V_QScale=v_scale,
                out_=None,
                qo_indptr=qo_indptr,
                high_precision=0,
            )
            torch.cuda.synchronize()
            history.append((query, seq_lens, qo_indptr, block_tables, out))
        except Exception as e:
            print(f"CRASH at iter={i}  bs={args.bs} ctx={args.ctx} qlen={args.qlen}: "
                  f"{type(e).__name__}: {e}", flush=True)
            return 1

    print(f"ALL OK — {args.n_repeat} calls of bs={args.bs} ctx={args.ctx} "
          f"qlen={args.qlen}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
