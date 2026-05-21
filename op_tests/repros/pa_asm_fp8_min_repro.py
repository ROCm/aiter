#!/usr/bin/env python3
# Minimal aiter-only single-call reproducer for the fp8 ASM PA crash.
#
# This is *single call* of aiter.pa_fwd_asm — no streams, no loop, no race.
# Reproduces an HIP illegal memory access purely from one bad shape.
#
# Discovery path:
#   The stress driver (pa_asm_crash_repro.py) crashed at iter 3 of seed=1
#   with: batch_size=128, ctx_len=6724, qlen=3, GQA=8, head_dim=128,
#   block_size=16, fp8 KV per-token quant.
#   This script isolates exactly that call.
#
# Usage:
#   python pa_asm_fp8_min_repro.py
#
# Knobs (env var):
#   PA_REPRO_PAD=1   pad block_tables to max_num_blocks_per_seq=1024 (default).
#   PA_REPRO_PAD=0   use tight block_tables of (batch_size, num_blocks_per_seq).
#   PA_REPRO_BS=N    override batch_size (default 128).
#   PA_REPRO_CTX=N   override ctx_len (default 6724).
#   PA_REPRO_QLEN=N  override qlen   (default 3).
#   PA_REPRO_NBLOCKS=N override KV pool num_blocks (default 8192).
#
# Expected on a buggy build (current aiter ee28d47ac + PR3211 aff40475d):
#   [AITER] hipModuleLaunchKernel failed -> HIP illegal memory access
#   (kernel: pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co)

import os
import sys
import torch
import aiter
from aiter import pertoken_quant


def main():
    torch.manual_seed(0)
    device = "cuda:0"
    torch.set_default_device(device)

    # ---- shape (matches Kimi-K2.5 MLA-via-MHA, eagle3 MTP, TP=8 per-rank) ----
    head_size   = 128
    block_size  = 16
    num_q_heads = 8        # per-rank Q heads (TP=8 on 64 heads)
    num_kv_heads = 1       # MLA absorbs to 1 latent KV head -> GQA=8

    batch_size = int(os.environ.get("PA_REPRO_BS",  128))
    ctx_len    = int(os.environ.get("PA_REPRO_CTX", 6724))
    qlen       = int(os.environ.get("PA_REPRO_QLEN", 3))
    num_blocks = int(os.environ.get("PA_REPRO_NBLOCKS", 8192))
    pad_bt     = bool(int(os.environ.get("PA_REPRO_PAD", "1")))

    max_seq_len = 16384
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size  # 1024
    num_blocks_per_seq     = (ctx_len   + block_size - 1) // block_size

    print(f"[min-repro] batch={batch_size} ctx_len={ctx_len} qlen={qlen} "
          f"num_blocks={num_blocks} pad_block_tables={pad_bt}")
    print(f"[min-repro] num_blocks_per_seq={num_blocks_per_seq}  "
          f"max_num_blocks_per_seq={max_num_blocks_per_seq}")

    # ---- allocate KV cache (bf16) and pertoken-quant it to fp8 ASM layout ----
    x = 16 // 2  # bf16 itemsize
    k_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size // x, block_size, x),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    v_cache = torch.empty(
        (num_blocks, num_kv_heads, head_size, block_size),
        dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)

    # pertoken quant -> fp8
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
    # ASM V shuffle: [B, KVH, head_size, block_size] -> [B, KVH, block_size/x, head_size, x]
    qx = 16 // v_quant.element_size()
    v_quant_asm = (
        v_quant.view(num_blocks, num_kv_heads, head_size, block_size // qx, qx)
        .permute(0, 1, 3, 2, 4).contiguous()
    )

    # ---- per-iter request inputs ----
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    seq_lens_qo = torch.full((batch_size,), qlen, dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_qo = int(qo_indptr[-1].item())
    query = torch.empty(
        (total_qo, num_q_heads, head_size), dtype=torch.bfloat16, device=device,
    ).uniform_(-1, 1)
    seq_lens = torch.full((batch_size,), ctx_len, dtype=torch.int32, device=device)

    if pad_bt:
        block_tables = torch.zeros(
            (batch_size, max_num_blocks_per_seq), dtype=torch.int32, device=device,
        )
    else:
        block_tables = torch.zeros(
            (batch_size, num_blocks_per_seq), dtype=torch.int32, device=device,
        )
    for i in range(batch_size):
        idx = torch.randint(
            0, num_blocks, (num_blocks_per_seq,), dtype=torch.int32, device=device,
        )
        block_tables[i, :num_blocks_per_seq] = idx

    print(f"[min-repro] query={tuple(query.shape)}  qo_indptr[-1]={total_qo}")
    print(f"[min-repro] block_tables={tuple(block_tables.shape)}  "
          f"stride0={block_tables.stride(0)}")
    print(f"[min-repro] k_quant={tuple(k_quant.shape)}  "
          f"v_quant_asm={tuple(v_quant_asm.shape)}")
    print(f"[min-repro] K_QScale={tuple(k_scale_asm.shape)} {k_scale_asm.dtype}")
    torch.cuda.synchronize()

    # ---- single call ----
    print(f"[min-repro] calling pa_fwd_asm ...", flush=True)
    out = aiter.pa_fwd_asm(
        query,
        k_quant,
        v_quant_asm,
        block_tables,
        seq_lens,
        block_tables.stride(0),
        max_qlen=qlen,
        K_QScale=k_scale_asm,
        V_QScale=v_scale_asm,
        out_=None,
        qo_indptr=qo_indptr,
        high_precision=0,
    )
    torch.cuda.synchronize()
    print(f"[min-repro] OK -> out={tuple(out.shape)} {out.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
