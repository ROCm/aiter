# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the DeepSeek-V4 Q/KV producer and the standalone Q pack.

fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert does, per token, a per-head
RMSNorm and GPT-J RoPE on Q, and RoPE + UE8M0 FP8 quant + paged insert on KV.
With ``pack_q`` it also emits the a8w8 Q pair off the same registers, which is
the alternative to calling fused_deepseek_v4_mxfp8_quant_q_pack as a second
pass. Benching all three together is what says whether fusing the pack is
worth it -- the question VLLM_DSV4_QPACK_FUSION exists to answer:

    pack=on            producer emits the pair itself
    pack=off + q_pack  producer writes bf16 Q, a second kernel packs it

    python bench_fused_deepseek_v4_qnorm_rope_kv_insert.py --tokens 256 1024

``--no-write-q`` measures the pure-decode shortcut, where the bf16 Q is never
read back and the producer may skip writing it.
"""
import argparse
import sys

import torch
import triton

from aiter.ops.triton.quant.fused_mxfp8_quant import (
    fused_deepseek_v4_mxfp8_quant_q_pack,
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
)

_QK, _NOPE, _ROPE = 512, 448, 64
_REC = 640  # the producer writes the aligned record only


def build(tokens, heads, padded, nb, block, device="cuda"):
    torch.manual_seed(0)
    q = (torch.randn(tokens, heads, _QK, device=device) * 0.125).to(torch.bfloat16)
    kv = (torch.randn(tokens, _QK, device=device) * 0.4).to(torch.bfloat16)
    cache = torch.zeros(nb, block, _REC, dtype=torch.uint8, device=device)
    slot = torch.randperm(nb * block, device=device, dtype=torch.int32)[:tokens]
    pos = torch.randint(0, 128, (tokens,), device=device, dtype=torch.int64)
    cos_sin = torch.randn(256, _ROPE, device=device, dtype=torch.float32)
    return q, kv, cache, slot, pos, cos_sin


def producer_bytes(tokens, heads, padded, pack, write_q):
    """read Q and KV; write the bf16 Q, the KV record, and maybe the pair."""
    n = tokens * heads * _QK * 2 + tokens * _QK * 2  # read
    n += tokens * _REC  # the KV record
    if write_q:
        n += tokens * padded * _QK * 2
    if pack:
        n += tokens * padded * _QK + tokens * padded * _ROPE * 2
    return n


def pack_bytes(rows):
    """read bf16 Q, write the fp8 record plus the untouched bf16 RoPE."""
    return rows * (_QK * 2 + _QK + _ROPE * 2)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark the DSv4 Q/KV producer and Q pack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 64, 256, 1024])
    p.add_argument("--heads", type=int, default=16, help="local Q heads")
    p.add_argument("--padded-heads", type=int, default=16,
                   help="head count the attention kernel wants; >= --heads")
    p.add_argument("--block", type=int, default=64, help="paged cache block size")
    p.add_argument("--no-q-norm", dest="q_norm", action="store_false",
                   help="skip the per-head RMSNorm on Q")
    p.add_argument("--no-write-q", dest="write_q", action="store_false",
                   help="do not write the bf16 Q (pure-decode shortcut)")
    p.add_argument("--rep", type=int, default=20,
                   help="do_bench_cudagraph measurement target, in ms")
    a = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    if a.padded_heads < a.heads:
        sys.exit("--padded-heads must be >= --heads")

    print("dsv4 producer  heads=%d padded=%d block=%d q_norm=%s write_q=%s"
          % (a.heads, a.padded_heads, a.block, a.q_norm, a.write_q))
    print("%-8s %-11s %-11s %-11s %-11s %s"
          % ("tokens", "pack=on us", "pack=off us", "q_pack us",
             "fused GB/s", "fused vs split"))

    for t in a.tokens:
        nb = max(1, (t + a.block - 1) // a.block * 2)
        q, kv, cache, slot, pos, cs = build(t, a.heads, a.padded_heads, nb, a.block)
        args = (q, kv, cache, slot, pos, cs, a.block, 1e-6, a.padded_heads)
        kw = dict(apply_q_norm=a.q_norm, write_q=a.write_q)

        def fused():
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(*args, pack_q=True, **kw)

        def unfused():
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(*args, pack_q=False, **kw)

        q_out = fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            *args, pack_q=False, **kw
        )

        def only_pack():
            fused_deepseek_v4_mxfp8_quant_q_pack(q_out)

        for fn in (fused, unfused, only_pack):  # compile outside the timed region
            fn()
        torch.cuda.synchronize()

        f = triton.testing.do_bench_cudagraph(fused, rep=a.rep) * 1e3
        u = triton.testing.do_bench_cudagraph(unfused, rep=a.rep) * 1e3
        k = triton.testing.do_bench_cudagraph(only_pack, rep=a.rep) * 1e3
        gbs = producer_bytes(t, a.heads, a.padded_heads, True, a.write_q) / (f * 1e-6) / 1e9
        print("%-8d %-11.2f %-11.2f %-11.2f %-11.1f %.2fx"
              % (t, f, u, k, gbs, (u + k) / f))


if __name__ == "__main__":
    main()
