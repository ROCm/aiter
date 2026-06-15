# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the gemm2 atomic A_scale buffer-resource bound at BM=64.

The bug
-------
A_scale is laid out in fixed 32-row chunks; the read side addresses it with
`chunk_base = (BM==16) ? m_row/BM_GRID : m_row/32` (i.e. divisor 16 only for
BM=16, else 32 — NOT BM). The rsrc *bound* divided by `kAtomic ? BM_GRID : 32`,
which for BM=64 (atomic) is 64 — 2x smaller than the 32 the addressing uses. So
the descriptor only covers max_sorted/64 chunks while addressing reaches
max_sorted/32: every sorted row with m_row >= max_sorted/2 fails the hw bounds
check, the scale load returns 0 -> e8m0=0 -> scale 2^-127 -> those output rows
collapse to ~0. Silent wrong results.

(This only became reachable once the bound was tightened from the old fixed
MAX_M=655360 to the runtime max_sorted; with MAX_M the /64 bound was still huge
enough to cover everything, so the off-by-2x was masked.)

How this test catches it
------------------------
gemm2's per-row math is independent of the M-tile size, and A_q / A_scale use
the SAME byte layout for BM=32 and BM=64 (both address scale in 32-row chunks).
So running the same inputs through the BM=32 kernel (correct bound) and the BM=64
kernel (the buggy one) must agree. We size M_logical/cumsum so the BM=64 sorted
length exceeds the (buggy) max_sorted/2, i.e. the upper rows fall in the clipped
region. The atomic epilog is run-to-run nondeterministic (atomicAdd FP order), so
we measure that noise floor from BM=32-vs-BM=32 and only count a token as a REAL
break where BM=32 is self-consistent yet BM=32 != BM=64.

  fixed bound  -> 0 real breaks (PASS)
  buggy bound  -> many tokens collapse in the BM=64 run (FAIL)

Run on GPU5 (needs module_moe_mxfp4_gemm built from this checkout):
    HIP_VISIBLE_DEVICES=5 PYTHONPATH=/home/yankui87@sjtu.edu.cn/up-aiter \
        python3 op_tests/test_mxfp4_moe_gemm2_bm64_ascale_bound.py
"""

import argparse
import torch

import aiter
from aiter import dtypes

print("[selfcheck] aiter pkg:", aiter.__file__)
torch.set_default_device("cuda")

NE = 385
D_HIDDEN = 7168          # gemm2 N_OUT
D_INTER = 512            # gemm2 K
TOPK = 9
KN = "mxfp4_moe_g2_a4w4_NE{ne}_H{h}_E{e}_TOPK{tk}_BM{bm}_ATOMIC"


def build(cumsum, M_logical, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    # fp4 nibbles forced NON-NEGATIVE (clear each nibble's sign bit, &0x77) so every
    # A*B product is >= 0: the per-token atomic sum has no cancellation, making it
    # order-insensitive (tiny FP noise) and the clipped-row signal unambiguous.
    pos4 = lambda *s: (torch.randint(0, 256, s, dtype=torch.uint8, generator=g) & 0x77)
    e8 = lambda *s: torch.randint(126, 130, s, dtype=torch.uint8, generator=g)  # tame e8m0
    # gemm2 A_q (inter) and its 32-row-chunk e8m0 scale. Scale is over-allocated so
    # the DATA exists for every row — only the kernel's rsrc bound gates the read.
    inter_q = pos4(cumsum, D_INTER // 2)
    inter_s = e8(cumsum, 512)                       # >> (cumsum/32)*kAS_per_chunk_dw*4 bytes
    w2 = pos4(NE, D_HIDDEN, D_INTER // 2)
    w2s = e8(NE, D_HIDDEN, D_INTER // 32)
    tok = torch.randint(0, M_logical, (cumsum,), dtype=torch.int32, generator=g)
    wts = torch.rand((cumsum,), dtype=torch.float32, generator=g)
    eid64 = torch.randint(0, NE, (cumsum // 64,), dtype=torch.int32, generator=g)
    eid32 = eid64.repeat_interleave(2)             # per-row expert identical to BM=64
    cs = torch.tensor([cumsum], dtype=torch.int32)
    return dict(inter_q=inter_q, inter_s=inter_s, w2=w2, w2s=w2s, tok=tok,
                wts=wts, eid32=eid32, eid64=eid64, cs=cs)


def run(d, bm, M_logical, cumsum):
    eid = d["eid64"] if bm == 64 else d["eid32"]
    out = torch.zeros((M_logical, D_HIDDEN), dtype=dtypes.bf16)  # atomic-add target
    aiter.mxfp4_moe_gemm2_a4w4(
        cumsum_tensor=d["cs"],
        inter_sorted_quant=d["inter_q"],
        inter_sorted_shuffled_scale=d["inter_s"],
        w3_shuffled_quant=d["w2"],
        w3_shuffled_scale=d["w2s"],
        sorted_token_ids=d["tok"],
        sorted_expert_ids=eid,
        sorted_weights=d["wts"],
        flat_out=out,
        M_logical=M_logical,
        max_sorted=cumsum,
        kernelName=KN.format(ne=NE, h=D_HIDDEN, e=D_INTER, tk=TOPK, bm=bm),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    # M_logical=8192 -> derived max_sorted(BM64)~97984; cumsum=73728 (=8192*9, mult of 64)
    # lies in (max_sorted/2, derived_BM32) so buggy-BM64 clips while BM32 + fixed-BM64 are OK.
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--cumsum", type=int, default=73728)
    ap.add_argument("--atol", type=float, default=1e-1)
    ap.add_argument("--rtol", type=float, default=5e-2)
    args = ap.parse_args()
    M, C = args.tokens, args.cumsum
    assert C % 64 == 0, "cumsum must be a multiple of 64"
    print(f"M_logical={M}  cumsum={C}  (BM=64 buggy bound clips rows >= ~max_sorted/2)")

    d = build(C, M)
    ref_a = run(d, 32, M, C)     # correct reference
    ref_b = run(d, 32, M, C)     # atomic-noise floor
    test = run(d, 64, M, C)      # the bound under test
    torch.cuda.synchronize()

    def bad(a, b):
        a, b = a.float(), b.float()
        return ((a - b).abs() > args.atol + args.rtol * b.abs()).any(dim=1)

    noise = bad(ref_a, ref_b)
    diff = bad(ref_a, test)
    real = diff & ~noise
    di = (ref_a.float() - test.float()).abs()
    print(f"[BM32 vs BM32 noise] rows={int(noise.sum())}/{M}")
    print(f"[BM32 vs BM64      ] rows={int(diff.sum())}/{M}  max|Δ|={di.max().item():.4e}")
    print(f"[REAL breaks (BM32 deterministic, yet != BM64)] = {int(real.sum())}/{M}")
    ok = int(real.sum()) == 0
    if not ok:
        idx = real.nonzero().flatten()[:8].tolist()
        r0 = idx[0]
        print(f"    rows: {idx}")
        print(f"    e.g. token {r0}: |bm32|={ref_a.float()[r0].abs().mean():.3e} "
              f"|bm64|={test.float()[r0].abs().mean():.3e} (collapsed if ~0)")
    print("RESULT:", "PASS ✅ (BM64 A_scale bound matches BM32)"
          if ok else "FAIL ❌ (BM64 rows collapse → A_scale bound too small)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
