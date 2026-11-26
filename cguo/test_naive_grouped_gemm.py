"""Correctness test for naive_grouped_gemm_mxfp4 against a torch fp32 reference.

    python test_naive_grouped_gemm.py                 # built-in cases
    python test_naive_grouped_gemm.py --fmt a8w4 --E 16 --N 1152 --K 5120 \
        --max-rows 300 --tile-m 128 --tile-n 384 --tile-k 256 --m-warp 2 --n-warp 2 \
        --num-buffers 4

Every case checks (1) the valid rows against dequant(a) @ dequant(w[e]).T and
(2) that the padding rows between experts were not written.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

from naive_grouped_gemm_mxfp4 import (
    naive_grouped_gemm_mxfp4,
    preshuffle_act_scale,
    preshuffle_weights,
)


def build_layout(rows_per_expert, tile_m):
    """Contiguous-M routed layout: tile-aligned starts and psum = start + rows."""
    rows = torch.tensor(rows_per_expert, dtype=torch.int64)
    padded = (rows + tile_m - 1) // tile_m * tile_m
    starts = torch.cumsum(padded, 0) - padded
    psum = (starts + rows).to(torch.int32)
    m_rows = max(tile_m, int(padded.sum()))
    return starts.tolist(), psum, m_rows


def dequant(data, scale, is_fp4):
    """[..., K/2] fp4 or [..., K] fp8 bytes with [..., K/32] e8m0 -> fp32 [..., K]."""
    values = mxfp4_to_f32(data) if is_fp4 else data.view(torch.float8_e4m3fn).float()
    return values * e8m0_to_f32(scale).repeat_interleave(32, dim=-1)


def run_case(fmt, E, N, K, max_rows, tile_m, tile_n, tile_k, m_warp, n_warp, nb, seed):
    tag = (
        f"{fmt} E={E} N={N} K={K} max_rows={max_rows} "
        f"tile={tile_m}x{tile_n}x{tile_k} w{m_warp}x{n_warp} b{nb}"
    )
    is_fp4 = fmt == "a4w4"
    gen = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, max_rows + 1, (E,), generator=gen).tolist()
    rows[0] = max(rows[0], 1)  # at least one valid row
    starts, psum, M = build_layout(rows, tile_m)

    def rand_u8(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=gen, dtype=torch.uint8).cuda()

    # Every fp4 byte is two finite values; e8m0 124..130 is 2^-3..2^3.
    w = rand_u8(E, N, K // 2)
    w_scale = rand_u8(E, N, K // 32, lo=124, hi=131)
    if is_fp4:
        a = rand_u8(M, K // 2)
    else:
        a = (torch.randn(M, K, generator=gen) * 2).to(torch.float8_e4m3fn).cuda()
    a_scale = rand_u8(M, K // 32, lo=124, hi=131)
    psum = psum.cuda()

    out = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    w_shuf, w_scale_shuf = preshuffle_weights(w, w_scale, tile_k)
    a_scale_shuf = preshuffle_act_scale(a_scale, tile_m, tile_k)
    naive_grouped_gemm_mxfp4(
        a, a_scale_shuf, w_shuf, w_scale_shuf, psum, out=out,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, m_warp=m_warp, n_warp=n_warp,
        num_buffers=nb,
    )  # fmt: skip
    torch.cuda.synchronize()

    a_f32 = dequant(a.view(torch.uint8), a_scale, is_fp4)
    valid = torch.zeros(M, dtype=torch.bool, device="cuda")
    ref = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    for e in range(E):
        r0, r1 = starts[e], int(psum[e])
        if r1 > r0:
            ref[r0:r1] = a_f32[r0:r1] @ dequant(w[e], w_scale[e], True).T
            valid[r0:r1] = True

    got, exp = out[valid].float(), ref[valid]
    rel_err = ((got - exp).abs().max() / exp.abs().max().clamp_min(1e-30)).item()
    cosine = F.cosine_similarity(got.flatten(), exp.flatten(), dim=0).item()
    padding_untouched = bool(out[~valid].isnan().all())
    ok = rel_err < 1e-2 and cosine > 0.9999 and padding_untouched
    print(
        f"[{'PASS' if ok else 'FAIL'}] {tag}: rows={sum(rows)} M={M} "
        f"rel_err={rel_err:.3e} cos={cosine:.6f} padding_untouched={padding_untouched}"
    )
    return ok


DEFAULT_CASES = [
    # fmt, E, N, K, max_rows, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers
    ("a4w4", 8, 256, 512, 40, 16, 32, 128, 1, 1, 1),  # single buffer
    ("a8w4", 8, 256, 512, 40, 16, 32, 128, 1, 1, 2),
    ("a4w4", 8, 256, 512, 40, 16, 32, 256, 1, 1, 2),  # 2 WMMA K steps per K tile
    ("a8w4", 8, 256, 512, 40, 16, 32, 512, 1, 1, 2),  # 1 K tile: b2 -> b1
    ("a4w4", 8, 256, 576, 40, 16, 32, 128, 1, 1, 4),  # K % 128 == 64 (host pad)
    ("a8w4", 8, 256, 576, 40, 16, 32, 256, 1, 1, 3),
    ("a4w4", 6, 1152, 5120, 70, 32, 64, 256, 1, 1, 3),  # dsv41 TP4 gemm1 shape
    ("a4w4", 6, 5120, 576, 70, 16, 64, 128, 1, 1, 2),  # dsv41 TP4 gemm2 shape
    ("a8w4", 6, 5120, 576, 70, 32, 32, 128, 1, 1, 4),
    # multi-wave tiles and buffer counts used by the production table
    ("a4w4", 6, 1152, 5120, 300, 16, 384, 128, 1, 2, 3),
    ("a4w4", 6, 1152, 5120, 300, 128, 384, 256, 2, 2, 4),
    ("a4w4", 6, 5120, 576, 300, 128, 256, 128, 2, 2, 4),
    ("a8w4", 6, 5120, 576, 300, 32, 128, 128, 1, 2, 3),
    ("a8w4", 6, 1152, 5120, 300, 64, 128, 256, 2, 2, 2),
    ("a4w4", 6, 5120, 576, 300, 16, 512, 128, 1, 8, 2),  # 8 waves: 4 issue, 4 don't
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fmt", choices=("a4w4", "a8w4"))
    p.add_argument("--E", type=int, default=8)
    p.add_argument("--N", type=int, default=256)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--max-rows", type=int, default=40)
    p.add_argument("--tile-m", type=int, default=16)
    p.add_argument("--tile-n", type=int, default=32)
    p.add_argument("--tile-k", type=int, default=128)
    p.add_argument("--m-warp", type=int, default=1)
    p.add_argument("--n-warp", type=int, default=1)
    p.add_argument("--num-buffers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cases = (
        [
            (args.fmt, args.E, args.N, args.K, args.max_rows)
            + (args.tile_m, args.tile_n, args.tile_k, args.m_warp, args.n_warp)
            + (args.num_buffers,)
        ]
        if args.fmt
        else DEFAULT_CASES
    )
    results = [run_case(*case, seed=args.seed) for case in cases]
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
