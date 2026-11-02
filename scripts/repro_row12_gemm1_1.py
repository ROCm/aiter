# triton 3.9.0+gitad50bab5.gitad50bab5 | gfx1250 | kernel a4w4 shape 9b (EP2, E_local 128, D=I=2304)
# M=4096 block_m= 32 in-kernel gather cos=1.000000 bad_rows=0/16308 OK
# M=4096 block_m= 32 pre-gathered     cos=1.000000 bad_rows=0/16308 OK
# M=4096 block_m= 64 in-kernel gather cos=1.000000 bad_rows=0/16308 OK
# M=4096 block_m= 64 pre-gathered     cos=1.000000 bad_rows=0/16308 OK
# M=4096 block_m=128 in-kernel gather cos=0.995695 bad_rows=2813/16308 WRONG
#     bad-element fraction by sub-tile, full tiles only: rows = 32-row band in tile, cols = 64-col band in each 256-col block
#     rows   0- 31: 0.058  0.000  0.000  0.000
#     rows  32- 63: 0.000  0.000  0.000  0.000
#     rows  64- 95: 0.000  0.000  0.000  0.000
#     rows  96-127: 0.000  0.000  0.000  0.000
# M=4096 block_m=128 pre-gathered     cos=0.996385 bad_rows=2314/16308 WRONG
#     bad-element fraction by sub-tile, full tiles only: rows = 32-row band in tile, cols = 64-col band in each 256-col block
#     rows   0- 31: 0.057  0.000  0.000  0.000
#     rows  32- 63: 0.000  0.000  0.000  0.000
#     rows  64- 95: 0.000  0.000  0.000  0.000
#     rows  96-127: 0.000  0.000  0.000  0.000

# triton 3.9.0+gitad50bab5.gitad50bab5 | gfx1250 | kernel a8w4 shape 9b (EP2, E_local 128, D=I=2304)
# M=4096 block_m= 32 in-kernel gather cos=0.999868 bad_rows=0/16308 OK
# M=4096 block_m= 32 pre-gathered     cos=0.999868 bad_rows=0/16308 OK
# M=4096 block_m= 64 in-kernel gather cos=0.999868 bad_rows=0/16308 OK
# M=4096 block_m= 64 pre-gathered     cos=0.999868 bad_rows=0/16308 OK
# M=4096 block_m=128 in-kernel gather cos=0.975522 bad_rows=2515/16308 WRONG
#     bad-element fraction by sub-tile, full tiles only: rows = 32-row band in tile, cols = 64-col band in each 256-col block
#     rows   0- 31: 0.001  0.113  0.001  0.113
#     rows  32- 63: 0.001  0.001  0.001  0.001
#     rows  64- 95: 0.001  0.001  0.001  0.001
#     rows  96-127: 0.001  0.001  0.001  0.001
# M=4096 block_m=128 pre-gathered     cos=0.999868 bad_rows=0/16308 OK



# repro_row12_gemm1.py
#!/usr/bin/env python3
"""Minimal standalone repro (torch + AITER + Triton only, random weights) for AMD DVT sheet row 12.

AITER Gluon moe_gemm_a4w4 / moe_gemm_a8w4 GEMM1 (MXFP4 weights, SwiGLU, MX-quantized output) on gfx1250
at block_m 128. One EP rank (rank 0 of EP2 -> 128 local experts of 256), top-8, hidden = intermediate = 2304
("9b") or 4096 ("t5", EP8 -> 32 local experts). GEMM1 output rows (expert-sorted) are compared with an fp32
reference that models the MX requant of the GEMM1 output. Correct output gives 0 bad rows (per-row cos < 0.99) and
cos 0.999999 (a4w4) / 0.99987 (a8w4: MXFP8 requant rounding flips).

  python repro_row12_gemm1.py                        # a4w4, 9b, M 4096, block_m 32/64/128, in-kernel gather + pre-gathered
  python repro_row12_gemm1.py --kernel a8w4 --shape t5

For each wrong case it prints where the bad elements are inside the block_m x 256 output tiles
(fraction of bad elements per 32-row x 64-col sub-tile). At block_m 128 the a4w4 tile is 128 x 512 (256 output cols
after SwiGLU), and the a8w4 tile is 128 x 256 (128 output cols, so each 256-col block holds two tiles).
"""

import argparse
import importlib.metadata as md

import torch
import triton
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    ep_sort_routing,
    EpScatterGeometry,
    ExptData,
    RoutingData,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.utils.shuffle import moe_weight_decode_view

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", choices=["a4w4", "a8w4"], default="a4w4")
ap.add_argument("--shape", choices=["9b", "t5"], default="9b")
ap.add_argument("--M", type=int, nargs="+", default=[4096])
ap.add_argument("--block-m", type=int, nargs="+", default=[32, 64, 128])
ap.add_argument("--apply-swiglu", action='store_true', default=False, help='whether apply swiglu')
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()

E, TOPK = 256, 8
EP, D, I = (2, 2304, 2304) if A.shape == "9b" else (8, 4096, 4096)
EL = E // EP
dev = "cuda"
E2M1 = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=dev
)

if A.kernel == "a4w4":
    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4 as gemm, mxfp4_quant

    def act_quant(x):
        return mxfp4_quant(x)
else:
    from aiter.ops.triton.moe.moe_op_gemm_a8w4 import moe_gemm_a8w4 as gemm

    def act_quant(x):
        return downcast_to_mxfp(x, torch.float8_e4m3fn, axis=-1)


def upcast(q, s):
    """MX tensor (e2m1 packed low nibble first, or fp8) + e8m0 scales along the last dim -> fp32."""
    v = (
        torch.stack((E2M1[(q & 0xF).long()], E2M1[(q >> 4).long()]), -1).flatten(-2)
        if q.dtype == torch.uint8
        else q.float()
    )
    return (
        v
        * torch.exp2(s.view(torch.uint8).float() - 127.0).repeat_interleave(32, -1)[
            ..., : v.shape[-1]
        ]
    )


def weights(K, N, seed):
    """random [EL, K, N] bf16 -> MXFP4 kernel weight + shuffled scales, and the fp32 dequant [EL, K, N]"""
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randn((EL, K, N), generator=g, device=dev, dtype=torch.bfloat16) / K**0.5
    wq, ws = downcast_to_mxfp(w, torch.uint8, axis=1)
    ref = upcast(wq.transpose(1, 2), ws.transpose(1, 2)).transpose(1, 2)
    ws = moe_shuffle_scale(ws.transpose(-1, -2)).transpose(-1, -2)
    wq = moe_weight_decode_view(moe_shuffle_weight(wq.transpose(-1, -2)))
    return wq, ws, ref


def routing(ids, wts, block_m):
    M = ids.shape[0]
    n_gates = M * TOPK
    emap = torch.full((E + 1,), -1, dtype=torch.int32, device=dev)
    emap[:EL] = torch.arange(
        EL, dtype=torch.int32, device=dev
    )  # this rank owns experts [0, EL)
    bufs = _compute_expt_data_internal(EL, n_gates, block_m, dev)
    geo = EpScatterGeometry(
        src_token_map=torch.arange(M, dtype=torch.int32, device=dev),
        max_tokens_per_rank=M,
        peer_rows=n_gates,
    )
    hist, gidx, _, gscal, _, _ = ep_sort_routing(
        wts,
        ids,
        emap,
        EL,
        torch.full((1,), M, dtype=torch.int32, device=dev),
        M,
        TOPK,
        n_gates,
        bufs,
        ep_scatter_geometry=geo,
    )
    h = hist[:EL]
    return RoutingData(
        block_m, gscal, h, EL, TOPK, ExptData(h, bufs[0], bufs[1], bufs[2])
    ), gidx


def gemm1(xq, xs, w1, s1, rd, gidx, apply_swiglu=True):
    kw = dict(
        gather_indx=gidx,
        swizzle_mx_scale="GFX1250_SCALE",
        out_mx_quant=True,
        apply_swiglu=apply_swiglu,
        alpha=1.0,
        limit=1e4,
        swiglu_add_residual=False,
    )
    if A.kernel == "a4w4":
        return gemm(xq, w1, xs, s1, None, rd, preshuffle_weights=True, **kw)
    return gemm(xq, w1, xs, s1, None, None, None, rd, preshuffled=True, **kw)


def main():
    p = torch.cuda.get_device_properties(0)
    print(
        f"triton {md.version('triton')} | aiter {md.version('amd-aiter')} | torch {torch.__version__} | {p.gcnArchName} "
        f"| kernel {A.kernel} shape {A.shape} (EP{EP}, E_local {EL}, D=I={D})",
        flush=True,
    )
    w1, s1, w1r = weights(D, 2 * I, A.seed + 1)
    for M in A.M:
        g = torch.Generator(device=dev).manual_seed(A.seed)
        x = torch.randn((M, D), generator=g, device=dev, dtype=torch.bfloat16)
        vals, ids = torch.topk(
            torch.randn((M, E), generator=g, device=dev), TOPK, dim=-1
        )
        ids, wts = (
            ids.to(torch.int32).contiguous(),
            torch.softmax(vals, -1).contiguous(),
        )
        xq, xs = act_quant(x)
        xr = upcast(xq, xs)
        apply_swiglu = A.apply_swiglu
        for bm in A.block_m:
            rd, gidx = routing(ids, wts, bm)
            n = int(rd.expt_data.hist.sum())
            gate = gidx.long()[:n]
            tok = gate // TOPK
            exp = ids.flatten().long()[gate]
            if apply_swiglu:
                ref = torch.empty((n, I), device=dev)
            else:
                ref = torch.empty((n, 2 * I), device=dev)
            for e in exp.unique().tolist():
                s = (exp == e).nonzero(as_tuple=True)[0]
                h = xr[tok[s]] @ w1r[e]
                if apply_swiglu:
                    ref[s] = (
                        h[:, ::2] * torch.sigmoid(h[:, ::2]) * h[:, 1::2]
                    )  # kernel convention: gate even, up odd cols
                else:
                    ref[s] = h
            ref_rq = upcast(*act_quant(ref.bfloat16()))
            hist = rd.expt_data.hist.long()
            off = (
                torch.arange(n, device=dev) - (torch.cumsum(hist, 0) - hist)[exp]
            )  # row offset inside its expert
            
            for mode in ("in-kernel gather",):
                print(f"apply_swiglu = {apply_swiglu}")
                if mode == "in-kernel gather":
                    y, ys = gemm1(xq, xs, w1, s1, rd, gidx, apply_swiglu=apply_swiglu)
                else:
                    y, ys = gemm1(xq[tok], xs[tok], w1, s1, rd, None, apply_swiglu=apply_swiglu)
                torch.cuda.synchronize()
                got = upcast(y[:n], ys[:n])
                cos = torch.nn.functional.cosine_similarity(
                    got.flatten(), ref_rq.flatten(), dim=0
                ).item()
                rc = torch.nn.functional.cosine_similarity(got, ref_rq, dim=1)
                bad = ~(rc > 0.99)
                print(
                    f"M={M} block_m={bm:3d} {mode:16s} cos={cos:.6f} bad_rows={int(bad.sum())}/{n} "
                    f"{'OK' if cos > 0.999 and not bad.any() else 'WRONG'}",
                    flush=True,
                )
                if bad.any():
                    full = (
                        hist[exp] - (off // bm) * bm >= bm
                    )  # rows in full block_m tiles
                    err = (
                        got - ref_rq
                    ).abs() > 0.05 * ref_rq.abs().mean() + 0.1 * ref_rq.abs()
                    err = err[full]
                    r32 = (off[full] % bm) // 32
                    nb = I // 256
                    print(
                        "    bad-element fraction by sub-tile, full tiles only: rows = 32-row band in tile, "
                        "cols = 64-col band in each 256-col block"
                    )
                    for rb in range(bm // 32):
                        sub = err[r32 == rb].view(-1, nb, 4, 64)
                        print(
                            f"    rows {rb * 32:3d}-{rb * 32 + 31:3d}: "
                            + "  ".join(
                                f"{float(sub[:, :, cb, :].float().mean()):.3f}"
                                for cb in range(4)
                            ),
                            flush=True,
                        )


if __name__ == "__main__":
    main()

