"""Speed of the naive grouped MXFP4 GEMM vs the production TDM kernel.

    HIP_VISIBLE_DEVICES=0 python bench_vs_prod.py
    HIP_VISIBLE_DEVICES=0 python bench_vs_prod.py --tokens 1 128 8192 --gemm 2

Shape: DeepSeek-V4.1 Flash TP4 a4w4 (E=384, topk=6, hidden 5120, inter 576):
gemm1 N=1152 K=5120, gemm2 N=5120 K=576, balanced routing like the tuner.

* production: ``flydsl_grouped_gemm_a8w4_masked`` with the tuned row of
  model_configs/dsv41_flash_tuned_grouped_fmoe.csv for this token bucket, plain
  bf16 output (stage1_act=0) so both kernels do the same GEMM; operands in the
  production layouts ((16,16) weight preshuffle, n32k4 B scale, preshuffled A
  scale, unpadded K; see the K-tail note in run() for the A pitch and weight tail).
* naive: the same tile_m/tile_n/tile_k/m_warp/n_warp/num_buffers and routed
  layout as the production row (it has no split-K or clusters); K padding and
  the A-scale grouping (preshuffle_act_scale) are done outside the timing.

Both outputs are checked against a torch reference on a few experts and against
each other on every valid row before timing (profiler device time).
"""

import argparse
import csv
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiter.ops.flydsl.grouped_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked
from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _as_int,
    _grouped_a8w4_preshuffle_e8m0_scale,
)
from aiter.ops.shuffle import shuffle_scale_n32k4, shuffle_weight
from aiter.test_common import run_perftest
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

from naive_grouped_gemm_mxfp4 import (
    ceildiv,
    naive_grouped_gemm_mxfp4,
    preshuffle_act_scale,
    preshuffle_weights,
)

CSV = os.path.join(
    os.path.dirname(os.path.abspath(sys.modules["aiter"].__file__)),
    "configs/model_configs/dsv41_flash_tuned_grouped_fmoe.csv",
)
E, TOPK, HIDDEN, INTER = 384, 6, 5120, 576
GEMMS = {1: (2 * INTER, HIDDEN), 2: (HIDDEN, INTER)}  # stage -> (N, K)

def tuned_row(tokens):
    bucket = 1 << (tokens - 1).bit_length()
    with open(CSV) as f:
        for r in csv.DictReader(f):
            if (
                r["gfx"] == "gfx1250"
                and (int(r["expert"]), int(r["topk"]), int(r["inter_dim"])) == (E, TOPK, INTER)
                and r["q_dtype_a"] == "torch.float4_e2m1fn_x2"
                and int(r["token"]) == bucket
            ):
                return r
    raise LookupError(f"no tuned row for token bucket {bucket}")


def prod_kwargs(r, stage):
    """The CSV row -> kernel knobs, with grouped_moe_gfx1250's defaults."""
    def g(k, d):
        return _as_int(r.get(k), d)

    kw = {
        "tile_m": g("tile_m", 64),
        "tile_n": g("tile_n", 256),
        "tile_k": g("tile_k", 128),
        "m_warp": g("m_warp", 1),
        "n_warp": g("n_warp", 4),
        "num_buffers": g("num_buffers", 2),
        "cluster_m": g("cluster_m", -1),
        "cluster_n": g("cluster_n", -1),
        "waves_per_tensor_tdm": g("waves_per_tensor_tdm", -1),
        "k_waves": max(1, g("split_k1", 1)),
        "persistent_workers": g("persistent_workers", 0),
    }
    if stage == 2:
        kw = {
            "tile_m": g("tile_m2", kw["tile_m"]),
            "tile_n": g("tile_n2", kw["tile_n"]),
            "tile_k": g("tile_k2", kw["tile_k"]),
            "m_warp": g("m_warp2", kw["m_warp"]),
            "n_warp": g("n_warp2", kw["n_warp"]),
            "num_buffers": g("num_buffer_stage2", kw["num_buffers"]),
            "cluster_m": g("cluster_m2", -1),
            "cluster_n": g("cluster_n2", kw["cluster_n"]),
            "waves_per_tensor_tdm": g("waves_per_tensor_tdm2", kw["waves_per_tensor_tdm"]),
            "k_waves": max(1, g("split_k2", 1)),
            "persistent_workers": g("persistent_workers2", 0),
        }
    for k in ("next_stage_prefetch", "tdm_as_in_prologue", "tdm_b_th", "lds_soa_load_interleave", "rolled_drain"):
        kw[k] = g(k, 0)
    assert g("a_preshuffle", 0) == 0, "A-preshuffled rows are not covered here"
    return kw


class Layout:
    """Contiguous-M rows for one tile alignment, sized like production."""

    def __init__(self, counts, align):
        padded = (counts + align - 1) // align * align
        self.starts = torch.cumsum(padded, 0) - padded
        self.psum = (self.starts + counts).to(torch.int32).cuda()
        routes = int(counts.sum())
        bound = min(routes + E * align - TOPK, routes + min(E, routes) * (align - 1))
        self.m = max(align, ceildiv(bound, align) * align)
        first = torch.cumsum(counts, 0) - counts
        self.rows = (
            self.starts.repeat_interleave(counts)
            + torch.arange(routes)
            - first.repeat_interleave(counts)
        ).cuda()  # contiguous row of every route, in expert order

    def place(self, x):
        out = torch.zeros((self.m, *x.shape[1:]), dtype=x.dtype, device=x.device)
        out[self.rows] = x
        return out


def pad_k(x, cols, value):
    return torch.nn.functional.pad(x, (0, cols - x.shape[-1]), value=value)


def with_zero_tail(x, slack=1 << 20):
    """``x`` copied into a buffer followed by ``slack`` zero bytes."""
    buf = torch.zeros(x.numel() + slack, dtype=x.dtype, device=x.device)
    buf[: x.numel()].copy_(x.flatten())
    return buf[: x.numel()].view(x.shape)


def bench(fn):
    _, us = run_perftest(fn, num_iters=30, num_warmup=5, num_rotate_args=1)
    return us


def run(tokens, stage, check_experts=3):
    N, K = GEMMS[stage]
    gen = torch.Generator(device="cuda").manual_seed(stage * 1000 + tokens)
    ids = (torch.arange(tokens * TOPK) % E).view(tokens, TOPK)  # balanced routing
    counts = torch.bincount(ids.flatten(), minlength=E)
    routes = int(counts.sum())

    def rand(*shape, lo=0, hi=256):
        return torch.randint(lo, hi, shape, generator=gen, device="cuda", dtype=torch.uint8)

    a, a_s = rand(routes, K // 2), rand(routes, K // 32, lo=124, hi=131)
    w, w_s = rand(E, N, K // 2), rand(E, N, K // 32, lo=124, hi=131)

    # torch reference for the first few non-empty experts (routes are in expert order)
    first = torch.cumsum(counts, 0) - counts
    def deq(d, s):
        return mxfp4_to_f32(d) * e8m0_to_f32(s).repeat_interleave(32, -1)

    ref = {}
    for e in torch.nonzero(counts).flatten()[:check_experts].tolist():
        r0, r1 = int(first[e]), int(first[e] + counts[e])
        ref[(r0, r1)] = deq(a[r0:r1], a_s[r0:r1]) @ deq(w[e], w_s[e]).T

    def check(name, out_rows):
        for (r0, r1), exp in ref.items():
            err = ((out_rows[r0:r1].float() - exp).abs().max() / exp.abs().max()).item()
            assert err < 1e-2, f"{name}: rows {r0}:{r1} rel_err {err:.3e}"

    # ---- production ----------------------------------------------------------
    kw = prod_kwargs(tuned_row(tokens), stage)
    lay_p = Layout(counts, kw["tile_m"])
    wmma_rep = kw["tile_m"] // kw["m_warp"] // 16
    k128 = ceildiv(K, 128)
    # When tile_k does not divide K (K=576), the kernel's A and B TDM views span
    # ceil(K / tile_k) whole tiles, so the last tile reads past every A row and
    # past each 16-row weight group: A rows get a zero-padded pitch (otherwise the
    # next row's bytes enter the tail WMMA) and the weights a zero tail (otherwise
    # the last expert reads unmapped memory).
    a_pitch = ceildiv(K, kw["tile_k"]) * kw["tile_k"] // 2
    a_p = pad_k(lay_p.place(a), a_pitch, 0)
    as_p = _grouped_a8w4_preshuffle_e8m0_scale(
        pad_k(lay_p.place(a_s), k128 * 4, 127).view(1, lay_p.m, -1),
        warp_tile=wmma_rep * 16,
    )
    w_p = with_zero_tail(shuffle_weight(w, layout=(16, 16)))
    ws_p = shuffle_scale_n32k4(w_s).reshape(-1).view(torch.int32)
    out_p = torch.empty((lay_p.m, N), dtype=torch.bfloat16, device="cuda")
    pitch_kw = {"a_row_stride_bytes": a_pitch} if a_pitch != K // 2 else {}

    def prod():
        flydsl_grouped_gemm_a8w4_masked(
            out_p, a_p, w_p, as_p, ws_p, lay_p.psum,
            n_experts=E, contiguous_m=lay_p.m, N=N, K=K, a_is_fp4=1, stage1_act=0,
            **pitch_kw, **kw,
        )  # fmt: skip

    prod()
    rows_p = out_p[lay_p.rows]
    check("production", rows_p)
    us_p = bench(prod)

    # ---- naive: the production tile and wave grid, on the same layout ---------
    keys = ("tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "num_buffers")
    tile = {k: kw[k] for k in keys}
    kp = ceildiv(K, kw["tile_k"]) * kw["tile_k"]
    a_n = pad_k(lay_p.place(a), kp // 2, 0)
    as_n = preshuffle_act_scale(lay_p.place(a_s), kw["tile_m"], kw["tile_k"])
    w_n, ws_n = preshuffle_weights(w, w_s, kw["tile_k"])
    out_n = torch.empty((lay_p.m, N), dtype=torch.bfloat16, device="cuda")

    def naive():
        naive_grouped_gemm_mxfp4(a_n, as_n, w_n, ws_n, lay_p.psum, out=out_n, **tile)

    naive()
    rows_n = out_n[lay_p.rows]
    check("naive", rows_n)
    diff = ((rows_n.float() - rows_p.float()).abs().max() / rows_p.float().abs().max()).item()
    assert diff < 2e-2, f"naive vs production: {diff:.3e}"
    us_n = bench(naive)

    def tflops(us):
        return 2 * routes * N * K / us / 1e6

    cfg = f"{kw['tile_m']}x{kw['tile_n']}x{kw['tile_k']} w{kw['m_warp']}x{kw['n_warp']}"
    cfg += f" b{kw['num_buffers']}"
    extra = f"sk{kw['k_waves']}" if kw["k_waves"] > 1 else "-"  # not in naive
    return {
        "tokens": tokens, "gemm": f"gemm{stage} N={N} K={K}", "routes": routes,
        "cfg": cfg, "prod_extra": extra, "prod_us": us_p, "prod_tflops": tflops(us_p),
        "naive_us": us_n, "naive_tflops": tflops(us_n), "slowdown": us_n / us_p,
    }  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 128, 1024, 8192])
    p.add_argument("--gemm", type=int, nargs="+", default=[1, 2], choices=(1, 2))
    args = p.parse_args()
    rows = []
    for tokens, stage in itertools.product(args.tokens, args.gemm):
        rows.append(run(tokens, stage))
        r = rows[-1]
        print(
            f"T={r['tokens']:>5} {r['gemm']:<20} routes={r['routes']:>6} | "
            f"tile {r['cfg']:<23} | prod ({r['prod_extra']:<3}) {r['prod_us']:>8.1f} us "
            f"{r['prod_tflops']:>7.1f} TF | naive {r['naive_us']:>8.1f} us "
            f"{r['naive_tflops']:>7.1f} TF | naive/prod {r['slowdown']:.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
