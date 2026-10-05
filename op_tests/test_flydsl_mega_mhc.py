# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-launch Mega-mHC seam (DeepSeek-V4.1 delayed mHC) vs fp32 torch.

Three tables:
  test_mega_mhc          candidates flydsl (policy config) and the Triton seam,
                         plus a CUDA-graph replay check of the flydsl launch.
  test_mega_mhc_config   one row per legal knob set (the tuning sweep), with ISA
                         resources when run under FLYDSL_DUMP_IR=1.
  test_mega_mhc_streams  two seams on two streams at once (per-stream scratch).
"""

import argparse
import glob
import itertools
import os
import re

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]

# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat (DSV4.1)
ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)
NORM_EPS = 1e-6
FN_BYTES = 2 * 24 * 4 * 2  # prepacked bf16 hi/lo per hidden column (x H)

_FAILURES = []


def run_torch(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    sinkhorn_repeat,
    pre_mix,
    sublayer_out=None,
    post_layer_mix=None,
    comb_res_mix=None,
    norm_weight=None,
    norm_eps=1e-6,
):
    """fp32 torch reference (vLLM's mhc_post_torch + mhc_pre_delayed_torch + RMSNorm),
    as in op_tests/triton_tests/fusions/test_mhc_fused_post_pre_delayed_rmsnorm.py.
    Also returns the un-rounded normalized fp32 layer input, for the FP8 check."""
    T, n, _H = residual.shape
    if sublayer_out is not None:
        mixed = torch.einsum(
            "tij,tih->tjh", comb_res_mix.float().view(T, n, n), residual.float()
        )
        R = (
            mixed
            + post_layer_mix.float().view(T, n, 1) * sublayer_out.float().unsqueeze(1)
        ).to(residual.dtype)
    else:
        R = residual
    x = R.flatten(1).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = (
        torch.sigmoid(mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n])
        * hc_post_mult
    )
    comb = mixes[:, 2 * n :].view(-1, n, n) * hc_scale[2] + hc_base[2 * n :].view(
        1, n, n
    )
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    li = (pre_mix.float().view(T, n, 1) * R.float()).sum(dim=1).to(residual.dtype)
    lf = li.float()
    li_f32 = (
        lf
        * torch.rsqrt(lf.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    )
    return R, post.unsqueeze(-1), comb, li_f32.to(residual.dtype), pre, li_f32


def make_inputs(T, H, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = (torch.randn(T, 4, H, generator=g) * 1.5).to(torch.bfloat16)
    y = torch.randn(T, H, generator=g).to(torch.bfloat16)
    fn = torch.randn(24, 4 * H, generator=g) * 0.01
    hc_scale = torch.tensor([0.7, 1.3, 0.9])
    hc_base = torch.randn(24, generator=g) * 0.5
    post = torch.rand(T, 4, 1, generator=g) * 2
    comb = torch.softmax(torch.randn(T, 4, 4, generator=g), -1)
    pre = torch.sigmoid(torch.randn(T, 4, generator=g)) + 1e-6
    w = (1 + 0.2 * torch.randn(H, generator=g)).to(torch.bfloat16)
    return residual, y, fn, hc_scale, hc_base, post, comb, pre, w


def _call_kwargs(mode, T, H, seed=0):
    """Inputs, call kwargs and the reference, as the model calls the seam:
    a preallocated residual_out for the post-mix seam, none for the Engram seam."""
    residual, y, fn, hc_scale, hc_base, post, comb, pre, w = make_inputs(T, H, seed)
    kw = {"norm_weight": w, "norm_eps": NORM_EPS}
    if mode != "no_post":
        kw.update(sublayer_out=y, post_layer_mix=post, comb_res_mix=comb)
    ref_pre = pre
    if mode == "identity_pre":
        ref_pre = torch.zeros_like(pre)
        ref_pre[:, 0] = 1
        pre = None
    ref = run_torch(residual, fn, hc_scale, hc_base, *ARGS, pre_mix=ref_pre, **kw)
    if mode != "no_post":
        kw["residual_out"] = torch.empty_like(residual)
    args = (residual, fn, hc_scale, hc_base, *ARGS)
    return args, dict(kw, pre_mix=pre), ref


def _fp8_parts(li_f32):
    """Reference group-32 FP8 scale for the normalized layer input."""
    T, H = li_f32.shape
    amax = li_f32.view(T, H // 32, 32).abs().amax(-1)
    fmax = torch.finfo(dtypes.fp8).max
    return torch.where(amax > 0, amax / fmax, torch.ones_like(amax))


def check_outputs(name, out, ref, out_dtype, mode):
    """All per-output checks; returns the worst mismatch ratio."""
    R, post_o, comb_o, li, pre_o = out
    errs = []
    if mode != "no_post":
        errs.append(
            checkAllclose(
                ref[0].float(),
                R.float(),
                rtol=1.6e-2,
                atol=1e-5,
                tol_err_ratio=0.0,
                msg=f"{name}: residual_out ",
            )
        )
    for o, e, tag in (
        (post_o, ref[1], "post"),
        (comb_o, ref[2], "comb"),
        (pre_o, ref[4], "pre"),
    ):
        errs.append(
            checkAllclose(
                e.float(),
                o.float(),
                rtol=1e-3,
                atol=5e-4,
                tol_err_ratio=0.0,
                msg=f"{name}: {tag} ",
            )
        )
    if out_dtype == "bf16":
        # a one-ulp rounding difference in R' carries into the collapse
        errs.append(
            checkAllclose(
                ref[3].float(),
                li.float(),
                rtol=1e-2,
                atol=2e-2,
                tol_err_ratio=0.0,
                msg=f"{name}: layer_input ",
            )
        )
    else:
        q, s = li
        T, H = q.shape
        deq = (q.float().view(T, H // 32, 32) * s.unsqueeze(-1)).view(T, H)
        errs.append(
            checkAllclose(
                ref[5],
                deq,
                rtol=7e-2,
                atol=2e-2,
                tol_err_ratio=0.0,
                msg=f"{name}: fp8 dequant ",
            )
        )
        errs.append(
            checkAllclose(
                _fp8_parts(ref[5]),
                s,
                rtol=1e-2,
                atol=1e-6,
                tol_err_ratio=0.0,
                msg=f"{name}: fp8 scale ",
            )
        )
    err = max(errs)
    if err > 0:
        _FAILURES.append(name)
    return err


def _traffic(T, H, mode, out_dtype):
    """HBM bytes of one seam at the (2n+2)d floor (+ the fn weight)."""
    out_b = 1 + 4 / 32 if out_dtype == "fp8" else 2
    per_tok = 4 * H * 2 + H * out_b
    if mode != "no_post":
        per_tok += H * 2 + 4 * H * 2
    return T * per_tok + FN_BYTES * H


def _graph_err(fn, ref, out_dtype, mode):
    """Capture the launch in a CUDA graph and replay it twice."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    errs = []
    for i in range(2):
        g.replay()
        torch.cuda.synchronize()
        errs.append(
            check_outputs(f"flydsl graph replay {i}", out, ref, out_dtype, mode)
        )
    return max(errs)


@benchmark()
def test_mega_mhc(T, H, mode, out_dtype):
    from aiter.ops.flydsl import flydsl_mega_mhc

    args, kw, ref = _call_kwargs(mode, T, H)
    candidates = {
        "flydsl": lambda: flydsl_mega_mhc(*args, out_dtype=out_dtype, **kw),
    }
    # the Triton seam has no FP8 output, so it is only a bf16 candidate
    if out_dtype == "bf16":
        try:
            from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
                mhc_fused_post_pre_delayed_rmsnorm,
            )

            candidates["triton_seam"] = lambda: mhc_fused_post_pre_delayed_rmsnorm(
                *args, **kw
            )
        except ImportError as e:  # pragma: no cover
            aiter.logger.warning("triton seam unavailable: %s", e)

    flops = 2 * T * 4 * H * 24
    nbytes = _traffic(T, H, mode, out_dtype)
    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = (
            check_outputs(name, out, ref, out_dtype, mode) if T else 0.0
        )
    if T:
        ret["flydsl graph err"] = _graph_err(candidates["flydsl"], ref, out_dtype, mode)
    ret["floor us"] = nbytes / 6.5e6  # MI355X ~6.5 TB/s achievable HBM
    return ret


def _isa_resources(name):
    """VGPR/AGPR/SGPR/spill/LDS of a dumped kernel (FLYDSL_DUMP_IR=1), else {}."""
    root = os.environ.get("FLYDSL_DUMP_DIR")
    if not (os.environ.get("FLYDSL_DUMP_IR") == "1" and root):
        return {}
    paths = sorted(glob.glob(os.path.join(root, name + "*", "*final_isa.s")))
    if not paths:
        return {}
    with open(paths[-1]) as f:
        text = f.read()
    res = {}
    for key in (
        "vgpr_count",
        "agpr_count",
        "sgpr_count",
        "vgpr_spill_count",
        "sgpr_spill_count",
        "group_segment_fixed_size",
        "private_segment_fixed_size",
    ):
        m = re.search(rf"\.{key}:\s+(\d+)", text)
        if m:
            res[key] = int(m.group(1))
    return res


def _wgs_per_cu(res, warps_per_wg):
    """Resident WGs per CU the ISA resources allow (4 SIMDs, 512 VGPRs, 160 KB LDS)."""
    if "vgpr_count" not in res:
        return None
    # gfx90a+: .vgpr_count is already the unified arch + acc total
    regs = res["vgpr_count"]
    regs = -(-max(regs, 1) // 8) * 8
    waves_per_simd = min(8, 512 // regs)
    by_regs = (4 * waves_per_simd) // warps_per_wg
    lds = res.get("group_segment_fixed_size", 0)
    by_lds = (160 * 1024) // lds if lds else by_regs
    return min(by_regs, by_lds)


def _auto_coherence(T, block_m, ksplit):
    return "none" if ksplit == 1 else "xcd"


@benchmark()
def test_mega_mhc_config(
    T,
    out_dtype,
    block_m,
    warp_split,
    warps_per_wg,
    warps_per_simd,
    ksplit,
    tile_k,
    nt_streams,
    coherence,
    H=5120,
    mode="post",
):
    from aiter.ops.flydsl import flydsl_mega_mhc
    from aiter.ops.flydsl.kernels.mega_mhc import kernel_name

    coh = _auto_coherence(T, block_m, ksplit)
    if coherence != "auto" and ksplit > 1:
        coh = coherence
    cfg = {
        "BLOCK_M": block_m,
        "WARP_SPLIT": warp_split,
        "WARPS_PER_WG": warps_per_wg,
        "WARPS_PER_SIMD": warps_per_simd,
        "NUM_KSPLIT": ksplit,
        "TILE_K": tile_k,
        "COHERENCE": coh,
        "NT_STREAMS": bool(nt_streams),
        "FN_PREPACKED": True,
        "SINKHORN_RCP": True,
    }
    args, kw, ref = _call_kwargs(mode, T, H)

    def run():
        return flydsl_mega_mhc(*args, out_dtype=out_dtype, config=cfg, **kw)

    out, us = run_perftest(run)
    nbytes = _traffic(T, H, mode, out_dtype)
    nblk = -(-T // block_m)
    ret = {
        "gfx": get_gfx(),
        "coh": coh,
        "us": us,
        "TFLOPS": 2 * T * 4 * H * 24 / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "err": check_outputs(f"cfg {cfg}", out, ref, out_dtype, mode),
        "fn MB": nblk * FN_BYTES * H / 1e6,
    }
    res = _isa_resources(
        kernel_name(cfg, mode != "no_post", mode == "identity_pre", out_dtype == "fp8")
    )
    ret.update(res)
    ret["wg/cu"] = _wgs_per_cu(res, warps_per_wg)
    return ret


@benchmark()
def test_mega_mhc_streams(T, H, ksplit, coherence):
    """Two seams on two streams at once; each must match its own reference.
    coherence="agent" spreads a token block's splits over all XCDs (cross-XCD
    finish), "xcd" keeps them on one."""
    from aiter.ops.flydsl import flydsl_mega_mhc

    cfg = {
        "BLOCK_M": 16,
        "WARPS_PER_WG": 4,
        "TILE_K": 32,
        "NUM_KSPLIT": ksplit,
        "COHERENCE": "none" if ksplit == 1 else coherence,
    }
    cases = [_call_kwargs("post", T, H, seed=s) for s in (1, 2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    torch.cuda.synchronize()
    outs = [None, None]
    for _ in range(20):
        for i, (args, kw, _ref) in enumerate(cases):
            with torch.cuda.stream(streams[i]):
                outs[i] = flydsl_mega_mhc(*args, config=cfg, **kw)
    torch.cuda.synchronize()
    errs = [
        check_outputs(f"stream {i}", outs[i], cases[i][2], "bf16", "post")
        for i in range(2)
    ]
    return {"gfx": get_gfx(), "err": max(errs)}


def _legal(H, cfg):
    from aiter.ops.flydsl.kernels.mega_mhc import check_config

    try:
        check_config(H, cfg)
        return True
    except ValueError:
        return False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl mega-mhc unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL Mega-mHC seam: correctness + perf",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[0, 1, 16, 32, 64, 384, 4096, 16384],
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[5120])
    parser.add_argument(
        "--mode", nargs="*", default=["post", "no_post", "identity_pre"]
    )
    parser.add_argument("-d", "--out_dtype", nargs="*", default=["bf16", "fp8"])
    # knob sweep axes (test_mega_mhc_config); empty --block_m skips the sweep
    parser.add_argument("--block_m", type=int, nargs="*", default=[])
    parser.add_argument("--warp_split", nargs="*", default=["cols", "tokens"])
    parser.add_argument("--warps_per_wg", type=int, nargs="*", default=[2, 4])
    parser.add_argument("--warps_per_simd", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--ksplit", type=int, nargs="*", default=[1])
    parser.add_argument("--tile_k", type=int, nargs="*", default=[32, 64])
    parser.add_argument("--nt_streams", type=int, nargs="*", default=[0])
    parser.add_argument("--coherence", nargs="*", default=["auto"])
    parser.add_argument("--stream_ksplit", type=int, nargs="*", default=[1, 10])
    parser.add_argument("--stream_coherence", nargs="*", default=["xcd", "agent"])
    args = parser.parse_args()

    rows = [
        test_mega_mhc(T, H, mode, d)
        for T, H, mode, d in itertools.product(
            args.tokens, args.hidden, args.mode, args.out_dtype
        )
    ]
    aiter.logger.info(
        "mega-mhc summary (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )

    if args.block_m:
        rows = []
        for T, d, bm, ws, w, wps, ks, tk, nt, coh in itertools.product(
            [t for t in args.tokens if t > 0],
            args.out_dtype,
            args.block_m,
            args.warp_split,
            args.warps_per_wg,
            args.warps_per_simd,
            args.ksplit,
            args.tile_k,
            args.nt_streams,
            args.coherence,
        ):
            c = coh if coh != "auto" and ks > 1 else _auto_coherence(T, bm, ks)
            cfg = {
                "BLOCK_M": bm,
                "WARP_SPLIT": ws,
                "WARPS_PER_WG": w,
                "WARPS_PER_SIMD": wps,
                "NUM_KSPLIT": ks,
                "TILE_K": tk,
                "COHERENCE": c,
            }
            if not _legal(args.hidden[0], cfg):
                continue
            if ks == 1 and coh != args.coherence[0]:
                continue  # coherence only applies to split-K
            rows.append(
                test_mega_mhc_config(
                    T, d, bm, ws, w, wps, ks, tk, nt, coh, H=args.hidden[0]
                )
            )
        aiter.logger.info(
            "mega-mhc config sweep (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )

    rows = [
        test_mega_mhc_streams(T, H, ks, coh)
        for T, H, ks, coh in itertools.product(
            [t for t in args.tokens if 0 < t <= 4096][-2:],
            args.hidden,
            args.stream_ksplit,
            args.stream_coherence,
        )
        if not (ks == 1 and coh != args.stream_coherence[0])
    ]
    aiter.logger.info(
        "mega-mhc two-stream check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    if _FAILURES:
        raise AssertionError(
            f"{len(_FAILURES)} mega-mhc checks failed: {_FAILURES[:8]}"
        )


if __name__ == "__main__":
    main()
