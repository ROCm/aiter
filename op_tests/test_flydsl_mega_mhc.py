# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL single-launch Mega-mHC seam (DeepSeek-V4.1 delayed mHC) vs fp32 torch.

Four tables:
  test_mega_mhc          candidates flydsl (policy config) and the Triton seam,
                         plus a CUDA-graph replay check of the flydsl launch.
  test_mega_mhc_config   one row per legal knob set (the tuning sweep), with ISA
                         resources when run under FLYDSL_DUMP_IR=1.
  test_mega_mhc_streams  two seams on two streams at once (per-stream scratch).
  test_mega_mhc_dist     DIST_FINISH bit-exact against the classic finisher, also with
                         forced hand-off (DIST_SPIN=0) and on two streams at once.
  test_mega_mhc_late     LATE_DESC and SHUFFLE_DPP bit-exact against the same
                         knob set without them, classic and distributed finisher, bf16 and FP8.
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
    """Capture the launch in a CUDA graph and replay it twice (warmed on the
    capture stream first, as the wrapper requires)."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
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


@benchmark()
def test_mega_mhc_config(
    T,
    out_dtype,
    block_m,
    warp_split,
    warps_per_wg,
    ksplit,
    tile_k,
    H=5120,
    mode="post",
):
    from aiter.ops.flydsl import flydsl_mega_mhc
    from aiter.ops.flydsl.kernels.mega_mhc import kernel_name

    cfg = {
        "BLOCK_M": block_m,
        "WARP_SPLIT": warp_split,
        "WARPS_PER_WG": warps_per_wg,
        "NUM_KSPLIT": ksplit,
        "TILE_K": tile_k,
    }
    args, kw, ref = _call_kwargs(mode, T, H)

    def run():
        return flydsl_mega_mhc(*args, out_dtype=out_dtype, config=cfg, **kw)

    out, us = run_perftest(run)
    nbytes = _traffic(T, H, mode, out_dtype)
    nblk = -(-T // block_m)
    ret = {
        "gfx": get_gfx(),
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
def test_mega_mhc_streams(T, H, ksplit):
    """Two seams on two streams at once; each must match its own reference."""
    from aiter.ops.flydsl import flydsl_mega_mhc

    cfg = {
        "BLOCK_M": 16,
        "WARPS_PER_WG": 4,
        "TILE_K": 32,
        "NUM_KSPLIT": ksplit,
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


@benchmark()
def test_mega_mhc_dist(T, H, ksplit, mode):
    """DIST_FINISH must give bit-identical outputs to the classic finisher, in a
    normal run, with every split forced to hand off (DIST_SPIN=0), and with two
    streams of DIST launches at once (per-stream scratch and publish words)."""
    from aiter.ops.flydsl import flydsl_mega_mhc
    from aiter.ops.flydsl.kernels.mega_mhc import dist_residency_error

    base = {
        "BLOCK_M": 16,
        "WARPS_PER_WG": 8,
        "NUM_KSPLIT": ksplit,
        "TILE_K": 64 if H % (ksplit * 8 * 64) == 0 else 32,
    }
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    if dist_residency_error(T, dict(base, DIST_FINISH=True), cu):
        return {"gfx": get_gfx(), "skipped": "not resident"}
    args, kw, ref = _call_kwargs(mode, T, H)
    classic = [
        x.clone() for x in flydsl_mega_mhc(*args, config=base, **kw)
    ]  # residual_out is the shared kw buffer: cloned
    ret = {"gfx": get_gfx()}
    for name, extra in (("dist", {}), ("dist spin0", {"DIST_SPIN": 0})):
        cfg = dict(base, DIST_FINISH=True, **extra)
        outs = [flydsl_mega_mhc(*args, config=cfg, **kw) for _ in range(20)]
        torch.cuda.synchronize()
        bad = sum(
            any(not torch.equal(a, b) for a, b in zip(o[1:], classic[1:])) for o in outs
        )
        ret[f"{name} != classic"] = bad
        ret[f"{name} err"] = check_outputs(
            f"dist {name} T={T}", outs[-1], ref, "bf16", mode
        )
        if bad:
            _FAILURES.append(f"dist {name} T={T} ks={ksplit}: {bad} bad")
    cases = [_call_kwargs(mode, T, H, seed=sd) for sd in (1, 2)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    cfg = dict(base, DIST_FINISH=True)
    torch.cuda.synchronize()
    gold = [flydsl_mega_mhc(*a, config=base, **k) for (a, k, _r) in cases]
    gold = [[x.clone() for x in g] for g in gold]
    outs = [[], []]
    for _ in range(30):
        for i, (a, k, _r) in enumerate(cases):
            with torch.cuda.stream(streams[i]):
                outs[i].append(flydsl_mega_mhc(*a, config=cfg, **k))
    torch.cuda.synchronize()
    sbad = sum(
        any(not torch.equal(x, y) for x, y in zip(o[1:], gold[i][1:]))
        for i in range(2)
        for o in outs[i]
    )
    ret["two-stream != classic"] = sbad
    if sbad:
        _FAILURES.append(f"dist two-stream T={T} ks={ksplit}: {sbad} bad")
    return ret


@benchmark()
def test_mega_mhc_late(T, H, ksplit, mode, out_dtype):
    """LATE_DESC (descriptors built after the first loads) only moves instructions and
    SHUFFLE_DPP swaps ``ds_swizzle`` lane exchanges for DPP ones (same values, same order):
    outputs must be bit-identical to the same knob set without them, with the classic
    finisher and (bf16, when the grid is resident) the distributed one."""
    from aiter.ops.flydsl import flydsl_mega_mhc
    from aiter.ops.flydsl.kernels.mega_mhc import dist_residency_error

    base = {
        "BLOCK_M": 16,
        "WARPS_PER_WG": 8,
        "NUM_KSPLIT": ksplit,
        "TILE_K": 64 if H % (ksplit * 8 * 64) == 0 else 32,
    }
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    args, kw, ref = _call_kwargs(mode, T, H)
    ret = {"gfx": get_gfx()}
    variants = [("classic", {})]
    if out_dtype == "bf16" and not dist_residency_error(
        T, dict(base, DIST_FINISH=True), cu
    ):
        variants.append(("dist", {"DIST_FINISH": True}))
    for name, extra in variants:
        plain = [
            x.clone() if torch.is_tensor(x) else x
            for x in flydsl_mega_mhc(
                *args, out_dtype=out_dtype, config=dict(base, **extra), **kw
            )
        ]
        if out_dtype == "fp8":  # layer_input is a (q, scale) tuple
            plain[3] = tuple(x.clone() for x in plain[3])
        cfg = dict(base, LATE_DESC=True, SHUFFLE_DPP=True, **extra)
        outs = [
            flydsl_mega_mhc(*args, out_dtype=out_dtype, config=cfg, **kw)
            for _ in range(10)
        ]
        torch.cuda.synchronize()

        def same(a, b):
            if isinstance(a, tuple):
                return all(torch.equal(x, y) for x, y in zip(a, b))
            return torch.equal(a, b)

        bad = sum(any(not same(a, b) for a, b in zip(o[1:], plain[1:])) for o in outs)
        ret[f"{name} != plain"] = bad
        ret[f"{name} err"] = check_outputs(
            f"late {name} T={T}", outs[-1], ref, out_dtype, mode
        )
        if bad:
            _FAILURES.append(f"late {name} T={T} ks={ksplit} {out_dtype}: {bad} bad")
    return ret


def _scratch_sizes(T, H, out_dtype):
    """(partial floats, counter ints) the policy's split-K scratch needs at T."""
    from aiter.ops.flydsl.kernels.mega_mhc import grid_size
    from aiter.ops.flydsl.mega_mhc_kernels import get_mega_mhc_config

    dev = torch.cuda.current_device()
    cfg = get_mega_mhc_config(
        T,
        H,
        get_gfx(),
        torch.cuda.get_device_properties(dev).multi_processor_count,
        out_dtype == "fp8",
    )
    _nblk, n_wg = grid_size(T, cfg)
    ks = cfg["NUM_KSPLIT"]
    return max(T * ks * 32, 32), max(n_wg // ks * 32, 32)


@benchmark()
def test_mega_mhc_capture(T_small, T_large, H, out_dtype):
    """CUDA-graph capture safety, as serving captures many batch sizes.

    cold:   capture on a fresh stream with a never-packed fn must raise (no
            garbage pack or un-zeroed counter may be cached); if it does not,
            an eager call right after the capture must still be correct.
    growth: warm + capture T_small, then warm + capture T_large on the same
            stream (the split-K scratch grows), grab freed blocks with junk,
            make an eager call, and replay both graphs in reverse order.
    """
    from aiter.ops.flydsl import flydsl_mega_mhc

    ret = {"gfx": get_gfx()}
    s = torch.cuda.Stream()
    torch.cuda.synchronize()

    # cold capture
    args, kw, ref = _call_kwargs("post", T_small, H, seed=11)
    g = torch.cuda.CUDAGraph()
    raised = False
    try:
        with torch.cuda.graph(g, stream=s):
            flydsl_mega_mhc(*args, out_dtype=out_dtype, **kw)
    except RuntimeError as e:
        raised = "capture" in str(e)
    torch.cuda.synchronize()
    ret["cold raises"] = raised
    with torch.cuda.stream(s):
        out = flydsl_mega_mhc(*args, out_dtype=out_dtype, **kw)
    torch.cuda.synchronize()
    ret["cold eager err"] = check_outputs(
        "capture cold eager", out, ref, out_dtype, "post"
    )
    if not raised:
        _FAILURES.append("capture cold: no RuntimeError")

    # growth across two captured graphs on one stream
    s = torch.cuda.Stream()
    cases = {T: _call_kwargs("post", T, H, seed=T) for T in (T_small, T_large)}
    graphs, outs = {}, {}
    for T in (T_small, T_large):
        a, k, _ = cases[T]
        with torch.cuda.stream(s):
            flydsl_mega_mhc(*a, out_dtype=out_dtype, **k)  # warm on the capture stream
        torch.cuda.synchronize()
        graphs[T] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[T], stream=s):
            outs[T] = flydsl_mega_mhc(*a, out_dtype=out_dtype, **k)
        torch.cuda.synchronize()
    # reuse any freed scratch blocks with junk, on the stream that owned them
    n_part, n_cnt = _scratch_sizes(T_small, H, out_dtype)
    with torch.cuda.stream(s):
        junk = [
            torch.full((n_part,), -7.0, dtype=torch.float32),
            torch.full((n_cnt,), 7, dtype=torch.int32),
        ]
        a, k, r = cases[T_large]
        eager = flydsl_mega_mhc(*a, out_dtype=out_dtype, **k)
    torch.cuda.synchronize()
    ret["eager err"] = check_outputs("capture eager", eager, r, out_dtype, "post")
    for T in (T_large, T_small):
        graphs[T].replay()
        torch.cuda.synchronize()
        ret[f"replay T={T} err"] = check_outputs(
            f"capture replay T={T}", outs[T], cases[T][2], out_dtype, "post"
        )
    del junk
    return ret


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
    parser.add_argument("--ksplit", type=int, nargs="*", default=[1])
    parser.add_argument("--tile_k", type=int, nargs="*", default=[32, 64])
    parser.add_argument("--stream_ksplit", type=int, nargs="*", default=[1, 10])
    parser.add_argument(
        "--dist",
        type=int,
        nargs=2,
        action="append",
        metavar=("T", "KSPLIT"),
        default=None,
        help="DIST_FINISH bit-exact checks: T NUM_KSPLIT (repeatable)",
    )
    parser.add_argument(
        "--capture_tokens",
        type=int,
        nargs=2,
        default=[64, 1024],
        help="T_small T_large for the CUDA-graph capture-safety check",
    )
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
        for T, d, bm, ws, w, ks, tk in itertools.product(
            [t for t in args.tokens if t > 0],
            args.out_dtype,
            args.block_m,
            args.warp_split,
            args.warps_per_wg,
            args.ksplit,
            args.tile_k,
        ):
            cfg = {
                "BLOCK_M": bm,
                "WARP_SPLIT": ws,
                "WARPS_PER_WG": w,
                "NUM_KSPLIT": ks,
                "TILE_K": tk,
            }
            if not _legal(args.hidden[0], cfg):
                continue
            rows.append(test_mega_mhc_config(T, d, bm, ws, w, ks, tk, H=args.hidden[0]))
        aiter.logger.info(
            "mega-mhc config sweep (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )

    rows = [
        test_mega_mhc_streams(T, H, ks)
        for T, H, ks in itertools.product(
            [t for t in args.tokens if 0 < t <= 4096][-2:],
            args.hidden,
            args.stream_ksplit,
        )
    ]
    aiter.logger.info(
        "mega-mhc two-stream check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    dist_cases = args.dist or [
        [4, 20],
        [32, 20],
        [128, 20],
        [256, 10],
        [400, 5],
        [1024, 4],
        [1536, 2],
        [64, 10],
    ]
    rows = [
        test_mega_mhc_dist(T, H, ks, mode)
        for (T, ks), H, mode in itertools.product(dist_cases, args.hidden, args.mode)
        if "bf16" in args.out_dtype
    ]
    aiter.logger.info(
        "mega-mhc DIST_FINISH check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    late_cases = [
        [1, 20],
        [4, 20],
        [32, 20],
        [144, 10],
        [400, 5],
        [1024, 4],
        [64, 10],
    ]
    rows = [
        test_mega_mhc_late(T, H, ks, mode, d)
        for (T, ks), H, mode, d in itertools.product(
            late_cases, args.hidden, args.mode, args.out_dtype
        )
    ]
    aiter.logger.info(
        "mega-mhc LATE_DESC/SHUFFLE_DPP check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    t_small, t_large = args.capture_tokens
    rows = [
        test_mega_mhc_capture(t_small, t_large, H, d)
        for H, d in itertools.product(args.hidden, args.out_dtype)
    ]
    aiter.logger.info(
        "mega-mhc CUDA-graph capture check (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    if _FAILURES:
        raise AssertionError(
            f"{len(_FAILURES)} mega-mhc checks failed: {_FAILURES[:8]}"
        )


if __name__ == "__main__":
    main()
