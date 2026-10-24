# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Delayed mHC seam (``mhc_fused_post_pre_delayed``) against aiter's own mHC ops and
torch references.

The op runs ASM kernels on gfx942 with 4 residual streams and hidden_size 5120 and has
no other path; on other archs this test is skipped. Per case (checkAllclose; the err
columns are its mismatch ratios):

  residual_out  bit-exact (atol = rtol = 0) against ``aiter.mhc_post`` (same fp32
                order, bf16 truncation)
  layer_input   bit-exact against a torch collapse of that residual: fp32 multiply and
                add over streams 0..3 from 0, rounded to nearest even
  post, comb    atol = GATE_ATOL against fp64 gates of that residual (the op computes
                them in fp32; measured error below 3e-7)
  pre           atol = PRE_ATOL against the fp64 pre gate (its sensitivity to the mixes
                is at most half of post's)

Each row times the op and aiter's existing ops on the same seam:

  mhc_post + mhc_pre    mhc_post, then mhc_pre (mhc_pre_gemm_sqrsum + mhc_pre_big_fuse)
  Triton rmsnorm        ``mhc_fused_post_pre_delayed_rmsnorm``, which folds the RMSNorm
                        of layer_input; timed against the op + ``aiter.rms_norm`` on
                        layer_input (what a caller of this op runs), and compared with
                        it at bf16 tolerance (max |diff| reported, and the Triton op's
                        own gate deviation from the torch fp32 gates)

in us, TFLOPS of the next pre projection (2 T 24 (4 H)) and TB/s of compulsory traffic
(read residual + sub-layer output, write residual_out + layer_input).

For the ASM seam kernel alone (the few-token kernel below MHC_SEAM_SMALL_MAX_T tokens,
the fused kernel from it): its split-K partials part[H / w, m, 32] against an fp64
reference of the same w-column slice sums (w = 128 or 512): |err| <= 5e-5 of the slice's
sum of |terms| (fp32 accumulation), columns 25..31 zero, and a guard band after part
untouched.

Launch split, graphs and compile: T = 65536 + 37 (two launches, byte offsets past 2^31)
with carried and identity pre gate against the same references; CUDA graphs captured at
16 and 128 tokens replay bit-exact after that larger eager call (which must not free a
buffer a graph reads); ``torch.compile(fullgraph=True, dynamic=True)`` equals eager bit for
bit.

Guards: m = 0 gives empty results of the right shapes; hidden_size 4096 and 7168,
hc_mult 2 and another arch raise NotImplementedError pointing to
mhc_fused_post_pre_delayed_asm_supported; a tensor of the wrong shape or on the wrong
device raises ValueError and of the wrong dtype TypeError, before any launch; a wrongly
shaped, typed or placed tensor passed to the ASM bindings directly fails a check in the C
loader and raises RuntimeError (the process continues).
"""

import argparse
import sys

import torch

import aiter
from aiter import dtypes
from aiter.benchmark_reporting import print_json_table
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

HC = 4
HC3 = 2 * HC + HC * HC
# DeepSeek-V4.1-Flash config.json: rms_norm_eps (also the RMSNorm of layer_input),
# hc_eps (pre and Sinkhorn), post mult, Sinkhorn
RMS_EPS, HC_EPS, POST_MULT, SINKHORN = 1e-20, 1e-6, 2.0, 20

torch.set_default_device("cuda")


def make_inputs(m, hidden_size, seed=0, fn_scale=2e-3):
    """Seam inputs with the model's shapes and coefficient ranges."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, s=1.0):
        return torch.randn(*shape, generator=g, device="cuda") * s

    def rand(*shape):
        return torch.rand(*shape, generator=g, device="cuda")

    comb = torch.softmax(randn(m, HC, HC, s=2.0), -1)
    comb = comb / comb.sum(-2, keepdim=True)
    return {
        "residual": randn(m, HC, hidden_size).to(dtypes.bf16),
        "sublayer_out": randn(m, hidden_size, s=0.5).to(dtypes.bf16),
        "post_layer_mix": rand(m, HC, 1) * POST_MULT,
        "comb_res_mix": comb.contiguous(),
        "pre_mix": rand(m, HC) + HC_EPS,
        "fn": randn(HC3, HC * hidden_size, s=fn_scale),
        "hc_scale": 0.5 + rand(3),
        "hc_base": randn(HC3, s=0.5),
        "norm_weight": (1.0 + randn(hidden_size, s=0.1)).to(dtypes.bf16),
    }


def collapse_ref(residual_out, pre_mix):
    """Sequential fp32 multiply and add over the streams from 0, rounded to nearest even."""
    acc = torch.zeros(residual_out.shape[0], residual_out.shape[2])
    for j in range(HC):
        acc = acc + pre_mix[:, j : j + 1] * residual_out[:, j].float()
    return acc.to(dtypes.bf16)


# Gate bounds against the fp64 reference (the op's gates are within 3e-7 of it).
GATE_ATOL = 1e-6
PRE_ATOL = 5e-7


def gates_ref(residual_out, fn, hc_scale, hc_base, dtype=torch.float64, rows=4096):
    """Gates of ``mhc_pre`` (vLLM's mhc_pre_delayed_torch) computed in `dtype`, over
    `rows` tokens at a time; returned in fp32."""
    outs = [
        _gates_ref(residual_out[i : i + rows], fn, hc_scale, hc_base, dtype)
        for i in range(0, residual_out.shape[0], rows)
    ]
    if not outs:
        outs = [_gates_ref(residual_out, fn, hc_scale, hc_base, dtype)]
    return tuple(torch.cat(o).float() for o in zip(*outs))


def _gates_ref(residual_out, fn, hc_scale, hc_base, dtype):
    fn, hc_scale, hc_base = fn.to(dtype), hc_scale.to(dtype), hc_base.to(dtype)
    xf = residual_out.flatten(1).to(dtype)
    mixes = (xf @ fn.t()) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + RMS_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * hc_scale[0] + hc_base[:HC]) + HC_EPS
    post = torch.sigmoid(mixes[:, HC : 2 * HC] * hc_scale[1] + hc_base[HC : 2 * HC])
    post = post * POST_MULT
    comb = mixes[:, 2 * HC :].view(-1, HC, HC) * hc_scale[2]
    comb = comb + hc_base[2 * HC :].view(1, HC, HC)
    comb = torch.softmax(comb, dim=-1) + HC_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)
    return post.unsqueeze(-1), comb, pre


def part_ref(residual_out, fn, sl=512):
    """fp64 slice partials [H / sl, m, 25] of residual_out @ fn^T (+ sum of squares)
    and the matching sums of |terms|."""
    m, _, hidden_size = residual_out.shape
    nc = hidden_size // sl
    r = residual_out.double().view(m, HC, nc, sl).permute(2, 0, 1, 3)
    r = r.reshape(nc, m, HC * sl)
    f = fn.double().view(HC3, HC, nc, sl).permute(2, 1, 3, 0).reshape(nc, HC * sl, HC3)
    sq = (r * r).sum(-1, keepdim=True)
    return torch.cat([r @ f, sq], -1), torch.cat([r.abs() @ f.abs(), sq], -1)


def max_diff(a, b):
    return (a.float() - b.float()).abs().max().item() if a.numel() else 0.0


def check_outputs(out, res_ref, pre_ref, g_ref, tag):
    """checkAllclose mismatch ratios of the op's outputs: residual_out and layer_input
    bit-exact, the gates within GATE_ATOL / PRE_ATOL of the fp64 gates g_ref."""
    residual_out, post_mix, comb_mix, layer_input, next_pre = out
    return {
        "residual_out err": checkAllclose(
            residual_out, res_ref, atol=0, rtol=0, msg=tag + "residual_out vs mhc_post "
        ),
        "layer_input err": checkAllclose(
            layer_input,
            collapse_ref(res_ref, pre_ref),
            atol=0,
            rtol=0,
            msg=tag + "layer_input vs fp32 collapse ",
        ),
        "post err": checkAllclose(
            post_mix, g_ref[0], atol=GATE_ATOL, rtol=0, msg=tag + "post vs fp64 gates "
        ),
        "comb err": checkAllclose(
            comb_mix, g_ref[1], atol=GATE_ATOL, rtol=0, msg=tag + "comb vs fp64 gates "
        ),
        "pre err": checkAllclose(
            next_pre, g_ref[2], atol=PRE_ATOL, rtol=0, msg=tag + "pre vs fp64 gates "
        ),
    }


def perf(m, hidden_size, us):
    """TFLOPS of the next pre projection and TB/s of the compulsory traffic."""
    flops = 2 * m * HC3 * HC * hidden_size
    nbytes = m * hidden_size * 2 * (2 * HC + 2)
    return flops / us / 1e6, nbytes / us / 1e6


SEAM_ARGS = ("residual", "sublayer_out", "post_layer_mix", "comb_res_mix", "fn")
GATE_ARGS = ("hc_scale", "hc_base")


def seam_new(residual, sublayer_out, post, comb, fn, hc_scale, hc_base, pre_mix):
    return aiter.mhc_fused_post_pre_delayed(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
        pre_mix,
        sublayer_out,
        post,
        comb,
    )


def seam_new_rmsnorm(*args, norm_weight):
    """The op followed by the RMSNorm of layer_input, as a caller runs it."""
    residual_out, post_mix, comb_mix, layer_input, next_pre = seam_new(*args)
    normed = aiter.rms_norm(layer_input, norm_weight, RMS_EPS)
    return residual_out, post_mix, comb_mix, normed, next_pre


def seam_triton_rmsnorm(
    residual, sublayer_out, post, comb, fn, hc_scale, hc_base, pre_mix, norm_weight
):
    """aiter's Triton fused delayed seam, which folds the RMSNorm of layer_input."""
    from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
        mhc_fused_post_pre_delayed_rmsnorm,
    )

    return mhc_fused_post_pre_delayed_rmsnorm(
        residual,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
        pre_mix,
        sublayer_out,
        post,
        comb,
        norm_weight,
        RMS_EPS,
    )


def seam_aiter_unfused(residual, sublayer_out, post, comb, fn, hc_scale, hc_base):
    """aiter's existing ops on the same seam: mhc_post, then mhc_pre (split-K GEMM +
    big_fuse). It produces post and comb but collapses with its own pre."""
    residual_out = torch.empty_like(residual)
    aiter.mhc_post(residual_out, sublayer_out, residual, post, comb)
    post_mix, comb_mix, _ = aiter.mhc_pre(
        residual_out,
        fn,
        hc_scale,
        hc_base,
        RMS_EPS,
        HC_EPS,
        HC_EPS,
        POST_MULT,
        SINKHORN,
    )
    return residual_out, post_mix, comb_mix


# against the Triton rmsnorm op: every output at bf16 tolerance; a mismatch ratio up to
# checkAllclose's tol_err_ratio passes (the RMSNorm rounds a bf16 layer_input here and
# an fp32 collapse there, so a few elements differ by 2 bf16 ulps)
TRITON_TOL = 1e-2
TRITON_ERR_RATIO = 0.05


@benchmark()
def test_mhc_fused_post_pre_delayed(m, hidden_size=5120, identity_pre=False, seed=0):
    inp = make_inputs(m, hidden_size, seed=seed)
    args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
    w = inp["norm_weight"]
    pre_mix = None if identity_pre else inp["pre_mix"]
    pre_ref = inp["pre_mix"]
    if identity_pre:
        pre_ref = torch.zeros(m, HC)
        pre_ref[:, 0] = 1.0

    (res_a, post_a, comb_a), aiter_us = run_perftest(seam_aiter_unfused, *args)
    out, us = run_perftest(seam_new, *args, pre_mix)
    out_n, norm_us = run_perftest(seam_new_rmsnorm, *args, pre_mix, norm_weight=w)
    out_t, triton_us = run_perftest(seam_triton_rmsnorm, *args, pre_mix, w)
    _, post_mix, comb_mix, _, next_pre = out

    g_ref = gates_ref(res_a, inp["fn"], inp["hc_scale"], inp["hc_base"])
    diff_post_a = max_diff(post_a, g_ref[0])
    diff_comb_a = max_diff(comb_a, g_ref[1])
    tag = f"m={m} identity_pre={identity_pre} "
    err = check_outputs(out, res_a, pre_ref, g_ref, tag)

    # the op + rms_norm against the Triton rmsnorm op (residual_out, post, comb,
    # normed layer_input, next pre)
    names = ("residual_out", "post", "comb", "layer_input", "pre")
    err_t = 0.0
    diff_t = {}
    for name, a, b in zip(names, out_n, out_t):
        e = checkAllclose(
            a,
            b,
            atol=TRITON_TOL,
            rtol=TRITON_TOL,
            tol_err_ratio=TRITON_ERR_RATIO,
            msg=tag + f"{name} vs Triton rmsnorm op ",
        )
        err_t = max(err_t, e)
        diff_t[name] = max_diff(a, b)
    # the Triton op's own gates against the fp64 gates of its residual_out
    g_ref_t = gates_ref(out_t[0], inp["fn"], inp["hc_scale"], inp["hc_base"])
    diff_gates_t = max(
        max_diff(out_t[1], g_ref_t[0]),
        max_diff(out_t[2], g_ref_t[1]),
        max_diff(out_t[4], g_ref_t[2]),
    )

    tflops, tbs = perf(m, hidden_size, us)
    tflops_a, tbs_a = perf(m, hidden_size, aiter_us)
    tflops_n, tbs_n = perf(m, hidden_size, norm_us)
    tflops_t, tbs_t = perf(m, hidden_size, triton_us)
    return {
        "gfx": get_gfx(),
        "us": us,
        "TFLOPS": tflops,
        "TB/s": tbs,
        "mhc_post+mhc_pre us": aiter_us,
        "mhc_post+mhc_pre TFLOPS": tflops_a,
        "mhc_post+mhc_pre TB/s": tbs_a,
        "op+rms_norm us": norm_us,
        "op+rms_norm TFLOPS": tflops_n,
        "op+rms_norm TB/s": tbs_n,
        "Triton rmsnorm us": triton_us,
        "Triton rmsnorm TFLOPS": tflops_t,
        "Triton rmsnorm TB/s": tbs_t,
        **err,
        "post |diff|": max_diff(post_mix, g_ref[0]),
        "post |diff| mhc_pre": diff_post_a,
        "comb |diff|": max_diff(comb_mix, g_ref[1]),
        "comb |diff| mhc_pre": diff_comb_a,
        "pre |diff|": max_diff(next_pre, g_ref[2]),
        "vs Triton err": err_t,
        "vs Triton residual_out |diff|": diff_t["residual_out"],
        "vs Triton layer_input |diff|": diff_t["layer_input"],
        "vs Triton gates |diff|": max(diff_t["post"], diff_t["comb"], diff_t["pre"]),
        "Triton gates |diff| vs fp64": diff_gates_t,
        "ok": all(e == 0 for e in err.values()) and err_t <= TRITON_ERR_RATIO,
    }


@benchmark()
def test_mhc_seam_fused_part(m, kernel="fused", seed=0):
    """The split-K partials of an ASM seam kernel against an fp64 reference: the fused
    kernel (512-column chunks) or the few-token kernel (128-column slices)."""
    from aiter.ops.mhc import mhc_seam_fused_asm, mhc_seam_small_asm

    hidden_size = 5120
    sl = 512 if kernel == "fused" else 128
    inp = make_inputs(m, hidden_size, seed=seed)
    residual_out = torch.empty_like(inp["residual"])
    layer_input = torch.empty(m, hidden_size, dtype=dtypes.bf16)
    n = hidden_size // sl * m * 32
    buf = torch.full((n + 1024,), -7.0)  # a guard band after part must stay untouched
    part = buf[:n].view(hidden_size // sl, m, 32)
    seam_in = (
        residual_out,
        layer_input,
        part,
        inp["residual"],
        inp["sublayer_out"],
        inp["post_layer_mix"].view(m, HC),
        inp["comb_res_mix"],
        inp["pre_mix"],
        inp["fn"],
    )
    if kernel == "fused":
        _, us = run_perftest(mhc_seam_fused_asm, *seam_in, 0, m >= 8192)
    else:
        _, us = run_perftest(mhc_seam_small_asm, *seam_in)
    ref, mag = part_ref(residual_out, inp["fn"], sl)
    rel = ((part[:, :, :25].double() - ref).abs() / (mag + 1e-300)).max().item()
    zeros = bool((part[:, :, 25:] == 0).all())
    guard = bool((buf[n:] == -7.0).all())
    tflops, tbs = perf(m, hidden_size, us)
    return {
        "gfx": get_gfx(),
        "kernel": kernel,
        "us": us,
        "TFLOPS": tflops,
        "TB/s": tbs,
        "part rel err": rel,
        "cols 25..31 zero": zeros,
        "guard intact": guard,
        "ok": rel <= 5e-5 and zeros and guard,
    }


def check_guards(aiter_mhc):
    """Outside gfx942 / 4 streams / hidden_size 5120 the op raises NotImplementedError
    (pointing to the probe) and the probe returns False. Returns the failure count."""
    probe = aiter_mhc.mhc_fused_post_pre_delayed_asm_supported
    failed = 0

    def expect_raise(name, call):
        nonlocal failed
        try:
            call()
        except NotImplementedError as exc:
            if "mhc_fused_post_pre_delayed_asm_supported" in str(exc):
                aiter.logger.info("guard ok: %s raises: %s", name, exc)
                return
            aiter.logger.error("guard %s: unexpected message: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            aiter.logger.error("guard %s: %s: %s", name, type(exc).__name__, exc)
        else:
            aiter.logger.error("guard %s: no error raised", name)
        failed += 1

    def call_with(m, hidden_size, hc_mult=HC):
        inp = make_inputs(m, hidden_size)
        if hc_mult != HC:
            hc3 = 2 * hc_mult + hc_mult * hc_mult
            inp["residual"] = inp["residual"][:, :hc_mult].contiguous()
            inp["post_layer_mix"] = inp["post_layer_mix"][:, :hc_mult].contiguous()
            inp["comb_res_mix"] = inp["comb_res_mix"][:, :hc_mult, :hc_mult]
            inp["pre_mix"] = inp["pre_mix"][:, :hc_mult].contiguous()
            inp["fn"] = inp["fn"][:hc3, : hc_mult * hidden_size].contiguous()
            inp["hc_base"] = inp["hc_base"][:hc3].contiguous()
        return lambda: seam_new(
            *(inp[k] for k in SEAM_ARGS + GATE_ARGS), inp["pre_mix"]
        )

    for hidden_size in (4096, 7168):
        expect_raise(f"hidden_size {hidden_size}", call_with(16, hidden_size))
    expect_raise("hc_mult 2", call_with(16, 5120, hc_mult=2))
    gfx = aiter_mhc.get_gfx_runtime
    aiter_mhc.get_gfx_runtime = lambda: "gfx950"
    try:
        expect_raise("arch gfx950", call_with(16, 5120))
    finally:
        aiter_mhc.get_gfx_runtime = gfx

    want = {
        (5120, 4, "gfx942"): True,
        (4096, 4, "gfx942"): False,
        (7168, 4, "gfx942"): False,
        (5120, 2, "gfx942"): False,
        (5120, 4, "gfx950"): False,
    }
    for (hidden_size, hc_mult, arch), ok in want.items():
        if probe(hidden_size, hc_mult, arch) != ok:
            aiter.logger.error(
                "probe(%d, %d, %s) != %s", hidden_size, hc_mult, arch, ok
            )
            failed += 1
    return failed


def check_loader_errors(aiter_mhc):
    """A tensor the C loader rejects (wrong shape or dtype, passed to the ASM bindings
    directly, with no Python check in between) raises RuntimeError with the loader's
    message, and the process goes on: a valid seam afterwards still matches mhc_post.
    Returns the failure count."""
    m = 16
    inp = make_inputs(m, 5120, seed=7)
    seam_out = (
        torch.empty_like(inp["residual"]),
        torch.empty(m, 5120, dtype=dtypes.bf16),
    )
    seam_in = (
        inp["residual"],
        inp["sublayer_out"],
        inp["post_layer_mix"].view(m, HC),
        inp["comb_res_mix"],
        inp["pre_mix"],
    )
    gates_out = (torch.empty(m, HC, 1), torch.empty(m, HC, HC), torch.empty(m, HC))
    gate_consts = (RMS_EPS, HC_EPS, HC_EPS, POST_MULT, SINKHORN, HC * 5120)
    cases = {
        "gates: hc_scale (4,)": (
            "hc_scale dim 0",
            lambda: aiter_mhc.mhc_seam_gates_asm(
                *gates_out,
                torch.zeros(40, m, 32),
                torch.ones(4),
                inp["hc_base"],
                *gate_consts,
            ),
        ),
        "fused: fn bf16": (
            "fn has dtype",
            lambda: aiter_mhc.mhc_seam_fused_asm(
                *seam_out,
                torch.empty(10, m, 32),
                *seam_in,
                inp["fn"].to(dtypes.bf16),
                0,
                False,
            ),
        ),
        "small: part (10, m, 32)": (
            "part dim 0",
            lambda: aiter_mhc.mhc_seam_small_asm(
                *seam_out, torch.empty(10, m, 32), *seam_in, inp["fn"]
            ),
        ),
        "fused: fn on the CPU": (
            "fn is on device -1",
            lambda: aiter_mhc.mhc_seam_fused_asm(
                *seam_out,
                torch.empty(10, m, 32),
                *seam_in,
                inp["fn"].cpu(),
                0,
                False,
            ),
        ),
    }
    failed = 0
    for name, (expect, call) in cases.items():
        try:
            call()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if expect in str(exc):
                aiter.logger.info("loader guard ok: %s raises: %s", name, exc)
                continue
            aiter.logger.error("loader guard %s: unexpected message: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            aiter.logger.error("loader guard %s: %s: %s", name, type(exc).__name__, exc)
        else:
            aiter.logger.error("loader guard %s: no error raised", name)
        failed += 1

    args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
    res_a = seam_aiter_unfused(*args)[0]
    residual_out = seam_new(*args, inp["pre_mix"])[0]
    if checkAllclose(
        residual_out, res_a, atol=0, rtol=0, msg="seam after loader errors "
    ):
        failed += 1
    return failed


def check_input_errors():
    """Wrong shapes, dtypes or devices raise ValueError or TypeError in the op, before
    any launch, instead of being reshaped or cast. Returns the failure count."""
    m = 16
    inp = make_inputs(m, 5120, seed=3)

    def call(**over):
        d = {**inp, **over}
        return lambda: seam_new(*(d[k] for k in SEAM_ARGS + GATE_ARGS), d["pre_mix"])

    cases = {
        "residual on the CPU": (
            ValueError,
            "residual must be",
            call(residual=inp["residual"].cpu()),
        ),
        "fn on the CPU": (ValueError, "fn is on cpu", call(fn=inp["fn"].cpu())),
        "hc_scale on the CPU": (
            ValueError,
            "hc_scale is on cpu",
            call(hc_scale=inp["hc_scale"].cpu()),
        ),
        "post_layer_mix bf16": (
            TypeError,
            "post_layer_mix has dtype",
            call(post_layer_mix=inp["post_layer_mix"].to(dtypes.bf16)),
        ),
        "sublayer_out fp32": (
            TypeError,
            "sublayer_out has dtype",
            call(sublayer_out=inp["sublayer_out"].float()),
        ),
        "pre_mix (4, T)": (
            ValueError,
            "pre_mix has shape",
            call(pre_mix=inp["pre_mix"].t().contiguous()),
        ),
        "comb_res_mix (T, 16)": (
            ValueError,
            "comb_res_mix has shape",
            call(comb_res_mix=inp["comb_res_mix"].reshape(m, HC * HC)),
        ),
    }
    failed = 0
    for name, (exc_type, expect, fn) in cases.items():
        try:
            fn()
            torch.cuda.synchronize()
        except exc_type as exc:
            if expect in str(exc):
                aiter.logger.info("input check ok: %s raises: %s", name, exc)
                continue
            aiter.logger.error("input check %s: unexpected message: %s", name, exc)
        except Exception as exc:  # noqa: BLE001
            aiter.logger.error("input check %s: %s: %s", name, type(exc).__name__, exc)
        else:
            aiter.logger.error("input check %s: no error raised", name)
        failed += 1
    return failed


def identity_gate(m):
    pre = torch.zeros(m, HC)
    pre[:, 0] = 1.0
    return pre


def check_against_refs(out, inp, pre_mix, tag):
    """The op's outputs against mhc_post, the fp32 collapse and the fp64 gates.
    Returns the failure count."""
    res_ref = torch.empty_like(inp["residual"])
    aiter.mhc_post(
        res_ref,
        inp["sublayer_out"],
        inp["residual"],
        inp["post_layer_mix"],
        inp["comb_res_mix"],
    )
    m = res_ref.shape[0]
    pre_ref = identity_gate(m) if pre_mix is None else pre_mix
    g_ref = gates_ref(res_ref, inp["fn"], inp["hc_scale"], inp["hc_base"])
    err = check_outputs(out, res_ref, pre_ref, g_ref, tag)
    return sum(e != 0 for e in err.values())


def outputs_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


def check_graphs_and_launch_split(aiter_mhc, seed):
    """CUDA graphs captured at 16 and 128 tokens (carried and identity pre gate), then
    eager calls of MHC_SEAM_MAX_T_PER_LAUNCH + 37 tokens (two launches, byte offsets past
    2^31) checked against the references, then the graph replayed with fresh NaN
    allocations around: every output must equal its eager result bit for bit (the
    larger call must not free a buffer the graph reads). Returns the failure count."""
    failed = 0
    cases = []
    for m in (16, 128):
        for identity in (False, True):
            inp = make_inputs(m, 5120, seed=seed + m)
            args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
            pre = None if identity else inp["pre_mix"]
            eager = [t.clone() for t in seam_new(*args, pre)]
            cases.append((f"graph m={m} identity_pre={identity}", args, pre, eager))

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _, args, pre, _ in cases:
            seam_new(*args, pre)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [seam_new(*args, pre) for _, args, pre, _ in cases]

    m = aiter_mhc.MHC_SEAM_MAX_T_PER_LAUNCH + 37
    for identity in (False, True):
        inp = make_inputs(m, 5120, seed=seed + 1)
        pre = None if identity else inp["pre_mix"]
        out = seam_new(*(inp[k] for k in SEAM_ARGS + GATE_ARGS), pre)
        bad = check_against_refs(out, inp, pre, f"m={m} identity_pre={identity} ")
        if bad:
            aiter.logger.error("launch split m=%d identity_pre=%s failed", m, identity)
        failed += bad
        del inp, out
    junk = [torch.full((65536, HC), float("nan")) for _ in range(64)]

    graph.replay()
    torch.cuda.synchronize()
    for (name, _, _, eager), out in zip(cases, outs):
        if outputs_equal(out, eager):
            aiter.logger.info("%s: replay after a larger eager call equals eager", name)
        else:
            aiter.logger.error("%s: replay differs from eager", name)
            failed += 1
    del junk, graph
    return failed


def check_compile(seed):
    """torch.compile(fullgraph=True, dynamic=True) equals eager bit for bit. Returns the
    failure count."""
    compiled = torch.compile(seam_new, fullgraph=True, dynamic=True)
    failed = 0
    for m in (16, 128):
        for identity in (False, True):
            inp = make_inputs(m, 5120, seed=seed + 2 * m)
            args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
            pre = None if identity else inp["pre_mix"]
            if outputs_equal(compiled(*args, pre), seam_new(*args, pre)):
                aiter.logger.info(
                    "compile m=%d identity_pre=%s equals eager", m, identity
                )
            else:
                aiter.logger.error("compile m=%d identity_pre=%s differs", m, identity)
                failed += 1
    return failed


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="delayed mHC seam (mhc_fused_post_pre_delayed)",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[
            1,
            7,
            16,
            39,
            40,
            59,
            60,
            64,
            128,
            333,
            1024,
            4096,
            8192,
            16384,
            32768,
        ],
        help="""Token counts.
    e.g.: -m 16384""",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if get_gfx() != "gfx942":
        aiter.logger.warning(
            "mhc_fused_post_pre_delayed runs on gfx942 only, skipping on %s",
            get_gfx(),
        )
        return 0

    from aiter.ops import mhc as aiter_mhc

    df = []
    for m in args.m:
        for identity_pre in (False, True):
            df.append(
                test_mhc_fused_post_pre_delayed(
                    m=m, identity_pre=identity_pre, seed=args.seed + m
                )
            )
    print_json_table("mhc_fused_post_pre_delayed summary", df)
    failed = sum(not r["ok"] for r in df)

    small_max = aiter_mhc.MHC_SEAM_SMALL_MAX_T
    dp = [
        test_mhc_seam_fused_part(m=m, kernel="fused", seed=args.seed + m)
        for m in args.m
        if m >= small_max
    ]
    dp += [
        test_mhc_seam_fused_part(m=m, kernel="small", seed=args.seed + m)
        for m in args.m
        if m < small_max
    ]
    print_json_table("ASM seam kernel partials summary", dp)
    failed += sum(not r["ok"] for r in dp)

    # no tokens: empty results of the right shapes
    inp = make_inputs(0, 5120)
    out = seam_new(*(inp[k] for k in SEAM_ARGS + GATE_ARGS), inp["pre_mix"])
    want = [(0, HC, 5120), (0, HC, 1), (0, HC, HC), (0, 5120), (0, HC)]
    got = [tuple(t.shape) for t in out]
    if got != want:
        aiter.logger.error("m=0: shapes %s", got)
        failed += 1
    failed += check_guards(aiter_mhc)
    failed += check_loader_errors(aiter_mhc)
    failed += check_input_errors()
    failed += check_compile(args.seed)
    failed += check_graphs_and_launch_split(aiter_mhc, args.seed)
    aiter.logger.info("test_mhc_seam: %d failure(s)", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
