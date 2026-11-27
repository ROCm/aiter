# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Delayed mHC seam: the ASM op (``mhc_fused_post_pre_delayed``) and the FlyDSL op
(``aiter.ops.flydsl.flydsl_mhc_seam``) against aiter's own mHC ops and torch references.

Both run on gfx942 only; on other archs this test is skipped. The ASM op takes 4
residual streams and hidden_size 5120 with the post block; the FlyDSL op also runs
without it (mode no_post) and at the other hidden sizes (--hidden), where the ASM
columns are left out. Per ASM case (checkAllclose; the err columns are its mismatch
ratios):

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

Per FlyDSL case: residual_out within FLY_RES_TOL of the post block in fp32 rounded to
bf16, layer_input within FLY_LAYER_INPUT_TOL of its fp32 collapse and RMSNorm
(``flydsl_ref``), the gates within FLY_GATE_ATOL of the fp64 gates of the op's own
residual_out; eager and after capture in a CUDA graph and two replays, with the post
block into a preallocated NaN residual_out.
Also for FlyDSL: m = 0 in every mode; each RMSNorm path (fused, separate) at its policy
config; a graph after an in-place update of fn, and after a write through fn.data with
an explicit fn_packed refreshed. Per hidden size: two seams sharing one fn on two
streams, builds the launch policy does not reach, forced, and invalid inputs, which
raise ValueError or TypeError. With two visible gfx942 GPUs, calls alternate between
them with both matching and different current devices.

Launch split, graphs and compile: T = 65536 + 37 (two launches, byte offsets past 2^31)
with carried and identity pre gate against the same references; CUDA graphs captured at
16 and 128 tokens replay bit-exact after that larger eager call (which must not free a
buffer a graph reads); a graph captured with the identity gate while its cache is empty
replays bit-exact; ``torch.compile(fullgraph=True, dynamic=True)`` equals eager bit for
bit, also for a permuted (non-contiguous) residual.

Guards: m = 0 gives empty results of the right shapes; hidden_size 4096 and 7168,
hc_mult 2 and another arch raise NotImplementedError pointing to
mhc_fused_post_pre_delayed_asm_supported; a tensor of the wrong shape or on the wrong
device, or sinkhorn_repeat < 1, raises ValueError and of the wrong dtype TypeError, before any launch; a wrongly
shaped, typed or placed tensor passed to the ASM bindings directly fails a check in the C
loader and raises RuntimeError (the process continues).
"""

import argparse
import itertools
import sys

import torch

import aiter
from aiter import dtypes
from aiter.benchmark_reporting import print_json_table
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

HC = 4
HC3 = 2 * HC + HC * HC
# rms_eps (also the RMSNorm of layer_input), hc_pre_eps, hc_sinkhorn_eps, post mult,
# Sinkhorn repeats. DeepSeek-V4.1-Flash uses 1e-20, 1e-6, 1e-6, 2.0, 20; the eps values
# here are distinct and large enough to move the gates, so swapped arguments fail.
RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, POST_MULT, SINKHORN = 1e-2, 1e-3, 1e-4, 2.0, 20

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
        "pre_mix": rand(m, HC) + HC_PRE_EPS,
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
    pre = torch.sigmoid(mixes[:, :HC] * hc_scale[0] + hc_base[:HC]) + HC_PRE_EPS
    post = torch.sigmoid(mixes[:, HC : 2 * HC] * hc_scale[1] + hc_base[HC : 2 * HC])
    post = post * POST_MULT
    comb = mixes[:, 2 * HC :].view(-1, HC, HC) * hc_scale[2]
    comb = comb + hc_base[2 * HC :].view(1, HC, HC)
    comb = torch.softmax(comb, dim=-1) + HC_SINKHORN_EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS)
    for _ in range(SINKHORN - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_SINKHORN_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_SINKHORN_EPS)
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
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
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
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
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
        HC_PRE_EPS,
        HC_SINKHORN_EPS,
        POST_MULT,
        SINKHORN,
    )
    return residual_out, post_mix, comb_mix


# against the Triton rmsnorm op: every output at bf16 tolerance; a mismatch ratio up to
# checkAllclose's tol_err_ratio passes (the RMSNorm rounds a bf16 layer_input here and
# an fp32 collapse there, so a few elements differ by 2 bf16 ulps)
TRITON_TOL = 1e-2
TRITON_ERR_RATIO = 0.05


# The FlyDSL seam (aiter.ops.flydsl.flydsl_mhc_seam) folds the RMSNorm of layer_input,
# also runs without the post block, and takes other hidden sizes than the ASM seam.
# Modes:
#   post          the post block mixes sublayer_out into the residual streams; the
#                 carried pre_mix collapses them
#   identity_pre  the post block, then pre_mix=None: the collapse is stream 0
#   no_post       no post block: the residual passes through as residual_out; the
#                 carried pre_mix collapses it
MODES = ("post", "identity_pre", "no_post")

# (rtol, atol) of the FlyDSL seam against flydsl_ref. residual_out: its fp32 post block
# sums in another order, then rounds to bf16. layer_input: a bf16 rounding difference
# in residual_out carries into the collapse.
FLY_RES_TOL = (1.6e-2, 1e-5)
FLY_LAYER_INPUT_TOL = (1e-2, 2e-2)
# The FlyDSL gates against the fp64 gates of its own residual_out, as the ASM gates are
# checked: fp32 split-K sums, well within this. Dropping the low bf16 limb of the
# packed fn moves them by over 3e-4.
FLY_GATE_ATOL = 1e-5


def flydsl_ref(inp, mode):
    """residual_out and layer_input of the FlyDSL seam in torch: the post block in fp32
    rounded to bf16, then its fp32 collapse rounded to bf16 and RMSNorm (eps RMS_EPS) in
    fp32 rounded to bf16."""
    residual = inp["residual"]
    m = residual.shape[0]
    res = residual
    if mode != "no_post":
        mixed = torch.einsum(
            "tij,tih->tjh", inp["comb_res_mix"].float(), residual.float()
        )
        post = inp["post_layer_mix"].float().view(m, HC, 1)
        res = (mixed + post * inp["sublayer_out"].float().unsqueeze(1)).to(dtypes.bf16)
    pre = identity_gate(m) if mode == "identity_pre" else inp["pre_mix"]
    x = collapse_ref(res, pre).float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    return res, (x * inp["norm_weight"].float()).to(dtypes.bf16)


def seam_flydsl(inp, mode, **kw):
    """The FlyDSL seam as the model calls it, as (call, activations): with the post
    block it writes into a preallocated residual_out (NaN, so an unwritten element
    fails). fn stays bound in call, since run_perftest deep-copies the arguments it
    rotates and the seam would repack a copied fn on every call."""
    from aiter.ops.flydsl import flydsl_mhc_seam

    post = mode != "no_post"

    def call(residual, sublayer_out, post_mix, comb_mix, pre_mix, residual_out):
        return flydsl_mhc_seam(
            residual,
            inp["fn"],
            inp["hc_scale"],
            inp["hc_base"],
            RMS_EPS,
            HC_PRE_EPS,
            HC_SINKHORN_EPS,
            POST_MULT,
            SINKHORN,
            pre_mix,
            sublayer_out,
            post_mix,
            comb_mix,
            inp["norm_weight"],
            RMS_EPS,
            residual_out=residual_out,
            **kw,
        )

    args = (
        inp["residual"],
        inp["sublayer_out"] if post else None,
        inp["post_layer_mix"] if post else None,
        inp["comb_res_mix"] if post else None,
        None if mode == "identity_pre" else inp["pre_mix"],
        torch.full_like(inp["residual"], float("nan")) if post else None,
    )
    return call, args


def check_flydsl(out, inp, mode, tag):
    """checkAllclose mismatch ratios of the FlyDSL seam's outputs on inp: residual_out
    and layer_input against flydsl_ref, the gates against the fp64 gates of the op's own
    residual_out. Without the post block residual_out is the input itself and is not
    compared."""
    residual_out, post_mix, comb_mix, layer_input, next_pre = out
    res_ref, layer_input_ref = flydsl_ref(inp, mode)
    g_ref = gates_ref(residual_out, inp["fn"], inp["hc_scale"], inp["hc_base"])
    checks = [
        ("residual_out", residual_out, res_ref, FLY_RES_TOL, "fp32 reference"),
        ("post", post_mix, g_ref[0], (0, FLY_GATE_ATOL), "fp64 gates"),
        ("comb", comb_mix, g_ref[1], (0, FLY_GATE_ATOL), "fp64 gates"),
        (
            "layer_input",
            layer_input,
            layer_input_ref,
            FLY_LAYER_INPUT_TOL,
            "fp32 reference",
        ),
        ("pre", next_pre, g_ref[2], (0, FLY_GATE_ATOL), "fp64 gates"),
    ]
    err = {}
    for name, o, r, (rtol, atol), against in checks:
        if name == "residual_out" and mode == "no_post":
            continue
        err[f"flydsl {name} err"] = checkAllclose(
            r.float(),
            o.float(),
            rtol=rtol,
            atol=atol,
            tol_err_ratio=0.0,
            msg=f"{tag}flydsl {name} vs {against} ",
        )
    return err


def flydsl_graph_err(call, args, inp, mode, tag):
    """The seam captured in a CUDA graph after an eager call, replayed twice; the worst
    mismatch ratio of the replays against the reference."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        call(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(*args)
    errs = []
    for i in range(2):
        graph.replay()
        torch.cuda.synchronize()
        err = check_flydsl(out, inp, mode, f"{tag}graph replay {i} ")
        errs.append(max(err.values()))
    return max(errs)


def flydsl_bytes_moved(m, hidden_size, mode):
    """Bytes the FlyDSL seam must move, in bf16 rows of hidden_size per token: read the
    residual's HC rows and write layer_input, plus with the post block read
    sublayer_out and write residual_out's HC rows."""
    rows = HC + 1 if mode == "no_post" else 2 * HC + 2
    return m * hidden_size * 2 * rows


@benchmark()
def test_mhc_fused_post_pre_delayed(m, hidden_size=5120, mode="post", seed=0):
    inp = make_inputs(m, hidden_size, seed=seed)
    ret = {"gfx": get_gfx_runtime()}
    ok = True
    # ASM, Triton and aiter's unfused ops: hidden_size 5120 with the post block only.
    if mode != "no_post" and aiter.mhc_fused_post_pre_delayed_asm_supported(
        hidden_size, HC
    ):
        asm = asm_candidates(inp, m, hidden_size, mode == "identity_pre")
        ok = asm.pop("ok")
        ret.update(asm)

    call, args = seam_flydsl(inp, mode)
    _, us = run_perftest(call, *args)
    tag = f"m={m} H={hidden_size} {mode} "
    err = check_flydsl(call(*args), inp, mode, tag)
    graph_err = flydsl_graph_err(call, args, inp, mode, tag)
    ret["flydsl us"] = us
    ret["flydsl TFLOPS"] = perf(m, hidden_size, us)[0]
    ret["flydsl TB/s"] = flydsl_bytes_moved(m, hidden_size, mode) / us / 1e6
    ret.update(err)
    ret["flydsl graph err"] = graph_err
    ret["ok"] = ok and graph_err == 0 and all(e == 0 for e in err.values())
    return ret


def asm_candidates(inp, m, hidden_size, identity_pre):
    """The ASM seam, the ASM seam + aiter.rms_norm, the Triton rmsnorm seam and aiter's
    unfused ops on one input set: timings and checks."""
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
        "gfx": get_gfx_runtime(),
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
    gate_consts = (RMS_EPS, HC_PRE_EPS, HC_SINKHORN_EPS, POST_MULT, SINKHORN, HC * 5120)
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
        "sinkhorn_repeat 0": (
            ValueError,
            "sinkhorn_repeat must be >= 1",
            lambda: aiter.mhc_fused_post_pre_delayed(
                inp["residual"],
                inp["fn"],
                inp["hc_scale"],
                inp["hc_base"],
                RMS_EPS,
                HC_PRE_EPS,
                HC_SINKHORN_EPS,
                POST_MULT,
                0,
                inp["pre_mix"],
                inp["sublayer_out"],
                inp["post_layer_mix"],
                inp["comb_res_mix"],
            ),
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


def check_graph_cold_identity(aiter_mhc, seed):
    """A CUDA graph captured with the identity pre gate while the identity cache is
    empty (the buffer is then built inside the capture) replays equal to eager, and the
    capture leaves the cache empty. Returns the failure count."""
    m = 16
    inp = make_inputs(m, 5120, seed=seed + 5)
    args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
    eager = [t.clone() for t in seam_new(*args, None)]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        seam_new(*args, inp["pre_mix"])  # warm-up that does not touch the identity
    torch.cuda.current_stream().wait_stream(side)
    aiter_mhc._MHC_IDENTITY_PRE.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = seam_new(*args, None)
    failed = 0
    if aiter_mhc._MHC_IDENTITY_PRE:
        aiter.logger.error("graph with a cold identity cache: capture filled the cache")
        failed += 1
    junk = [torch.full((65536, HC), float("nan")) for _ in range(16)]
    graph.replay()
    torch.cuda.synchronize()
    if outputs_equal(out, eager):
        aiter.logger.info("graph with a cold identity cache: replay equals eager")
    else:
        aiter.logger.error(
            "graph with a cold identity cache: replay differs from eager"
        )
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
    # a permuted (non-contiguous) residual: the op copies it, so the outputs are
    # contiguous, and the fake must say so
    inp = make_inputs(64, 5120, seed=seed + 7)
    args = [inp[k] for k in SEAM_ARGS + GATE_ARGS]
    args[0] = args[0].transpose(1, 2).contiguous().transpose(1, 2)
    try:
        same = outputs_equal(
            compiled(*args, inp["pre_mix"]), seam_new(*args, inp["pre_mix"])
        )
    except Exception as exc:  # noqa: BLE001
        aiter.logger.error(
            "compile permuted residual: %s: %s", type(exc).__name__, str(exc)[:300]
        )
        same = False
    if same:
        aiter.logger.info("compile permuted residual equals eager")
    else:
        aiter.logger.error("compile permuted residual differs from eager")
        failed += 1
    return failed


def check_flydsl_empty():
    """FlyDSL with no tokens, in every mode: empty results of the right shapes, and
    without the post block the input residual itself. Returns the failure count."""
    failed = 0
    for mode in MODES:
        inp = make_inputs(0, 5120)
        call, args = seam_flydsl(inp, mode)
        out = call(*args)
        got = [tuple(t.shape) for t in out]
        want = [(0, HC, 5120), (0, HC, 1), (0, HC, HC), (0, 5120), (0, HC)]
        if got != want or (mode == "no_post" and out[0] is not inp["residual"]):
            aiter.logger.error("flydsl m=0 %s: shapes %s", mode, got)
            failed += 1
    return failed


def check_flydsl_norm_paths(seed):
    """FlyDSL with the RMSNorm in its finish pass (fused) and as aiter.rms_norm after
    it (separate), each at its own policy config, eager and replayed from a CUDA graph.
    Returns the failure count."""
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.mhc_seam_kernels import get_mhc_seam_config

    failed = 0
    for m in (37, 333, 1024):
        for norm in ("fused", "separate"):
            cfg = get_mhc_seam_config(
                m, 5120, get_gfx_runtime(), get_cu_num(), norm=norm
            )
            inp = make_inputs(m, 5120, seed=seed + m)
            call, args = seam_flydsl(inp, "post", config=dict(cfg, NORM=norm))
            tag = f"m={m} NORM={norm} "
            err = check_flydsl(call(*args), inp, "post", tag)
            failed += any(e != 0 for e in err.values())
            failed += flydsl_graph_err(call, args, inp, "post", tag) != 0
    return failed


def check_flydsl_weight_update(seed):
    """A FlyDSL graph follows an in-place update of fn after one eager call, and, with
    an explicit fn_packed, a write through fn.data once the caller repacks into it;
    on both RMSNorm paths. Checked against the reference of the updated fn: a stale
    pack would leave the eager call stale too. Returns the failure count."""
    from aiter.ops.flydsl import flydsl_mhc_seam_pack_fn

    def capture(call, args):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            call(*args)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = call(*args)
        return graph, out

    failed = 0
    for norm in ("fused", "separate"):
        inp = make_inputs(256, 5120, seed=seed + 11)
        call, args = seam_flydsl(inp, "post", config={"NORM": norm})
        graph, out = capture(call, args)
        inp["fn"].mul_(2)
        call(*args)  # the eager call that repacks
        graph.replay()
        torch.cuda.synchronize()
        err = check_flydsl(out, inp, "post", f"weight update in place {norm} ")
        failed += any(e != 0 for e in err.values())

        packed = flydsl_mhc_seam_pack_fn(inp["fn"])
        call, args = seam_flydsl(inp, "post", config={"NORM": norm}, fn_packed=packed)
        graph, out = capture(call, args)
        inp["fn"].data.copy_(inp["fn"] * 0.5)
        packed.copy_(flydsl_mhc_seam_pack_fn(inp["fn"]))
        graph.replay()
        torch.cuda.synchronize()
        err = check_flydsl(out, inp, "post", f"fn_packed refresh {norm} ")
        failed += any(e != 0 for e in err.values())
    return failed


def check_flydsl_multi_device(seed):
    """Two gfx942 GPUs alternate calls with one specialization, on both RMSNorm paths.
    Also call with the other GPU current and check that the op restores it. Returns
    the failure count; skips when fewer than two supported GPUs are visible."""
    from aiter.ops.flydsl import flydsl_mhc_seam_pack_fn

    devices = [
        i
        for i in range(torch.cuda.device_count())
        if torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] == "gfx942"
    ][:2]
    if len(devices) < 2:
        aiter.logger.warning(
            "flydsl multi-device: need two visible gfx942 GPUs; skipping"
        )
        return 0

    failed = 0
    for norm in ("fused", "separate"):
        cases = {}
        for device in devices:
            with torch.cuda.device(device):
                inp = make_inputs(32, 5120, seed=seed + device)
                packed = flydsl_mhc_seam_pack_fn(inp["fn"])
                call, args = seam_flydsl(
                    inp, "post", config={"NORM": norm}, fn_packed=packed
                )
                cases[device] = inp, call, args
                torch.cuda.synchronize(device)

        # Start with a different current device so a cold compile needs the guard too.
        for other_current in (True, False):
            for device in (devices[0], devices[1], devices[0]):
                current = (
                    devices[1 - devices.index(device)] if other_current else device
                )
                inp, call, args = cases[device]
                tag = f"device={device} current={current} NORM={norm} "
                with torch.cuda.device(current):
                    out = call(*args)
                    if torch.cuda.current_device() != current:
                        aiter.logger.error(
                            "%sFlyDSL did not restore the current GPU", tag
                        )
                        failed += 1
                with torch.cuda.device(device):
                    torch.cuda.synchronize(device)
                    err = check_flydsl(out, inp, "post", tag)
                    failed += any(e != 0 for e in err.values())
    return failed


def flydsl_config_error(m, hidden_size, mode, config):
    """Returns the error message the FlyDSL seam raises for this explicit config in
    `mode`, or None if the config is legal. Like the op, it merges config over
    MHC_SEAM_DEFAULTS and calls mhc_seam.check_config, so the geometry rules stay only
    in the kernel module."""
    from aiter.jit.utils.chip_info import get_lds_capacity_bytes
    from aiter.ops.flydsl.kernels import mhc_seam
    from aiter.ops.flydsl.mhc_seam_kernels import MHC_SEAM_DEFAULTS

    cfg = {**MHC_SEAM_DEFAULTS, **config}
    if cfg["NORM"] == "separate" and mode != "post":
        return "NORM='separate' needs the post block and a carried pre_mix"
    try:
        mhc_seam.check_config(
            m,
            hidden_size,
            HC,
            cfg["BLOCK_M"],
            cfg["CHUNK_C"],
            cfg["SPLITS"],
            lds_bytes=get_lds_capacity_bytes(get_gfx_runtime()),
            fast_fp_math=cfg["FAST_FP_MATH"],
            fast_narrow=cfg["FAST_NARROW"],
        )
    except ValueError as exc:
        return str(exc)
    return None


def check_flydsl_streams(hidden_size, seed):
    """Two FlyDSL seams with their own activations and one fn on two streams at once,
    on both RMSNorm paths, with and without split-K; each against its own reference. A
    SPLITS this hidden size does not divide into is skipped. Returns the failure
    count."""
    failed = 0
    for m, splits, norm in itertools.product(
        (1024, 4096), (1, 10), ("fused", "separate")
    ):
        config = {"SPLITS": splits, "NORM": norm}
        why = flydsl_config_error(m, hidden_size, "post", config)
        if why:
            aiter.logger.info(
                "skipping streams m=%d H=%d %s: %s", m, hidden_size, config, why
            )
            continue
        failed += flydsl_stream_failures(m, hidden_size, config, seed)
    return failed


def flydsl_stream_failures(m, hidden_size, config, seed):
    first = make_inputs(m, hidden_size, seed=seed + 21)
    second = dict(make_inputs(m, hidden_size, seed=seed + 22), fn=first["fn"])
    cases = [seam_flydsl(inp, "post", config=config) for inp in (first, second)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = [None, None]
    for _ in range(20):
        for i, (call, args) in enumerate(cases):
            with torch.cuda.stream(streams[i]):
                outs[i] = call(*args)
    torch.cuda.synchronize()
    failed = 0
    for i, inp in enumerate((first, second)):
        tag = f"stream {i} m={m} H={hidden_size} {config} "
        err = check_flydsl(outs[i], inp, "post", tag)
        failed += any(e != 0 for e in err.values())
    return failed


def check_flydsl_forced_configs(hidden_size, seed):
    """FlyDSL builds the launch policy does not reach at 128 tokens, forced; each
    against the reference. A build this hidden size cannot take is skipped. Returns the
    failure count."""
    cases = [
        ("no_post", {"STAGE_NT": True}),
        ("post", {"NORM": "fused", "STAGE_NT": True, "GATE_STEP": False}),
        # The policy prefetches at every size here; force the load-then-use loop.
        ("post", {"PREFETCH": False}),
        ("no_post", {"BLOCK_M": 32, "CHUNK_C": 64, "PREFETCH": False}),
        # Two chunks per segment and two slots per thread, which the policy leaves off.
        ("post", {"NORM": "separate", "SPLITS": hidden_size // 512, "PREFETCH": True}),
    ]
    failed = 0
    for mode, config in cases:
        why = flydsl_config_error(128, hidden_size, mode, config)
        if why:
            aiter.logger.info(
                "skipping forced H=%d %s %s: %s", hidden_size, mode, config, why
            )
            continue
        inp = make_inputs(128, hidden_size, seed=seed + 31)
        call, args = seam_flydsl(inp, mode, config=config)
        tag = f"forced H={hidden_size} {mode} {config} "
        err = check_flydsl(call(*args), inp, mode, tag)
        failed += any(e != 0 for e in err.values())
    return failed


def check_flydsl_input_errors(hidden_size):
    """Invalid FlyDSL calls raise the documented exception type. Returns the failure
    count."""
    m = 4
    inp = make_inputs(m, hidden_size, seed=3)

    def call(mode="post", config=None, **over):
        kw = {} if config is None else {"config": config}
        f, args = seam_flydsl({**inp, **over}, mode, **kw)
        return lambda: f(*args)

    # The illegal SPLITS comes from the validator, on the tile the policy picks for this
    # hidden size: the default tile may not fit it, which would raise for the tile.
    from aiter.jit.utils.chip_info import get_cu_num
    from aiter.ops.flydsl.mhc_seam_kernels import get_mhc_seam_config

    policy = get_mhc_seam_config(m, hidden_size, get_gfx_runtime(), get_cu_num())
    tile = {"BLOCK_M": policy["BLOCK_M"], "CHUNK_C": policy["CHUNK_C"]}
    bad_splits = next(
        sp
        for sp in range(2, hidden_size + 1)
        if flydsl_config_error(m, hidden_size, "post", dict(tile, SPLITS=sp))
    )
    cases = {
        "fp32 residual": (TypeError, call(residual=inp["residual"].float())),
        "hc_mult 2": (ValueError, call(residual=inp["residual"][:, :2].contiguous())),
        "fn shape": (ValueError, call(fn=inp["fn"][:, :-32].contiguous())),
        "fn bf16": (TypeError, call(fn=inp["fn"].to(dtypes.bf16))),
        "no norm_weight": (ValueError, call(norm_weight=None)),
        "post without comb": (ValueError, call(comb_res_mix=None)),
        "unknown knob": (ValueError, call(config={"TILE_K": 64})),
        "unknown norm": (ValueError, call(config={"NORM": "post"})),
        "illegal chunk": (ValueError, call(config={"CHUNK_C": 96})),
        "splits not dividing": (
            ValueError,
            call(config=dict(tile, SPLITS=bad_splits)),
        ),
        "chunk lanes not 2^k": (ValueError, call(config={"CHUNK_C": 320})),
        "block_m above block": (ValueError, call(config={"BLOCK_M": 512})),
    }
    failed = 0
    for name, (exc_type, fn) in cases.items():
        try:
            fn()
        except exc_type as exc:
            aiter.logger.info("flydsl input check ok: %s raises: %s", name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            aiter.logger.error(
                "flydsl input check %s: %s: %s", name, type(exc).__name__, exc
            )
        else:
            aiter.logger.error("flydsl input check %s: no error raised", name)
        failed += 1
    return failed


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="delayed mHC seam: ASM (mhc_fused_post_pre_delayed) and FlyDSL",
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
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="*",
        default=[5120, 4096, 7168],
        help="""Hidden sizes. The ASM seam runs at 5120 only, the FlyDSL seam at each.""",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if get_gfx_runtime() != "gfx942":
        aiter.logger.warning(
            "the ASM and FlyDSL delayed seams run on gfx942 only, skipping on %s",
            get_gfx_runtime(),
        )
        return 0

    from aiter.ops import mhc as aiter_mhc

    df = [
        test_mhc_fused_post_pre_delayed(
            m=m, hidden_size=hidden_size, mode=mode, seed=args.seed + m
        )
        for hidden_size in args.hidden
        for m in args.m
        for mode in MODES
    ]
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
    failed += check_graph_cold_identity(aiter_mhc, args.seed)
    failed += check_graphs_and_launch_split(aiter_mhc, args.seed)
    failed += check_flydsl_empty()
    failed += check_flydsl_norm_paths(args.seed)
    failed += check_flydsl_weight_update(args.seed)
    failed += check_flydsl_multi_device(args.seed)
    for hidden_size in args.hidden:
        failed += check_flydsl_streams(hidden_size, args.seed)
        failed += check_flydsl_forced_configs(hidden_size, args.seed)
        failed += check_flydsl_input_errors(hidden_size)
    aiter.logger.info("test_mhc_seam: %d failure(s)", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
