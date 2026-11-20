# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL hidden-5120 mHC seam and step-end tail (DeepSeek-V4.1-Flash) vs fp32 torch.

Tables:
  test_mhc_post_pre_rmsnorm      the seam (``flydsl_mhc_post_pre_rmsnorm``) next to
                                 AITER's HIP ``mega_mhc`` and ``flydsl_mega_mhc`` on the
                                 same inputs, every mode / out_dtype / aux_mean; plus
                                 a repeat (bitwise) and a poisoned CUDA-graph replay
                                 check of the new kernel.
  test_mhc_post_collapse_rmsnorm the tail (``flydsl_mhc_post_collapse_rmsnorm``) with
                                 and without post-mix, keep_residual and aux_mean.
  test_mhc_seam_dispatch         every split-K configuration the dispatch picks
                                 (1 ... 4096 rows, and with a CU mask: no split-K)
                                 against the reference.
  test_mhc_seam_edges            zero rows, rows scaled by 2^-3 ... 2^3, partial row
                                 block, empty batch.

MXFP8 codes and E8M0 bytes must equal the reference quantization of the kernel's
own bf16 output to the bit.
"""

import argparse
import itertools
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
H = 5120
# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat (DSV4.1)
ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)
NORM_EPS = 1e-6

_FAILURES = []


# ---------------------------------------------------------------- references
def post_mix_torch(residual, sublayer_out, post_mix, comb_mix):
    T = residual.shape[0]
    p = post_mix.reshape(T, 4, 1).float()
    c = comb_mix.reshape(T, 4, 4).float()
    mixed = torch.einsum("thj,thk->tjk", c, residual.float())
    return (p * sublayer_out.float().unsqueeze(1) + mixed).bfloat16()


def rmsnorm_torch(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * w.float()).to(
        torch.bfloat16
    )


def run_torch_seam(
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
    sublayer_out,
    post_mix,
    comb_mix,
    norm_weight,
    norm_eps,
):
    """fp32 reference of the seam: (R', post, comb, layer_input, next_pre, mean)."""
    T = residual.shape[0]
    r = (
        residual
        if sublayer_out is None
        else post_mix_torch(residual, sublayer_out, post_mix, comb_mix)
    )
    flat = r.reshape(T, 4 * H).float()
    mix = flat @ fn.float().t()
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mix[:, 0:4] * rstd * hc_scale[0] + hc_base[0:4]) + hc_pre_eps
    post = torch.sigmoid(mix[:, 4:8] * rstd * hc_scale[1] + hc_base[4:8]) * hc_post_mult
    a = (mix[:, 8:24] * rstd * hc_scale[2] + hc_base[8:24]).reshape(T, 4, 4)
    comb = torch.softmax(a, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    if pre_mix is None:
        x1 = r[:, 0]
    else:
        x1 = torch.einsum("tj,tjh->th", pre_mix.reshape(T, 4).float(), r.float())
        x1 = x1.bfloat16()
    mean = r.float().mean(1).bfloat16()
    return (
        r,
        post.reshape(T, 4, 1),
        comb,
        rmsnorm_torch(x1, norm_weight, norm_eps),
        pre,
        mean,
    )


def mxfp8_torch(y):
    """bf16 (T, K) -> e4m3 codes (T, K) and E8M0 bytes (T, K / 32):
    byte = clamp(ceil(log2(amax / 448)) + 127, 0, 254), codes of y * 2^(127 - byte)."""
    T, K = y.shape
    blocks = y.float().reshape(T, K // 32, 32)
    amax = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    codes = (blocks * torch.exp2(127.0 - sb).unsqueeze(-1)).reshape(T, K)
    return codes.to(torch.float8_e4m3fn), sb.to(torch.uint8)


def run_torch_tail(residual, pre_mix, norm_weight, eps, sublayer_out, post, comb):
    """fp32 reference of the tail: (layer_input, R', mean)."""
    T = residual.shape[0]
    r = (
        residual
        if sublayer_out is None
        else post_mix_torch(residual, sublayer_out, post, comb)
    )
    x1 = torch.einsum("tj,tjh->th", pre_mix.reshape(T, 4).float(), r.float())
    return (
        rmsnorm_torch(x1.bfloat16(), norm_weight, eps),
        r,
        r.float().mean(1).bfloat16(),
    )


# ---------------------------------------------------------------- inputs
def make_inputs(T, seed=0):
    """Rows scaled by 2^-3 ... 2^3 so a per-row scale or rstd mix-up shows."""
    g = torch.Generator(device="cuda").manual_seed(seed + 7919 * T)

    def r(*s):
        return torch.randn(*s, generator=g)

    scale = torch.exp2(torch.randint(-3, 4, (T, 1, 1), generator=g).float())
    return {
        "residual": (r(T, 4, H) * scale).bfloat16(),
        "sublayer_out": (r(T, H) * scale[:, 0]).bfloat16(),
        "fn": r(24, 4 * H) * 0.02,
        "hc_scale": torch.tensor([0.5, 0.7, 0.9]),
        "hc_base": r(24) * 0.3,
        "post": torch.rand(T, 4, 1, generator=g) * 2,
        "comb": torch.softmax(r(T, 4, 4), -1),
        "pre": torch.rand(T, 4, generator=g),
        "w": (r(H) * 0.5 + 1).bfloat16(),
    }


def seam_call(d, mode):
    """(args, kwargs) of the seam call the model makes for ``mode``; fn unpacked."""
    kw = {"norm_weight": d["w"], "norm_eps": NORM_EPS}
    if mode != "no_post":
        kw.update(
            sublayer_out=d["sublayer_out"],
            post_layer_mix=d["post"],
            comb_res_mix=d["comb"],
            residual_out=torch.empty_like(d["residual"]),
        )
    pre = None if mode == "identity_pre" else d["pre"]
    args = (d["residual"], d["fn"], d["hc_scale"], d["hc_base"], *ARGS, pre)
    return args, kw


def seam_ref(d, mode):
    post = mode != "no_post"
    return run_torch_seam(
        d["residual"],
        d["fn"],
        d["hc_scale"],
        d["hc_base"],
        *ARGS,
        None if mode == "identity_pre" else d["pre"],
        d["sublayer_out"] if post else None,
        d["post"] if post else None,
        d["comb"] if post else None,
        d["w"],
        NORM_EPS,
    )


# ---------------------------------------------------------------- checks
def _flat(out):
    res = []
    for x in out:
        if isinstance(x, tuple):
            res.extend(x)
        else:
            res.append(x)
    return res


def _same(a, b):
    return all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        for x, y in zip(_flat(a), _flat(b))
    )


def _err(want, got, msg, rtol, atol):
    return checkAllclose(
        want.float(), got.float(), rtol=rtol, atol=atol, tol_err_ratio=0.0, msg=msg
    )


def check_seam(name, out, ref, mode, out_dtype, aux_mean):
    """Worst mismatch ratio of all seam outputs against the reference."""
    r_o, post_o, comb_o, li, pre_o = out[:5]
    norm = li[0] if out_dtype == "mxfp8" else li
    errs = []
    if mode != "no_post":
        errs.append(_err(ref[0], r_o, f"{name}: residual_out ", 1.6e-2, 1e-5))
    for o, e, tag in ((post_o, ref[1], "post"), (comb_o, ref[2], "comb")):
        errs.append(_err(e, o, f"{name}: {tag} ", 1e-3, 5e-4))
    errs.append(_err(ref[4], pre_o, f"{name}: next_pre ", 1e-3, 5e-4))
    # one bf16 ulp of R' can carry into the collapse
    errs.append(_err(ref[3], norm, f"{name}: layer_input ", 1e-2, 2e-2))
    if out_dtype == "mxfp8":
        q_r, e_r = mxfp8_torch(norm)
        errs.append(
            _err(
                q_r.view(torch.uint8), li[1].view(torch.uint8), f"{name}: codes ", 0, 0
            )
        )
        errs.append(_err(e_r, li[2], f"{name}: e8m0 ", 0, 0))
    if aux_mean:
        # against the mean of the kernel's own R' (checked above): a one-ulp
        # difference of an R' element moves the mean by more than its own ulp when
        # the four streams cancel
        own = r_o.float().mean(1).bfloat16()
        errs.append(_err(own, out[5], f"{name}: mean ", 1.6e-2, 1e-5))
    err = max(errs)
    if err > 0:
        _FAILURES.append(name)
    return err


def _graph_replay_err(fn, inputs, fresh):
    """Capture ``fn`` (warmed eagerly on the capture stream first), then per round:
    load new inputs, run eagerly, poison the captured outputs and replay; the
    replay must equal the eager call to the bit."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = fn()
    input_ptrs = {v.data_ptr() for v in inputs.values() if torch.is_tensor(v)}
    bad = 0
    for i in range(3):
        for k, v in fresh(i).items():
            inputs[k].copy_(v)
        with torch.cuda.stream(s):
            eager = [t.clone() for t in _flat(fn())]
        torch.cuda.synchronize()
        for t in _flat(out):
            if t.data_ptr() in input_ptrs:  # the residual passed through as is
                continue
            if t.dtype in (torch.uint8, torch.float8_e4m3fn):
                t.view(torch.uint8).fill_(255)
            else:
                t.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        bad += not _same(out, eager)
    return bad


def _seam_traffic(T, mode, out_dtype, aux_mean):
    per = 4 * H * 2 + H * 2  # R read, bf16 layer input written
    if mode != "no_post":
        per += H * 2 + 4 * H * 2  # y read, R' written
    if out_dtype == "mxfp8":
        per += H + H // 32
    if aux_mean:
        per += H * 2
    return T * per + 24 * 4 * H * 4


# ---------------------------------------------------------------- tests
@benchmark()
def test_mhc_post_pre_rmsnorm(T, mode, out_dtype, aux_mean):
    from aiter.ops.flydsl import (
        flydsl_mega_mhc,
        flydsl_mhc_pack_fn,
        flydsl_mhc_post_pre_rmsnorm,
    )

    d = make_inputs(T)
    ref = seam_ref(d, mode)
    args, kw = seam_call(d, mode)
    packed = (*args[:1], flydsl_mhc_pack_fn(d["fn"]), *args[2:])
    candidates = {
        "flydsl": lambda: flydsl_mhc_post_pre_rmsnorm(
            *packed, out_dtype=out_dtype, aux_mean=aux_mean, **kw
        )
    }
    # Existing single-launch seams on the same inputs (no aux mean: the model adds a
    # separate mean of residual_out for that).
    if not aux_mean:
        candidates["flydsl_mega_mhc"] = lambda: flydsl_mega_mhc(
            *args, out_dtype=out_dtype, **kw
        )
        from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn

        hip_fn = mega_mhc_shuffle_fn(d["fn"])
        hip_args = (*args[:1], hip_fn, *args[2:])
        candidates["hip_mega_mhc"] = lambda: mega_mhc(
            *hip_args,
            w_preshuffle_bf16=True,
            out_dtype=out_dtype,
            scale_row_major=out_dtype == "mxfp8",
            **kw,
        )

    flops = 2 * T * 4 * H * 24
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        if not T:
            fn()
            ret[f"{name} err"] = 0.0
            continue
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = _seam_traffic(T, mode, out_dtype, aux_mean) / us / 1e6
        ret[f"{name} err"] = check_seam(
            name, out, ref, mode, out_dtype, aux_mean and name == "flydsl"
        )
    if T:
        fly = candidates["flydsl"]
        first = fly()
        repeat_bad = sum(not _same(first, fly()) for _ in range(3))
        if repeat_bad:
            _FAILURES.append(f"seam repeat T={T}")
        ret["flydsl repeat mismatches"] = repeat_bad

        def fresh(i):
            n = make_inputs(T, seed=100 + i)
            return {"residual": n["residual"], "sublayer_out": n["sublayer_out"]}

        bad = _graph_replay_err(fly, d, fresh)
        if bad:
            _FAILURES.append(f"seam graph T={T} {mode} {out_dtype}")
        ret["flydsl graph mismatches"] = bad
    return ret


@benchmark()
def test_mhc_post_collapse_rmsnorm(T, mode, keep_residual, aux_mean):
    from aiter.ops.flydsl import flydsl_mhc_post_collapse_rmsnorm

    d = make_inputs(T)
    post = mode == "post"
    gates = (d["sublayer_out"], d["post"], d["comb"]) if post else (None, None, None)
    ref = run_torch_tail(d["residual"], d["pre"], d["w"], NORM_EPS, *gates)

    def fn():
        return flydsl_mhc_post_collapse_rmsnorm(
            d["residual"],
            d["pre"],
            d["w"],
            NORM_EPS,
            *gates,
            keep_residual=keep_residual,
            aux_mean=aux_mean,
        )

    ret = {"gfx": get_gfx_runtime()}
    if not T:
        fn()
        return ret
    out, us = run_perftest(fn)
    nbytes = T * (4 * H * 2 + H * 2) + H * 2
    if post:
        nbytes += T * H * 2
    if keep_residual:
        nbytes += T * 4 * H * 2
    if aux_mean:
        nbytes += T * H * 2
    errs = [_err(ref[0], out[0], "tail: layer_input ", 1e-2, 2e-2)]
    if keep_residual:
        errs.append(_err(ref[1], out[1], "tail: residual_out ", 1.6e-2, 1e-5))
    elif out[1] is not None:
        errs.append(1.0)
    if aux_mean:
        # against the mean of the kernel's own R' (see check_seam)
        own_r = out[1] if keep_residual else None
        if own_r is None and post:
            own_r = flydsl_mhc_post_collapse_rmsnorm(
                d["residual"], d["pre"], d["w"], NORM_EPS, *gates, keep_residual=True
            )[1]
        if own_r is None:
            own_r = d["residual"]
        own = own_r.float().mean(1).bfloat16()
        errs.append(_err(own, out[2], "tail: mean ", 1.6e-2, 1e-5))
    elif out[2] is not None:
        errs.append(1.0)
    err = max(errs)
    repeat_bad = sum(
        not _same([t for t in out if t is not None], [t for t in fn() if t is not None])
        for _ in range(3)
    )

    def fresh(i):
        n = make_inputs(T, seed=200 + i)
        return {"residual": n["residual"], "sublayer_out": n["sublayer_out"]}

    graph_bad = _graph_replay_err(
        lambda: tuple(t for t in fn() if t is not None), d, fresh
    )
    if err or repeat_bad or graph_bad:
        _FAILURES.append(f"tail T={T} {mode} keep={keep_residual} mean={aux_mean}")
    ret.update(
        {
            "flydsl us": us,
            "flydsl TB/s": nbytes / us / 1e6,
            "flydsl err": err,
            "repeat mismatches": repeat_bad,
            "graph mismatches": graph_bad,
        }
    )
    return ret


@benchmark()
def test_mhc_seam_dispatch(T, cu_masked):
    """Each dispatch configuration (split count, waves) against the reference."""
    from aiter.ops.flydsl import flydsl_mhc_pack_fn, flydsl_mhc_post_pre_rmsnorm
    from aiter.ops.flydsl import mhc_post_rmsnorm_kernels as mk

    d = make_inputs(T, seed=1)
    ref = seam_ref(d, "post")
    args, kw = seam_call(d, "post")
    packed = (*args[:1], flydsl_mhc_pack_fn(d["fn"]), *args[2:])
    cu = mk._cu_num(d["residual"].device)
    ks, waves = mk._seam_config(T, cu, cu_masked)
    with mock.patch.object(mk, "_cu_masked", lambda: cu_masked):
        out = flydsl_mhc_post_pre_rmsnorm(*packed, out_dtype="mxfp8", **kw)
    err = check_seam(f"dispatch T={T} ks={ks}", out, ref, "post", "mxfp8", False)
    in_cus = -(-T // 16) * ks <= cu or ks == 1
    if not in_cus:
        _FAILURES.append(f"dispatch T={T}: grid exceeds the CU count")
    return {"splits": ks, "waves": waves, "err": err, "grid within CUs": in_cus}


@benchmark()
def test_mhc_seam_edges(case):
    """Zero rows (all outputs of the row must be 0, E8M0 bytes 0), a partial row
    block, an empty batch, and input validation."""
    from aiter.ops.flydsl import (
        flydsl_mhc_pack_fn,
        flydsl_mhc_post_collapse_rmsnorm,
        flydsl_mhc_post_pre_rmsnorm,
    )

    if case == "zero_rows":
        T = 20
        d = make_inputs(T, seed=3)
        for t in (d["residual"], d["sublayer_out"]):
            t[3].zero_()
            t[17].zero_()
        args, kw = seam_call(d, "post")
        packed = (*args[:1], flydsl_mhc_pack_fn(d["fn"]), *args[2:])
        out = flydsl_mhc_post_pre_rmsnorm(*packed, out_dtype="mxfp8", **kw)
        err = check_seam("edge zero rows", out, seam_ref(d, "post"), "post", "mxfp8", 0)
        norm, codes, scales = out[3]
        # values, not bytes: a negative norm weight gives -0.0 (code 0x80)
        zero_ok = all(
            bool((x[r].float() == 0).all())
            for x in (norm, codes, scales)
            for r in (3, 17)
        )
        tail = flydsl_mhc_post_collapse_rmsnorm(
            d["residual"], d["pre"], d["w"], NORM_EPS
        )[0]
        zero_ok = zero_ok and bool((tail[3] == 0).all())
    elif case == "empty":
        d = make_inputs(0)
        args, kw = seam_call(d, "post")
        packed = (*args[:1], flydsl_mhc_pack_fn(d["fn"]), *args[2:])
        out = flydsl_mhc_post_pre_rmsnorm(
            *packed, out_dtype="mxfp8", aux_mean=True, **kw
        )
        tail = flydsl_mhc_post_collapse_rmsnorm(
            d["residual"], d["pre"], d["w"], NORM_EPS, aux_mean=True
        )
        zero_ok = out[3][0].shape == (0, H) and tail[0].shape == (0, H)
        err = 0.0
    else:  # invalid inputs raise instead of launching
        d = make_inputs(16)
        bad_calls = [
            lambda: flydsl_mhc_post_collapse_rmsnorm(
                d["residual"][:, :, :4096].contiguous(), d["pre"], d["w"], NORM_EPS
            ),
            lambda: flydsl_mhc_post_collapse_rmsnorm(
                d["residual"].float(), d["pre"], d["w"], NORM_EPS
            ),
            lambda: flydsl_mhc_post_pre_rmsnorm(
                d["residual"],
                d["fn"],
                d["hc_scale"],
                d["hc_base"],
                *ARGS,
                d["pre"],
                norm_weight=d["w"],
            ),
        ]
        raised = 0
        for call in bad_calls:
            try:
                call()
            except (ValueError, TypeError):
                raised += 1
        zero_ok = raised == len(bad_calls)
        err = 0.0
    if err or not zero_ok:
        _FAILURES.append(f"edge {case}")
    return {"err": err, "ok": zero_ok}


def summarize(name, rows):
    aiter.logger.info(
        "%s (markdown):\n%s", name, pd.DataFrame(rows).to_markdown(index=False)
    )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl mhc seam/tail unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL hidden-5120 mHC seam and tail: correctness + perf",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[0, 1, 16, 20, 64, 128, 192, 256, 384, 512, 768, 1024, 4096],
    )
    parser.add_argument(
        "--mode", nargs="*", default=["post", "no_post", "identity_pre"]
    )
    parser.add_argument("-d", "--out_dtype", nargs="*", default=["bf16", "mxfp8"])
    parser.add_argument("--aux_mean", type=int, nargs="*", default=[0, 1])
    parser.add_argument("--keep_residual", type=int, nargs="*", default=[0, 1])
    parser.add_argument(
        "--dispatch_tokens",
        type=int,
        nargs="*",
        default=[1, 128, 129, 256, 257, 384, 385, 512, 513, 816, 832, 2048, 4096],
    )
    parser.add_argument("--cu_masked", type=int, nargs="*", default=[0, 1])
    args = parser.parse_args()

    summarize(
        "mhc seam",
        [
            test_mhc_post_pre_rmsnorm(T, mode, d, bool(a))
            for T, mode, d, a in itertools.product(
                args.tokens, args.mode, args.out_dtype, args.aux_mean
            )
        ],
    )
    summarize(
        "mhc tail",
        [
            test_mhc_post_collapse_rmsnorm(T, mode, bool(k), bool(a))
            for T, mode, k, a in itertools.product(
                args.tokens,
                ["post", "no_post"],
                args.keep_residual,
                args.aux_mean,
            )
            if mode == "post" or not k  # keep_residual needs the post-mix
        ],
    )
    summarize(
        "mhc seam dispatch",
        [
            test_mhc_seam_dispatch(T, bool(m))
            for T, m in itertools.product(args.dispatch_tokens, args.cu_masked)
        ],
    )
    summarize(
        "mhc seam edges",
        [test_mhc_seam_edges(c) for c in ("zero_rows", "empty", "invalid")],
    )
    if _FAILURES:
        raise AssertionError(f"{len(_FAILURES)} mhc checks failed: {_FAILURES[:8]}")


if __name__ == "__main__":
    main()
