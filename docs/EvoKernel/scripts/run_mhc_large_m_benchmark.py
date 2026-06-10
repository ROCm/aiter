#!/usr/bin/env python3
"""Large-M (M>=1024) mhc_fused_post_pre benchmark for mhc_large_m branch."""
from __future__ import annotations

import argparse
import gc
import sys
from typing import Optional

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest

try:
    from aiter.ops.triton.fusions.mhc import mhc_post_pre as triton_mhc_post_pre

    _HAS_TRITON = True
except ImportError:
    triton_mhc_post_pre = None
    _HAS_TRITON = False

TRITON_MAX_M = 4096
LARGE_M_MIN = 1024
torch.set_default_device("cuda")


def mhc_pre_ref(
    residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
    hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=1e-6,
):
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    out = residual_flat @ fn.T
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    hc_scale = torch.cat([
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult * hc_mult),
    ])
    mixes = mixes * hc_scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    def sinkhorn(x, repeat, eps):
        x = x.softmax(-1) + eps
        x = x / (x.sum(-2, keepdim=True) + eps)
        for _ in range(repeat - 1):
            x = x / (x.sum(-1, keepdim=True) + eps)
            x = x / (x.sum(-2, keepdim=True) + eps)
        return x

    res_mix = sinkhorn(res_mix, sinkhorn_repeat, hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2)
    if norm_weight is not None:
        x = layer_input
        rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = (x.float() * rms * norm_weight.float()).bfloat16()
    return post_mix, res_mix, layer_input.bfloat16()


def mhc_post_ref(x, residual, post_layer_mix, comb_res_mix):
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_post_pre_ref(layer_input, residual_in, post_layer_mix, comb_res_mix, fn,
                     hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                     hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=1e-6):
    next_residual = mhc_post_ref(layer_input, residual_in, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input_out = mhc_pre_ref(
        next_residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
        hc_post_mult_value, sinkhorn_repeat, norm_weight, norm_eps,
    )
    return post_mix, comb_mix, layer_input_out, next_residual


def mhc_post_pre_unfused_hip(layer_input, residual_in, post_layer_mix, comb_res_mix,
                              fn, hc_scale, hc_base, **extra):
    next_residual = torch.empty_like(residual_in)
    aiter.mhc_post(next_residual, layer_input, residual_in, post_layer_mix, comb_res_mix)
    post_mix, comb_mix, layer_input_out = aiter.mhc_pre(
        next_residual, fn, hc_scale, hc_base, **extra,
    )
    return post_mix, comb_mix, layer_input_out, next_residual


@benchmark()
def test_mhc_post_pre(m, hidden_size, hc_mult, fuse_rmsnorm=False):
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size
    layer_input = torch.randn(m, hidden_size, dtype=dtypes.bf16)
    residual_in = torch.randn(m, hc_mult, hidden_size, dtype=dtypes.bf16)
    post_layer_mix = torch.randn(m, hc_mult, 1, dtype=dtypes.fp32)
    comb_res_mix = torch.randn(m, hc_mult, hc_mult, dtype=dtypes.fp32)
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=dtypes.fp32)
    hc_scale = torch.randn((3,), dtype=dtypes.fp32) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=dtypes.fp32) * 0.1
    norm_weight = torch.randn(hidden_size, dtype=dtypes.bf16) if fuse_rmsnorm else None
    extra = {
        "rms_eps": 1e-6, "hc_pre_eps": 1e-6, "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0, "sinkhorn_repeat": 20,
    }
    if fuse_rmsnorm:
        extra["norm_eps"] = 1e-6
    refs = mhc_post_pre_ref(
        layer_input, residual_in, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base,
        extra["rms_eps"], extra["hc_pre_eps"], extra["hc_sinkhorn_eps"],
        extra["hc_post_mult_value"], extra["sinkhorn_repeat"], norm_weight,
        extra.get("norm_eps", 1e-6),
    )
    hip_kw = {**extra}
    if fuse_rmsnorm:
        hip_kw["norm_weight"] = norm_weight
    unfused_out, unfused_us = run_perftest(
        mhc_post_pre_unfused_hip, layer_input, residual_in, post_layer_mix, comb_res_mix,
        fn, hc_scale, hc_base, num_rotate_args=0, **hip_kw,
    )
    fused_out, fused_us = run_perftest(
        aiter.mhc_fused_post_pre, layer_input, residual_in, post_layer_mix, comb_res_mix,
        fn, hc_scale, hc_base, force_fused=True, num_rotate_args=0, **hip_kw,
    )
    hip_unfused_err = checkAllclose(refs[2], unfused_out[2], msg="unfused/layer_input")
    hip_fused_err = checkAllclose(refs[2], fused_out[2], msg="fused/layer_input")
    ret = {
        "m": m, "hidden_size": hidden_size, "hc_mult": hc_mult,
        "fuse_rmsnorm": fuse_rmsnorm,
        "unfused_us": unfused_us, "hip_unfused_err": hip_unfused_err,
        "fused_us": fused_us, "hip_fused_err": hip_fused_err,
    }
    if _HAS_TRITON and not fuse_rmsnorm and m <= TRITON_MAX_M:
        phi = fn.T.contiguous()
        triton_out, triton_us = run_perftest(
            triton_mhc_post_pre, layer_input, residual_in, post_layer_mix.squeeze(-1),
            comb_res_mix, phi, hc_scale, hc_base, hc_mult,
            eps=extra["rms_eps"], hc_pre_eps=extra["hc_pre_eps"],
            hc_post_mult_value=extra["hc_post_mult_value"], sinkhorn_iters=extra["sinkhorn_repeat"],
            asymmetric_exp_domain=True, hc_sinkhorn_eps=extra["hc_sinkhorn_eps"],
            num_rotate_args=0,
        )
        ret["triton_us"] = triton_us
        ret["triton_fused_err"] = checkAllclose(
            refs[2], triton_out[2].to(refs[2].dtype), msg="triton/layer_input", rtol=2e-2,
        )
    else:
        ret["triton_us"] = float("nan")
    return ret


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", type=int, nargs="*", default=[1024, 2048, 8192, 65536])
    p.add_argument("-n", "--hidden_size", type=int, nargs="*", default=[4096, 7168])
    p.add_argument("-o", "--output", default="")
    args = p.parse_args()
    ms = [x for x in args.m if x >= LARGE_M_MIN]
    print(f"# torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"# mhc_large_m gfx950 hybrid (force_fused=True, M>={LARGE_M_MIN})", flush=True)
    rows = []
    for hs in args.hidden_size:
        for m in ms:
            torch.cuda.empty_cache()
            gc.collect()
            try:
                ret = test_mhc_post_pre(m=m, hidden_size=hs, hc_mult=4)
            except torch.OutOfMemoryError as e:
                print(f"# OOM M={m} C={hs}: {e}", flush=True)
                continue
            rows.append(ret)
            print(
                f"# done M={m} C={hs}: unfused={ret['unfused_us']:.4f} "
                f"fused={ret['fused_us']:.4f} triton={ret.get('triton_us', float('nan'))}",
                flush=True,
            )
            torch.cuda.empty_cache()
            gc.collect()
    df = pd.DataFrame(rows)
    try:
        md = df.to_markdown(index=False)
    except ImportError:
        md = df.to_string(index=False)
    print("\n[aiter] mhc_large_m summary (markdown):\n")
    print(md)
    if args.output:
        with open(args.output, "w") as f:
            f.write(md + "\n")
        print(f"# saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
