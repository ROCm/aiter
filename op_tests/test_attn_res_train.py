# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.attn_res_train import SUPPORTED_GFX, flydsl_attn_res_train
from aiter.ops.triton.fusions.attn_res import attn_res_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

EPS = 1e-5  # Kimi-K3 rms_norm_eps
INPUTS = ("query", "bank", "prefix", "rms_weight", "output_rms_weight")


def run_torch(inputs, eps, scale, dy):
    """fp32 AttnRes site; returns y and the gradient of every input."""
    leaves = {
        name: inputs[name].detach().float().requires_grad_()
        for name in INPUTS
        if inputs[name] is not None
    }
    candidates = [leaves["prefix"].unsqueeze(1)]
    if "bank" in leaves:
        candidates.insert(0, leaves["bank"])
    v = torch.cat(candidates, dim=1)  # [N, L, H]
    r = torch.rsqrt(v.pow(2).mean(-1) + eps)
    qw = leaves["query"].reshape(-1) * leaves["rms_weight"].reshape(-1)
    p = torch.softmax(r * (v * qw).sum(-1) * scale, dim=-1)
    y = (p.unsqueeze(-1) * v).sum(1)
    if "output_rms_weight" in leaves:
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
        y = y * leaves["output_rms_weight"].reshape(-1)
    y.backward(dy.float())
    return y.detach(), {name: leaf.grad for name, leaf in leaves.items()}


def make_inputs(n_tokens, hidden, num_bank, output_norm, dtype, weight_dtype):
    # Candidates at different magnitudes, so the RMSNorm inside the score matters
    # and the softmax is neither uniform nor one-hot.
    mags = torch.rand(num_bank + 1) * 3 + 0.5
    bank = (
        (torch.randn(n_tokens, num_bank, hidden) * mags[:num_bank, None]).to(dtype)
        if num_bank
        else None
    )
    return {
        "query": (torch.randn(1, hidden) * 0.05).to(weight_dtype),
        "bank": bank,
        "prefix": (torch.randn(n_tokens, hidden) * mags[-1]).to(dtype),
        "rms_weight": (1 + 0.1 * torch.randn(hidden)).to(weight_dtype),
        "output_rms_weight": (
            (1 + 0.1 * torch.randn(hidden)).to(weight_dtype) if output_norm else None
        ),
    }


@benchmark()
def test_attn_res_train(n_tokens, hidden, num_bank, output_norm, weight_dtype):
    torch.manual_seed(0)
    dtype = dtypes.bf16
    inputs = make_inputs(
        n_tokens, hidden, num_bank, output_norm, dtype, dtypes.d_dtypes[weight_dtype]
    )
    dy = torch.randn(n_tokens, hidden, dtype=dtype)
    y_ref, grad_ref = run_torch(inputs, EPS, 1.0, dy)

    leaves = {
        name: t.detach().clone().requires_grad_()
        for name, t in inputs.items()
        if t is not None
    }
    args = [leaves.get(name) for name in INPUTS]

    def fwd():
        return flydsl_attn_res_train(*args, eps=EPS)

    num_src = num_bank + 1
    row_bytes = n_tokens * hidden * dtype.itemsize
    ret = {"gfx": get_gfx_runtime()}

    y, us = run_perftest(fwd)
    err = checkAllclose(
        y_ref, y.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="flydsl fwd: y"
    )
    assert err == 0, f"flydsl fwd: {err:.2%} of y outside rtol/atol"
    ret["flydsl fwd us"] = us
    ret["flydsl fwd TFLOPS"] = 6 * n_tokens * num_src * hidden / us / 1e6
    ret["flydsl fwd TB/s"] = (num_src + 1) * row_bytes / us / 1e6
    ret["flydsl fwd err"] = err

    y = fwd()
    grad_args = list(leaves.values())
    grads, us = run_perftest(
        lambda: torch.autograd.grad(y, grad_args, dy, retain_graph=True)
    )
    errs = []
    for name, grad in zip(leaves, grads):
        ref = grad_ref[name]
        # Weight gradients sum over tokens; compare them relative to their scale.
        atol = 2e-2 * ref.abs().max().item() if name not in ("bank", "prefix") else 2e-2
        errs.append(
            checkAllclose(
                ref,
                grad.to(dtypes.fp32),
                rtol=2e-2,
                atol=atol,
                msg=f"flydsl bwd: d{name}",
            )
        )
        assert errs[-1] == 0, f"flydsl bwd: {errs[-1]:.2%} of d{name} outside rtol/atol"
    ret["flydsl bwd us"] = us
    ret["flydsl bwd TFLOPS"] = 12 * n_tokens * num_src * hidden / us / 1e6
    # dy, the candidates read once (not at all when dv is dy), one gradient row each.
    reads = num_src if num_bank or output_norm else 0
    ret["flydsl bwd TB/s"] = (1 + reads + num_src) * row_bytes / us / 1e6
    ret["flydsl bwd err"] = max(errs)

    if weight_dtype != "bf16":
        return ret
    # main's Triton forward on the same call, packed outside the timed region.
    packed = torch.cat(
        ([inputs["bank"]] if num_bank else []) + [inputs["prefix"].unsqueeze(1)], dim=1
    )
    y_triton, us = run_perftest(
        attn_res_fwd,
        inputs["query"],
        packed,
        inputs["rms_weight"],
        inputs["output_rms_weight"],
        EPS,
        1.0,
        layout="packed",
    )
    err = checkAllclose(
        y_ref, y_triton.to(dtypes.fp32), rtol=1e-2, atol=1e-2, msg="triton fwd: y"
    )
    assert err == 0, f"triton fwd: {err:.2%} of y outside rtol/atol"
    ret["triton fwd us"] = us
    ret["triton fwd TFLOPS"] = 6 * n_tokens * num_src * hidden / us / 1e6
    ret["triton fwd TB/s"] = (num_src + 1) * row_bytes / us / 1e6
    ret["triton fwd err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "attn_res_train: unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL AttnRes training forward and backward",
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[1, 37, 256, 2048, 8192]
    )
    parser.add_argument("--hidden", type=int, nargs="*", default=[7168, 512])
    parser.add_argument(
        "-k",
        "--num_bank",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 0],
        help="banked candidates; the prefix makes L = K + 1 (Kimi-K3: K = 1..8; "
        "16 covers reductions of more than 32 values)",
    )
    parser.add_argument(
        "--output_norm",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
    )
    parser.add_argument(
        "--weight_dtype",
        choices=["bf16", "fp32"],
        nargs="*",
        default=["bf16", "fp32"],
        help="dtype of query, rms_weight and output_rms_weight",
    )
    args = parser.parse_args()

    df = [
        test_attn_res_train(n, h, k, o, w)
        for n, h, k, o, w in itertools.product(
            args.tokens, args.hidden, args.num_bank, args.output_norm, args.weight_dtype
        )
    ]
    df = pd.DataFrame(df)
    aiter.logger.info("attn_res_train summary:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
