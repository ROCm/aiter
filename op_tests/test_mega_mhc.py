# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mega-MHC correctness checks and HIP / FlyDSL / Triton performance table.

Each selected shape runs reference checks before timing.
Use --check-only to check those shapes and backends without benchmarking.
"""

import argparse
import itertools
import math
from functools import cache

import pandas as pd
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.mega_mhc import _counter_state_ok, mega_mhc, prepare_mega_mhc
from aiter.test_common import benchmark, run_perftest

# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)
MODES = ("post", "no_post", "identity_pre", "no_post_identity")
BACKENDS = ("hip", "flydsl", "triton")
OUT_DTYPES = ("bf16", "mxfp8")
# M covers decode through prefill; use --hidden to test other model widths.
TOKEN_COUNTS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
HIDDEN_SIZES = (5120,)


def ref_mhc_fused_post_pre_delayed_rmsnorm(
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
    """FP32 post-mix, delayed pre-mix and RMSNorm, with BF16 seam rounding."""
    t, n, _ = residual.shape
    if sublayer_out is not None:
        mixed = torch.einsum(
            "tij,tih->tjh", comb_res_mix.float().view(t, n, n), residual.float()
        )
        r = (
            mixed
            + post_layer_mix.float().view(t, n, 1) * sublayer_out.float().unsqueeze(1)
        ).to(residual.dtype)
    else:
        r = residual
    x = r.flatten(1).float()
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
    li = (pre_mix.float().view(t, n, 1) * r.float()).sum(dim=1).to(residual.dtype)
    lf = li.float()
    li = (
        lf
        * torch.rsqrt(lf.square().mean(-1, keepdim=True) + norm_eps)
        * norm_weight.float()
    ).to(residual.dtype)
    return r, post.unsqueeze(-1), comb, li, pre


def make_inputs(t, h, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    residual = (torch.randn(t, 4, h, device=device, generator=g) * 1.5).to(
        torch.bfloat16
    )
    y = torch.randn(t, h, device=device, generator=g).to(torch.bfloat16)
    fn = torch.randn(24, 4 * h, device=device, generator=g) * 0.01
    scale = torch.tensor([0.7, 1.3, 0.9], device=device)
    base = torch.randn(24, device=device, generator=g) * 0.5
    post = torch.rand(t, 4, 1, device=device, generator=g) * 2
    comb = torch.softmax(torch.randn(t, 4, 4, device=device, generator=g), -1)
    pre = torch.sigmoid(torch.randn(t, 4, device=device, generator=g)) + 1e-6
    weight = (1 + 0.2 * torch.randn(h, device=device, generator=g)).to(torch.bfloat16)
    return residual, y, fn, scale, base, post, comb, pre, weight


def reference_for(args, kw):
    reference_kw = dict(kw)
    if kw["pre_mix"] is None:
        identity = torch.zeros(
            (args[0].shape[0], 4), device=args[0].device, dtype=torch.float32
        )
        identity[:, 0] = 1
        reference_kw["pre_mix"] = identity
    return ref_mhc_fused_post_pre_delayed_rmsnorm(*args, **reference_kw)


def case(t, h, mode, seed=0):
    r, y, fn, scale, base, post, comb, pre, w = make_inputs(t, h, seed=seed)
    args = (r, fn, scale, base, *ARGS)
    kw = {"pre_mix": pre, "norm_weight": w}
    if "no_post" not in mode:
        kw.update(sublayer_out=y, post_layer_mix=post, comb_res_mix=comb)
    if "identity" in mode:
        kw["pre_mix"] = None
    return args, kw, reference_for(args, kw)


def ref_mxfp8(norm):
    """Group32 E4M3 with ceil-power-of-two scaling on the rounded BF16 norm."""
    t, h = norm.shape
    amax = norm.float().view(t, h // 32, 32).abs().amax(-1).clamp_min(1e-10)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-126, 127)
    scale = torch.exp2(exponent).repeat_interleave(32, dim=-1)
    q = (norm.float() / scale).to(torch.float8_e4m3fn)
    return q, (exponent + 127).to(torch.uint8)


def check(out, reference):
    torch.testing.assert_close(out[0], reference[0], rtol=1.6e-2, atol=1e-5)
    norm = out[3]
    if isinstance(norm, tuple):
        norm, q, scale = norm
        assert norm.dtype == torch.bfloat16
        assert q.dtype == torch.float8_e4m3fn and q.shape == norm.shape
        assert scale.dtype == torch.uint8 and scale.shape == (
            norm.shape[0],
            norm.shape[1] // 32,
        )
        assert all(t.is_contiguous() and t.device == norm.device for t in (q, scale))
        # HIP and FlyDSL can round the MHC norm differently within its existing
        # tolerance. Quantization itself must be exact on each backend's norm.
        ref_q, ref_scale = ref_mxfp8(norm)
        torch.testing.assert_close(
            q.view(torch.uint8), ref_q.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(norm, reference[3], rtol=1e-2, atol=2e-2)
    for i in (1, 2, 4):
        torch.testing.assert_close(out[i], reference[i], rtol=1e-3, atol=5e-4)


@cache
def backend_function(name):
    if name == "hip":
        return mega_mhc
    if name == "flydsl":
        from aiter.ops.flydsl.mega_mhc_kernels import flydsl_mega_mhc

        return flydsl_mega_mhc
    if name == "triton":
        from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
            mhc_fused_post_pre_delayed_rmsnorm,
        )

        return mhc_fused_post_pre_delayed_rmsnorm
    raise ValueError(f"unknown backend: {name}")


def benchmark_bytes(m, hidden_size, mode, out_dtype="bf16"):
    """Dominant tensor traffic, matching the FlyDSL Mega-MHC benchmark.

    Count BF16 residual input and norm output, plus sublayer input and residual
    output with post-mix. Read the FP32 projection weight once (equivalently its
    BF16 hi/lo pack). Exclude scratch, repeated loads and small gate/norm tensors;
    this is a common effective-bandwidth model, not measured HBM traffic.
    MXFP8 adds one output byte per element and one exponent byte per group32.
    """
    if not m:
        return 0
    per_token = hidden_size * (10 if "no_post" in mode else 20)
    if out_dtype == "mxfp8":
        per_token += hidden_size + hidden_size // 32
    fn_bytes = 24 * 4 * hidden_size * 4
    return m * per_token + fn_bytes


class _BenchmarkBackend:
    """Prepare each rotated weight/workspace before run_perftest starts timing."""

    def __init__(self, name, tokens, scalar_args, out_dtype="bf16"):
        self.name = name
        self.backend = backend_function(name)
        self.tokens = tokens
        self.scalar_args = scalar_args
        self.out_dtype = out_dtype
        if name == "triton" and out_dtype != "bf16":
            raise ValueError("Triton Mega-MHC has no fused mxfp8 output")
        self.workspaces = {}
        self.prepared_addresses = set()
        self.used_addresses = set()

    def __call__(
        self, r, fn, scale, base, pre, sublayer, post, comb, weight, residual_out
    ):
        self.used_addresses.add(r.data_ptr())
        kw = {
            "pre_mix": pre,
            "sublayer_out": sublayer,
            "post_layer_mix": post,
            "comb_res_mix": comb,
            "norm_weight": weight,
            "residual_out": residual_out,
        }
        if self.name == "hip":
            kw["workspace"] = self.workspaces[id(fn)]
        if self.name != "triton":
            kw["out_dtype"] = self.out_dtype
        return self.backend(r, fn, scale, base, *self.scalar_args, **kw)

    def prepare(self, *rotated):
        # Called before profiling for the original tensors and every copy.
        # HIP owns a workspace per fn; FlyDSL warms its per-weight cache here.
        self.prepared_addresses.add(rotated[0].data_ptr())
        if self.name == "hip":
            self.workspaces[id(rotated[1])] = prepare_mega_mhc(rotated[1], self.tokens)
        self(*rotated)
        self.used_addresses.discard(rotated[0].data_ptr())


@benchmark()
def benchmark_mega_mhc(
    m,
    hidden_size,
    mode,
    backends=BACKENDS,
    seed=0,
    num_iters=101,
    out_dtype="bf16",
    check_only=False,
):
    """Check one M/H/mode/dtype combination and return its backend results."""
    args, kw, reference = case(m, hidden_size, mode, seed)
    result = {"status": "PASS"}
    num_bytes = benchmark_bytes(m, hidden_size, mode, out_dtype)
    # Pass actual tensors to run_perftest so its default memory-sizing and
    # deepcopy rotation can see all inputs and the residual output buffer.
    tensors = (
        *args[:4],
        kw["pre_mix"],
        kw.get("sublayer_out"),
        kw.get("post_layer_mix"),
        kw.get("comb_res_mix"),
        kw["norm_weight"],
        torch.empty_like(args[0]) if "sublayer_out" in kw else None,
    )
    for name in backends:
        if name == "triton" and m:
            from aiter.ops.triton.utils.mhc_config_utils import (
                get_mhc_fused_post_pre_delayed_rmsnorm_config,
            )

            config = get_mhc_fused_post_pre_delayed_rmsnorm_config(m)
            if hidden_size % (config["NUM_KSPLIT"] * config["TILE_K"]):
                # The Triton k-loop has no column mask. Its default launch
                # config is tuned for H=5120 and cannot cover every width.
                result[name] = "SKIP"
                result[f"{name}_us"] = result[f"{name}_TB/s"] = float("nan")
                continue
        caller = _BenchmarkBackend(name, m, args[4:], out_dtype)
        caller.prepare(*tensors)
        out = caller(*tensors)
        check(out, reference)
        result[name] = "PASS"
        if not check_only:
            us = float("nan")  # Empty tensors have no kernel to time.
            if m:
                out, us = run_perftest(
                    caller,
                    *tensors,
                    num_iters=num_iters,
                    prepare_func=caller.prepare,
                )
                if not math.isfinite(us) or us <= 0:
                    raise RuntimeError(f"{name} returned an invalid GPU timing: {us}")
                check(out, reference)
                assert caller.used_addresses == caller.prepared_addresses
            result[f"{name}_us"] = us
            result[f"{name}_TB/s"] = num_bytes / us / 1e6
        for workspace in caller.workspaces.values():
            assert _counter_state_ok(workspace)
        caller.workspaces.clear()
    result["status"] = (
        "PASS" if any(result[name] == "PASS" for name in backends) else "SKIP"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mega-MHC reference checks and HIP / FlyDSL / Triton timings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=TOKEN_COUNTS,
        help="M: number of tokens (decode to prefill)",
    )
    parser.add_argument(
        "-n",
        "--hidden",
        "--hidden-size",
        "--hidden_size",
        type=int,
        nargs="+",
        default=HIDDEN_SIZES,
        help="H: hidden size per stream; four streams per token",
    )
    parser.add_argument(
        "--mode",
        "--modes",
        choices=MODES,
        nargs="+",
        default=("post", "no_post"),
        help="post: include post-mix; no_post: start from residual; "
        "identity variants select stream 0 instead of learned pre-mix",
    )
    parser.add_argument(
        "--out-dtype",
        choices=OUT_DTYPES,
        nargs="+",
        default=("bf16",),
        help="BF16 norm, or BF16 norm plus fused MXFP8 codes and scales",
    )
    parser.add_argument(
        "--backends",
        choices=BACKENDS,
        nargs="+",
        help="default: HIP/FlyDSL/Triton on gfx950; HIP/Triton on gfx942; "
        "Triton is omitted for MXFP8",
    )
    parser.add_argument("--seed", type=int, default=0, help="shared input seed")
    parser.add_argument(
        "--iters",
        type=int,
        default=101,
        help="timed calls with automatic buffer rotation",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="check the selected shapes and backends, without timing",
    )
    args = parser.parse_args()
    if any(m < 0 for m in args.tokens):
        parser.error("tokens must be nonnegative")
    if any(not 0 < h <= 65536 or h % 32 for h in args.hidden):
        parser.error("hidden size must be a multiple of 32 in [32, 65536]")
    if not args.check_only and args.iters < 2:
        parser.error("iters must be at least 2 (the first sample is warmup)")
    if not torch.cuda.is_available():
        parser.error("Mega-MHC requires a GPU")
    arch = get_gfx_runtime()
    if arch not in ("gfx950", "gfx942"):
        parser.error("Mega-MHC requires gfx950 or gfx942")
    if "mxfp8" in args.out_dtype and arch != "gfx950":
        parser.error("Mega-MHC mxfp8 requires gfx950")
    backends = tuple(
        dict.fromkeys(
            args.backends or (BACKENDS if arch == "gfx950" else ("hip", "triton"))
        )
    )
    selected = {
        dtype: tuple(name for name in backends if dtype == "bf16" or name != "triton")
        for dtype in args.out_dtype
    }
    if any(not names for names in selected.values()):
        parser.error("Triton has no fused MXFP8 output; select HIP or FlyDSL")
    backends = tuple(
        name for name in backends if any(name in names for names in selected.values())
    )
    if "flydsl" in backends and arch != "gfx950":
        parser.error("FlyDSL Mega-MHC requires gfx950")
    for name in backends:
        try:
            backend_function(name)
        except ImportError as error:
            parser.error(f"{name} is unavailable: {error}; select installed --backends")
    print(f"Mega-MHC | {arch} | {torch.cuda.get_device_name()}", flush=True)
    if "triton" in backends:
        print(
            "Triton: skip MXFP8 and H not divisible by its default split_k * tile_k; "
            "skipped results show SKIP or '-'.",
            flush=True,
        )
    if not args.check_only:
        print(
            f"GPU time: {args.iters} calls with automatic buffer rotation. "
            "TB/s is effective tensor bandwidth; excludes scratch and repeated loads.",
            flush=True,
        )
    rows = []
    for dtype, h, mode, m in itertools.product(
        selected,
        dict.fromkeys(args.hidden),
        dict.fromkeys(args.mode),
        dict.fromkeys(args.tokens),
    ):
        rows.append(
            benchmark_mega_mhc(
                m,
                h,
                mode,
                selected[dtype],
                args.seed,
                args.iters,
                dtype,
                args.check_only,
            )
        )
    columns = ["m", "hidden_size", "mode", "out_dtype"]
    if args.check_only:
        columns += list(backends)
    else:
        columns += [f"{name}_{unit}" for name in backends for unit in ("us", "TB/s")]
        columns += ["status"]
    title = "correctness" if args.check_only else "performance (us, TB/s)"
    print(f"\nMega-MHC {title}:", flush=True)
    print(
        pd.DataFrame(rows)[columns].to_string(
            index=False, float_format="{:.3f}".format, na_rep="-"
        )
    )
    passed = sum(row["status"] == "PASS" for row in rows)
    count = sum(row.get(name) == "PASS" for row in rows for name in backends)
    skipped = len(rows) * len(backends) - count
    print(
        f"\nPassed: {passed}/{len(rows)} cases, {count} backend checks; "
        f"{skipped} backend comparisons skipped.",
        flush=True,
    )


if __name__ == "__main__":
    main()
