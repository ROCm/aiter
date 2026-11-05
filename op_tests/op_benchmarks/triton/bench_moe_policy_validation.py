# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import os
import statistics
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import (
    _bf16_g1u1_gfx1201_triton_policy_contains_m,
    fused_moe,
    torch_moe,
)

try:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
except ImportError:  # pragma: no cover - vLLM is present in validation images.
    MoEActivation = None
    fused_experts = None


_DISABLE_ENV = "AITER_DISABLE_GFX1201_BF16_G1U1_TRITON"


@contextmanager
def _triton_policy_enabled(enabled: bool):
    old = os.environ.get(_DISABLE_ENV)
    os.environ[_DISABLE_ENV] = "0" if enabled else "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(_DISABLE_ENV, None)
        else:
            os.environ[_DISABLE_ENV] = old


def _make_case(tokens: int):
    torch.manual_seed(2026 + tokens)
    model_dim = 2048
    inter_dim = 128
    experts = 256
    topk = 8
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (experts, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device="cuda"
    ).contiguous()
    w2 = torch.randn(
        (experts, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda"
    ).contiguous()
    topk_ids = torch.randint(
        0, experts, (tokens, topk), dtype=torch.int32, device="cuda"
    )
    topk_weight = torch.rand((tokens, topk), dtype=torch.float32, device="cuda")
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return hidden, w1, w2, topk_weight.contiguous(), topk_ids.contiguous()


def _aiter_call(hidden, w1, w2, topk_weight, topk_ids):
    return fused_moe(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
    )


def _vllm_call(hidden, w1, w2, topk_weight, topk_ids):
    if fused_experts is None or MoEActivation is None:
        raise RuntimeError("vLLM is not importable in this environment")
    return fused_experts(
        hidden,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=w1.shape[0],
    )


def _bench(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.mean(times), statistics.median(times)


def _status_bench(fn, warmup: int, iters: int):
    try:
        mean_us, median_us = _bench(fn, warmup, iters)
    except Exception as error:  # noqa: BLE001 benchmark reports path failures.
        return type(error).__name__, "", "", str(error).splitlines()[0]
    return "ok", f"{mean_us:.3f}", f"{median_us:.3f}", ""


def _reference_status(actual, hidden, w1, w2, topk_weight, topk_ids, atol, rtol):
    try:
        expected = torch_moe(
            hidden,
            w1,
            w2,
            topk_weight,
            topk_ids,
            activation=aiter.ActivationType.Silu,
        )
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    except Exception as error:  # noqa: BLE001 validation report keeps going.
        return type(error).__name__, str(error).splitlines()[0]
    return "ok", ""


def _parse_m_values(args):
    if args.m_list:
        return [int(value) for value in args.m_list.split(",") if value]
    return list(range(args.m_min, args.m_max + 1))


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--m-min", type=int, default=1024)
    parser.add_argument("--m-max", type=int, default=2560)
    parser.add_argument("--m-list", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--check-reference", action="store_true")
    parser.add_argument("--check-vllm-reference", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--probe-original-once", action="store_true")
    parser.add_argument("--atol", type=float, default=128.0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "M",
        "policy_selected",
        "new_status",
        "new_mean_us",
        "new_median_us",
        "new_reference_status",
        "vllm_status",
        "vllm_mean_us",
        "vllm_median_us",
        "vllm_reference_status",
        "original_status",
        "original_mean_us",
        "original_median_us",
        "delta_new_vs_vllm_pct",
        "delta_new_vs_original_pct",
        "error",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        print(",".join(fields), flush=True)
        original_probed = False
        for tokens in _parse_m_values(args):
            hidden, w1, w2, topk_weight, topk_ids = _make_case(tokens)
            policy_selected = _bf16_g1u1_gfx1201_triton_policy_contains_m(tokens)
            aiter_fn = partial(_aiter_call, hidden, w1, w2, topk_weight, topk_ids)
            vllm_fn = partial(_vllm_call, hidden, w1, w2, topk_weight, topk_ids)
            with _triton_policy_enabled(True):
                new_status, new_mean, new_median, new_error = _status_bench(
                    aiter_fn,
                    args.warmup,
                    args.iters,
                )
                new_out = (
                    aiter_fn() if args.check_reference and new_status == "ok" else None
                )
            vllm_status, vllm_mean, vllm_median, vllm_error = _status_bench(
                vllm_fn,
                args.warmup,
                args.iters,
            )
            if args.skip_original or (args.probe_original_once and original_probed):
                orig_status, orig_mean, orig_median, orig_error = "skipped", "", "", ""
            else:
                with _triton_policy_enabled(False):
                    orig_status, orig_mean, orig_median, orig_error = _status_bench(
                        aiter_fn,
                        args.warmup,
                        args.iters,
                    )
                original_probed = True
            new_ref_status = ""
            new_ref_error = ""
            vllm_ref_status = ""
            vllm_ref_error = ""
            if args.check_reference:
                if new_out is not None:
                    new_ref_status, new_ref_error = _reference_status(
                        new_out,
                        hidden,
                        w1,
                        w2,
                        topk_weight,
                        topk_ids,
                        args.atol,
                        args.rtol,
                    )
                if args.check_vllm_reference and vllm_status == "ok":
                    vllm_ref_status, vllm_ref_error = _reference_status(
                        _vllm_call(hidden, w1, w2, topk_weight, topk_ids),
                        hidden,
                        w1,
                        w2,
                        topk_weight,
                        topk_ids,
                        args.atol,
                        args.rtol,
                    )
            delta_vllm = ""
            if new_status == "ok" and vllm_status == "ok":
                delta = (float(vllm_mean) - float(new_mean)) / float(vllm_mean)
                delta_vllm = f"{delta * 100.0:+.2f}%"
            delta_orig = ""
            if new_status == "ok" and orig_status == "ok":
                delta = (float(orig_mean) - float(new_mean)) / float(orig_mean)
                delta_orig = f"{delta * 100.0:+.2f}%"
            error = " | ".join(
                item
                for item in [
                    new_error,
                    vllm_error,
                    orig_error,
                    new_ref_error,
                    vllm_ref_error,
                ]
                if item
            )
            row = {
                "M": tokens,
                "policy_selected": int(policy_selected),
                "new_status": new_status,
                "new_mean_us": new_mean,
                "new_median_us": new_median,
                "new_reference_status": new_ref_status,
                "vllm_status": vllm_status,
                "vllm_mean_us": vllm_mean,
                "vllm_median_us": vllm_median,
                "vllm_reference_status": vllm_ref_status,
                "original_status": orig_status,
                "original_mean_us": orig_mean,
                "original_median_us": orig_median,
                "delta_new_vs_vllm_pct": delta_vllm,
                "delta_new_vs_original_pct": delta_orig,
                "error": error,
            }
            writer.writerow(row)
            print(",".join(str(row[key]) for key in fields), flush=True)


if __name__ == "__main__":
    main()
