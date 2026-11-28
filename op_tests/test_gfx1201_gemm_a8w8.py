# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import sys

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gfx1201_gemm_a8w8 import (
    gemm_a8w8_bpreshuffle_gfx1201,
    shuffle_weight_gfx1201,
)


def unshuffle_weight_gfx1201(wq):
    """Inverse of shuffle_weight_gfx1201, written from the documented layout."""
    n, k = wq.shape
    src = wq.view(torch.uint8).reshape(n // 32, k // 64, 32, 2, 4, 8)
    out = torch.empty(n, k, dtype=torch.uint8, device=wq.device)
    for lane in range(32):
        h, hb, e = lane // 16, (lane // 8) % 2, lane % 8
        for t in range(2):
            col = 16 * hb + 8 * t + e
            for s in range(4):
                kb = 32 * h + 8 * s
                dst = out.view(n // 32, 32, k // 64, 64)[:, col, :, kb : kb + 8]
                dst.copy_(src[:, :, lane, t, s, :])
    return out.view(wq.dtype)


def quantize(x, per_row):
    amax = x.abs().amax(dim=1, keepdim=True) if per_row else x.abs().amax()
    scale = (amax.float() / 448.0).clamp_min(1e-12)
    q = (x.float() / scale).clamp(-448, 448).to(dtypes.fp8)
    return q, scale.reshape(-1, 1) if per_row else scale.reshape(1)


def reference(xq, w, x_scale, w_scale):
    """fp64 result and the fp64 sum of |products| (bounds fp32 accumulation)."""
    scale = x_scale.double().reshape(-1, 1) * w_scale.double().reshape(1, -1)
    acc = xq.double() @ w.double().t()
    mag = xq.double().abs() @ w.double().abs().t()
    return acc * scale, mag * scale


def check(name, got, ref):
    want, mag = ref
    want_bf16 = want.to(torch.bfloat16)
    # One bf16 ulp of the correctly rounded result plus an fp32 accumulation
    # error term, so a different summation order may flip the last bit.
    ulp = want_bf16.double().abs() * 2.0**-7 + mag * 2.0**-20
    diff = (got.double() - want_bf16.double()).abs()
    exact = (got == want_bf16).float().mean().item()
    # Round-to-nearest-even: all but rare accumulation-order ties match exactly.
    ok = bool((diff <= ulp).all()) and got.dtype == torch.bfloat16 and exact >= 0.999
    rel = (diff.max() / want.abs().max().clamp_min(1e-30)).item()
    print(f"{'ok  ' if ok else 'FAIL'} {name}: exact {exact:.4f} max_rel {rel:.2e}")
    return ok


def run_case(m, n, k, per_token, per_channel, device):
    x = torch.randn(m, k, device=device) * 3
    w = torch.randn(n, k, device=device) * 0.05
    w[: n // 7] *= 20  # spread the per-channel scales
    x[m // 2 :] *= 0.01  # and the per-token ones
    xq, xs = quantize(x, per_token)
    wq, ws = quantize(w, per_channel)
    got = gemm_a8w8_bpreshuffle_gfx1201(xq, shuffle_weight_gfx1201(wq), xs, ws)
    return check(
        f"M{m} N{n} K{k} x_scale {'token' if per_token else 'tensor'} "
        f"w_scale {'channel' if per_channel else 'tensor'}",
        got,
        reference(xq, wq, xs, ws),
    )


def expect_error(name, fn, exc):
    try:
        fn()
    except exc:
        print(f"ok   {name}: raised {exc.__name__}")
        return True
    print(f"FAIL {name}: no {exc.__name__}")
    return False


def bench(device):
    # MiniMax-H3-Image DiT linear layers at 1024x1024 (M = 4144 tokens).
    for m, n, k in ((4144, 21504, 5376), (4144, 28672, 5376), (4144, 5376, 14336)):
        bench_shape(m, n, k, device)


def bench_shape(m, n, k, device):
    xq, xs = quantize(torch.randn(m, k, device=device), True)
    wq, ws = quantize(torch.randn(n, k, device=device) * 0.05, True)
    wp = shuffle_weight_gfx1201(wq)
    wt = wq.t()
    one = torch.ones(1, device=device)
    runs = {
        "gfx1201": lambda: gemm_a8w8_bpreshuffle_gfx1201(xq, wp, xs, ws),
        "torch._scaled_mm": lambda: torch._scaled_mm(
            xq, wt, scale_a=xs, scale_b=ws.reshape(1, -1), out_dtype=torch.bfloat16
        ),
        "torch._scaled_mm tensorwise": lambda: torch._scaled_mm(
            xq, wt, scale_a=one, scale_b=one, out_dtype=torch.bfloat16
        ),
    }
    times = {}
    for name, fn in runs.items():
        try:
            fn()
        except RuntimeError as err:
            print(f"  {name}: {str(err).splitlines()[0]}")
            continue
        for _ in range(5):
            fn()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(50):
            fn()
        end.record()
        torch.cuda.synchronize()
        times[name] = start.elapsed_time(end) / 50
    line = ", ".join(
        f"{name} {ms:.3f} ms ({2 * m * n * k / ms / 1e9:.0f} TFLOPS)"
        for name, ms in times.items()
    )
    print(f"M{m} N{n} K{k}: {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    if get_gfx() != "gfx1201":
        print(f"skip: gemm_a8w8_bpreshuffle_gfx1201 needs gfx1201, got {get_gfx()}")
        return 0
    torch.manual_seed(0)
    device = "cuda"
    ok = True

    wq = torch.randn(512, 384, device=device).to(dtypes.fp8)
    ok &= bool(
        torch.equal(
            unshuffle_weight_gfx1201(shuffle_weight_gfx1201(wq)).view(torch.uint8),
            wq.view(torch.uint8),
        )
    )
    print(
        f"{'ok  ' if ok else 'FAIL'} shuffle_weight_gfx1201 matches the documented layout"
    )

    for m, n, k in (
        (1, 256, 128),
        (7, 512, 384),
        (128, 256, 256),
        (129, 768, 640),
        (300, 1024, 1152),
        (4144, 5376, 5376),
        (1000, 2048, 14336),
    ):
        ok &= run_case(m, n, k, True, True, device)
    ok &= run_case(77, 512, 256, False, True, device)
    ok &= run_case(77, 512, 256, True, False, device)
    ok &= run_case(77, 512, 256, False, False, device)

    # Zero rows and zero weight columns stay exactly zero.
    xq, xs = quantize(torch.randn(200, 256, device=device), True)
    wq, ws = quantize(torch.randn(256, 256, device=device), True)
    xq[5:9] = 0
    wq[17] = 0
    got = gemm_a8w8_bpreshuffle_gfx1201(xq, shuffle_weight_gfx1201(wq), xs, ws)
    zeros_ok = bool((got[5:9] == 0).all() and (got[:, 17] == 0).all())
    print(f"{'ok  ' if zeros_ok else 'FAIL'} zero rows and columns")
    ok &= zeros_ok

    # Non-default stream.
    stream = torch.cuda.Stream()
    xq, xs = quantize(torch.randn(333, 512, device=device), True)
    wq, ws = quantize(torch.randn(512, 512, device=device), True)
    wp = shuffle_weight_gfx1201(wq)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        got = gemm_a8w8_bpreshuffle_gfx1201(xq, wp, xs, ws)
    stream.synchronize()
    ok &= check("non-default stream", got, reference(xq, wq, xs, ws))

    wp = shuffle_weight_gfx1201(wq)
    ok &= expect_error(
        "N % 256 != 0",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(
            xq, shuffle_weight_gfx1201(wq[:480]), xs, ws[:480]
        ),
        ValueError,
    )
    ok &= expect_error(
        "K % 128 != 0",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(
            xq[:, :448].contiguous(),
            shuffle_weight_gfx1201(wq[:, :448].contiguous()),
            xs,
            ws,
        ),
        ValueError,
    )
    ok &= expect_error(
        "WQ K != XQ K",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(xq[:, :256].contiguous(), wp, xs, ws),
        ValueError,
    )
    ok &= expect_error(
        "fp16 output",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(xq, wp, xs, ws, dtype=torch.float16),
        ValueError,
    )
    ok &= expect_error(
        "bf16 activation",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(xq.to(torch.bfloat16), wp, xs, ws),
        TypeError,
    )
    ok &= expect_error(
        "x_scale length",
        lambda: gemm_a8w8_bpreshuffle_gfx1201(xq, wp, xs[:10], ws),
        ValueError,
    )
    ok &= expect_error(
        "shuffle N % 32 != 0",
        lambda: shuffle_weight_gfx1201(wq[:500]),
        ValueError,
    )

    if args.bench:
        bench(device)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
