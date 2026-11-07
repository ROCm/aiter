# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch

from aiter import gfx1201_sage_prepare_h3
from aiter.jit.utils.chip_info import get_gfx

HEAD_DIM = 128
K_GROUP_ROWS = 64
V_PAD_ROWS = 16
# The kernel picks key scales from a sampled maximum times a guard margin, so it cannot be
# compared against an exact reference. What must hold is that dequantization recovers the
# input: INT8 over a 64-row group leaves at most half a step of error, and the guard margin
# widens the step, so allow a few steps of slack rather than asserting a bitwise match.
MAX_K_STEPS = 4.0
MAX_V_REL = 0.15
NAMES = ("key_int8", "key_scale", "value_fp8", "value_scale", "value_residual")


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' ' + detail if detail else ''}")
    return ok


def inputs(batch, rows, heads, seed=0, scale=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (
            torch.randn(batch, rows, heads, HEAD_DIM, device="cuda", generator=g)
            * scale
        ).bfloat16()
        for _ in range(2)
    )


def test_shapes():
    ok = True
    for batch, rows, heads in ((1, 1, 1), (1, 64, 5), (2, 4096, 8), (1, 1000, 3)):
        key, value = inputs(batch, rows, heads, seed=rows)
        k8, ks, v8, vs, vr = gfx1201_sage_prepare_h3(key, value)
        groups = (rows + K_GROUP_ROWS - 1) // K_GROUP_ROWS
        padded = (rows + V_PAD_ROWS - 1) // V_PAD_ROWS * V_PAD_ROWS
        want = {
            "key_int8": ((batch, rows, heads, HEAD_DIM), torch.int8),
            "key_scale": ((batch, heads, groups), torch.float32),
            "value_fp8": ((batch, heads, HEAD_DIM, padded), torch.uint8),
            "value_scale": ((batch, heads, HEAD_DIM), torch.float32),
            "value_residual": ((batch, heads, HEAD_DIM), torch.float32),
        }
        for name, got in zip(NAMES, (k8, ks, v8, vs, vr)):
            shape, dtype = want[name]
            if tuple(got.shape) != shape or got.dtype != dtype:
                print(
                    f"  B{batch} S{rows} H{heads} {name}:"
                    f" {tuple(got.shape)} {got.dtype} vs {shape} {dtype}"
                )
                ok = False
        if not (torch.isfinite(ks).all() and torch.isfinite(vs).all()):
            print(f"  B{batch} S{rows} H{heads}: non-finite scale")
            ok = False
    return report("shapes and dtypes", ok)


def test_dequantizes():
    """K and V must come back from their codes within the quantization step."""
    ok = True
    for rows, heads in ((256, 4), (4096, 8)):
        key, value = inputs(1, rows, heads, seed=rows)
        k8, ks, v8, vs, _ = gfx1201_sage_prepare_h3(key, value)
        mean = key.float().mean(dim=1, keepdim=True)
        # key_scale is per 64-row group, laid out [B, H, groups]; broadcast it back.
        step = ks[0].t().repeat_interleave(K_GROUP_ROWS, dim=0)[:rows, :, None]
        err = ((k8[0].float() * step) - (key.float() - mean)[0]).abs()
        worst = (err / step).max().item()
        if worst > MAX_K_STEPS:
            print(f"  S{rows}H{heads} key: {worst:.3g} steps (limit {MAX_K_STEPS})")
            ok = False
        deq = v8[0].view(torch.float8_e4m3fn).float()[..., :rows] * vs[0][..., None]
        ref = value[0].float().permute(1, 2, 0)
        rel = (deq - ref).abs().max().item() / ref.abs().max().item()
        if rel > MAX_V_REL:
            print(f"  S{rows}H{heads} value: rel {rel:.3g} (limit {MAX_V_REL})")
            ok = False
    return report("dequantizes within quantization step", ok)


def test_padding_is_zero():
    rows, heads = 1000, 3
    key, value = inputs(1, rows, heads, seed=7)
    _, _, v8, _, _ = gfx1201_sage_prepare_h3(key, value)
    tail = v8[..., rows:]
    ok = tail.numel() == 0 or not tail.any()
    return report("value padding is zero", ok, f"({tail.numel()} padded entries)")


def test_stream_and_determinism():
    key, value = inputs(1, 2048, 4, seed=3)
    base = gfx1201_sage_prepare_h3(key, value)
    again = gfx1201_sage_prepare_h3(key, value)
    ok = all(torch.equal(a, b) for a, b in zip(base, again))
    ok = report("repeatable", ok)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        other = gfx1201_sage_prepare_h3(key, value)
    torch.cuda.current_stream().wait_stream(stream)
    return (
        report(
            "non-default stream", all(torch.equal(a, b) for a, b in zip(base, other))
        )
        and ok
    )


def test_invalid_inputs():
    ok = True
    key, value = inputs(1, 128, 2, seed=1)
    cases = {
        "rejects fp32": (key.float(), value.float()),
        "rejects non-contiguous": (key.transpose(1, 2), value.transpose(1, 2)),
        "rejects head dim 64": (
            key[..., :64].contiguous(),
            value[..., :64].contiguous(),
        ),
        "rejects shape mismatch": (key, value[:, :64].contiguous()),
        "rejects 3d": (key[0], value[0]),
    }
    for name, (k, v) in cases.items():
        try:
            gfx1201_sage_prepare_h3(k, v)
        except (RuntimeError, TypeError, ValueError):
            ok = report(name, True) and ok
        else:
            ok = report(name, False, "no error raised") and ok
    return ok


def main():
    if not get_gfx().startswith("gfx1201"):
        print("skipped: requires gfx1201")
        return 0
    ok = all(
        [
            test_shapes(),
            test_dequantizes(),
            test_padding_is_zero(),
            test_stream_and_determinism(),
            test_invalid_inputs(),
        ]
    )
    print("ALL PASS" if ok else "FAILURES")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
