# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx

HEAD_DIM = 128
K_GROUP_ROWS = 64
K_AMAX_FLOOR = 1e-7
V_PAD_ROWS = 16
V_HEADROOM = 2.25
# Everything except k_int8 is checked bitwise. k_int8 is not: the kernel reduces the K
# mean over tokens in a different order than torch, and a sum sitting on a BF16 rounding
# tie can land either way, shifting the group amax and with it one INT8 code.
TOLERANCE = {"k_int8": (1, 0.1), "v_fp8": (0, 0.0)}
NAMES = ("k_int8", "k_scale", "v_fp8", "v_scale")


def div(a, b):
    """Correctly rounded float32 division.

    torch's float32 divide on ROCm is reciprocal-based and lands 1 ULP off the IEEE
    result on ~0.1% of inputs, which is enough to move an FP8 code. float64 carries
    more than 2p+2 bits, so rounding its quotient back to float32 is exact.
    """
    b = b.double() if torch.is_tensor(b) else b
    return (a.double() / b).float()


def reference(key, value):
    batch, rows, heads, dim = key.shape
    mean = key.float().mean(dim=1).to(torch.bfloat16)

    groups = (rows + K_GROUP_ROWS - 1) // K_GROUP_ROWS
    centered = key.float() - mean.float()[:, None]
    centered = F.pad(centered, (0, 0, 0, 0, 0, groups * K_GROUP_ROWS - rows))
    blocks = centered.view(batch, groups, K_GROUP_ROWS, heads, dim)
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=K_AMAX_FLOOR)
    # The kernel scales by 127 / amax and reports amax / 127; the two are not reciprocal.
    inv_scale = div(torch.full_like(amax, 127.0), amax)
    key_int8 = (
        torch.round(blocks * inv_scale[:, :, None, :, None])
        .clamp(-128.0, 127.0)
        .to(torch.int8)
        .view(batch, groups * K_GROUP_ROWS, heads, dim)[:, :rows]
        .contiguous()
    )
    key_scale = div(amax, 127.0).permute(0, 2, 1).contiguous()

    value_scale = div(value.float().abs().amax(dim=1), V_HEADROOM)
    scaled = value.float().permute(0, 2, 3, 1)
    scaled = torch.where(
        value_scale[..., None] > 0, div(scaled, value_scale[..., None]), 0.0
    )
    padded = (rows + V_PAD_ROWS - 1) // V_PAD_ROWS * V_PAD_ROWS
    value_fp8 = F.pad(scaled, (0, padded - rows)).to(torch.float8_e4m3fn).contiguous()
    return key_int8, key_scale, value_fp8, value_scale.contiguous()


def report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{' ' + detail if detail else ''}")
    return ok


def check(name, got, want):
    ok = True
    for label, a, b in zip(NAMES, got, want):
        if a.shape != b.shape or a.dtype != b.dtype:
            print(
                f"  {name} {label}: {tuple(a.shape)} {a.dtype}"
                f" vs {tuple(b.shape)} {b.dtype}"
            )
            ok = False
            continue
        if label in TOLERANCE:
            max_ulp, max_rate = TOLERANCE[label]
            diff = (a.view(torch.int8).int() - b.view(torch.int8).int()).abs()
            beyond = (diff > max_ulp).sum().item()
            rate = (diff > 0).float().mean().item() * 100
            if beyond or rate > max_rate:
                print(
                    f"  {name} {label}: {beyond} beyond {max_ulp} ULP,"
                    f" {rate:.4g}% differ (limit {max_rate}%)"
                )
                ok = False
        elif not torch.equal(a, b):
            print(f"  {name} {label}: {(a != b).sum().item()} scales not bitwise equal")
            ok = False
    return report(name, ok)


def inputs(batch, rows, heads, seed=0, scale=3.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return tuple(
        (
            torch.randn(batch, rows, heads, HEAD_DIM, device="cuda", generator=g)
            * scale
        ).bfloat16()
        for _ in range(2)
    )


def test_shapes(prepare):
    ok = True
    for batch, rows, heads in (
        (1, 1, 1),
        (1, 65, 7),
        (2, 513, 14),
        (1, 4097, 28),
        (2, 2000, 3),
        (1, 20000, 56),
    ):
        k, v = inputs(batch, rows, heads, seed=rows)
        ok &= check(f"B{batch} S{rows} H{heads}", prepare(k, v), reference(k, v))
    return ok


def test_tails(prepare):
    ok = True
    # Both the 16-row V pad and the 64-row K scale group have to survive a partial tail.
    for rows in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1021, 1027):
        k, v = inputs(2, rows, 3, seed=rows)
        ok &= check(f"tail S{rows}", prepare(k, v), reference(k, v))
    return ok


def test_padding_is_zero(prepare):
    ok = True
    for rows in (1, 17, 65, 1021):
        k, v = inputs(1, rows, 5, seed=rows)
        v_fp8 = prepare(k, v)[2]
        tail = v_fp8[..., rows:].view(torch.uint8)
        ok &= report(
            f"V pad is zero S{rows}",
            tail.numel() == (16 - rows % 16) % 16 * 5 * HEAD_DIM
            and bool((tail == 0).all()),
            f"pad={tail.numel()}",
        )
    return ok


def test_degenerate(prepare):
    ok = True
    k, v = inputs(1, 300, 4, seed=11)
    # Channel 5 of V is all zero -> the scale == 0 branch; head 0 of K is constant
    # across tokens -> the centred group is all zero and hits the amax floor.
    v[..., 5] = 0
    k[:, :, 0] = k[:, 0:1, 0]
    got, want = prepare(k, v), reference(k, v)
    ok &= check("zero V channel and constant K head", got, want)
    ok &= report(
        "zero V channel stays zero",
        bool((got[2][:, :, 5].view(torch.uint8) == 0).all()),
    )
    ok &= report("scales finite", bool(torch.isfinite(got[1]).all()))
    return ok


def test_stream_and_determinism(prepare):
    k, v = inputs(1, 777, 4, seed=3)
    expected = prepare(k, v)
    ok = check("repeatable", prepare(k, v), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = prepare(k, v)
    torch.cuda.current_stream().wait_stream(stream)
    ok &= check("non-default stream", actual, expected)
    return ok


def test_invalid_inputs(prepare):
    ok = True
    k, v = inputs(1, 64, 2)
    cases = {
        "fp32": (k.float(), v.float()),
        "non-contiguous": (k.transpose(1, 2), v.transpose(1, 2)),
        "head dim 64": (k[..., :64].contiguous(), v[..., :64].contiguous()),
        "shape mismatch": (k, v[:, :32].contiguous()),
        "3d": (k[0], v[0]),
    }
    for name, args in cases.items():
        try:
            prepare(*args)
            ok &= report(f"rejects {name}", False)
        except (ValueError, RuntimeError):
            ok &= report(f"rejects {name}", True)
    return ok


def main():
    if get_gfx() != "gfx1201":
        print(f"skip: gfx1201_sage_prepare_h3 requires gfx1201, got {get_gfx()}")
        return 0
    from aiter import gfx1201_sage_prepare_h3 as prepare

    ok = True
    for test in (
        test_shapes,
        test_tails,
        test_padding_is_zero,
        test_degenerate,
        test_stream_and_determinism,
        test_invalid_inputs,
    ):
        ok &= test(prepare)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
