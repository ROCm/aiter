# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes, get_hip_quant, get_torch_quant, get_triton_quant
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import (
    dynamic_per_group_scaled_quant,
    fused_dynamic_mx_quant_moe_sort,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale, f32_to_mxfp4
from aiter.utility.mx_types import MxDtypeInt

torch.set_default_device("cuda")


@benchmark()
def test_quant(m, n, q_type, q_dtype, h_dtype):
    dim = (m, n)

    input = torch.randn(dim, dtype=h_dtype)
    ref, ref_scale = get_torch_quant(q_type)(input, quant_dtype=q_dtype)

    q_funcs = {
        "triton": get_triton_quant,
        "hip": get_hip_quant,
    }
    ret = {}
    for name, q_func in q_funcs.items():
        q_func = q_func(q_type)
        (out, scale), us1 = run_perftest(q_func, input, quant_dtype=q_dtype)
        err1 = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant",
        )
        checkAllclose(
            ref_scale.to(dtypes.fp32),
            scale.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: dynamic quant scale",
        )
        ret[f"{name} dq"] = us1
        ret[f"{name} dq err"] = err1
        if q_type == aiter.QuantType.per_Tensor:
            (out, scale), us2 = run_perftest(
                q_func, input, ref_scale, quant_dtype=q_dtype
            )
            err2 = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=1e-3,
                atol=1e-3,
                msg=f"{name}: static  quant",
            )
            ret[f"{name} sq"] = us2
            ret[f"{name} sq err"] = err2

    return ret


@benchmark()
def test_quant_1x128_e8m0(m, n, q_dtype, h_dtype, shuffle, strided):
    """Byte-exact check of dynamic_per_group_scaled_quant, group 128, e8m0 scales."""
    G = 128
    fp4 = q_dtype == dtypes.fp4x2
    src = torch.randn((m, n + (G if strided else 0)), dtype=h_dtype)
    x = src[:, :n]
    out = torch.empty((m, n // 2 if fp4 else n), dtype=q_dtype)
    scale = torch.empty((m, n // G), dtype=dtypes.fp8_e8m0)
    _, us = run_perftest(
        dynamic_per_group_scaled_quant, out, x, scale, G, shuffle_scale=shuffle
    )

    xf = x.float().view(m, n // G, G)
    _fnuz = get_gfx() == "gfx942"
    mx = (
        MxDtypeInt.FP4_E2M1
        if fp4
        else (MxDtypeInt.FP8_E4M3_FNUZ if _fnuz else MxDtypeInt.FP8_E4M3)
    )
    ref_scale = f32_to_mx_e8m0_scale(xf.abs().amax(-1), dtype=mx).view(torch.uint8)
    xs = xf / (ref_scale.to(torch.int32) - 127).float().exp2()[..., None]
    if fp4:
        ref_out = f32_to_mxfp4(xs.view(m, n)).view(torch.uint8)
    else:
        fp8_max = torch.finfo(dtypes.fp8).max
        ref_out = xs.clamp(-fp8_max, fp8_max).to(dtypes.fp8).view(m, n)
        ref_out = ref_out.view(torch.uint8)
    got_scale = scale.view(torch.uint8)
    if shuffle:
        got_scale = got_scale.view(n // G, m).t()
    scale_err = int((got_scale != ref_scale).sum())
    out_err = int((out.view(torch.uint8) != ref_out).sum())
    assert scale_err == 0 and out_err == 0, f"{scale_err=} {out_err=}"
    return {"us": us}


def _mx32_indices(rows, scale_n):
    """Flat MX-shuffle index of every valid (row, group) for group size 32."""
    pad = ((scale_n + 7) // 8) * 8
    x = torch.arange(rows)[:, None]
    y = torch.arange(scale_n)[None, :]
    return (
        (x // 32 * pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )


_GUARD = 256
_GUARD_BYTE = 0xA5


def _guarded(numel, dtype):
    """A contiguous buffer of `numel` elements followed by a canary tail."""
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    buf = torch.full((nbytes + _GUARD,), _GUARD_BYTE, dtype=torch.uint8)
    return buf, buf[:nbytes].view(dtype)


def _assert_guard(buf, what):
    tail = buf[-_GUARD:]
    assert torch.all(tail == _GUARD_BYTE).item(), f"{what}: write past the buffer end"


def _plant_nonfinite(x, G, pattern):
    """Put NaN/Inf into a few groups spread over rows, columns and tiles."""
    m, n = x.shape
    sn = n // G
    spots = [(0, 0), (m // 3, sn // 2), (m - 1, sn - 1), (m // 2, min(1, sn - 1))]
    if pattern == "snan":
        # The group max comes before the sNaN: gfx9's v_max_f32 used to restart the
        # amax at a bf16 sNaN and drop it.
        snan = 0x7F81 if x.dtype == dtypes.bf16 else 0x7C01
        for r, g in spots:
            x[r, g * G + 1] = 100.0
            x[r, g * G + 2 : g * G + 3].view(torch.int16).fill_(snan)
        return
    values = {
        "nan": [float("nan")],
        "posinf": [float("inf")],
        "neginf": [-float("inf")],
        "mixed": [float("nan"), float("inf"), -float("inf")],
    }[pattern]
    for i, (r, g) in enumerate(spots):
        x[r, g * G + (i * 7) % G] = values[i % len(values)]


def _device_arch():
    """Arch of the current device; get_gfx() is cached for the whole process."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def _mx_reference(x, G, q_dtype):
    """Expected (data bytes, [m, n / G] scale bytes) on the current device.

    MXFP8 non-finite inputs: the amax skips NaN and NaN elements are NaN (0xff,
    0x80 on FNUZ). A group holding an Inf gets scale 2^127 (0xfe): its Inf turns
    into NaN and its finite data is divided by 2^127 as usual. Compare against
    _fp8_bytes(out).
    """
    m, n = x.shape
    fp4 = q_dtype == dtypes.fp4x2
    xf = x.float().view(m, n // G, G)
    fnuz = _device_arch() == "gfx942"
    mx = (
        MxDtypeInt.FP4_E2M1
        if fp4
        else (MxDtypeInt.FP8_E4M3_FNUZ if fnuz else MxDtypeInt.FP8_E4M3)
    )
    amax = torch.where(torch.isnan(xf), 0.0, xf.abs()).amax(-1).clamp(min=1e-10)
    ref_scale = f32_to_mx_e8m0_scale(amax, dtype=mx).view(torch.uint8)
    if not fp4:
        ref_scale = ref_scale.clamp(max=0xFE)
    xs = xf / (ref_scale.to(torch.int32) - 127).float().exp2()[..., None]
    if fp4:
        return f32_to_mxfp4(xs.view(m, n)).view(torch.uint8), ref_scale
    fp8 = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8).max
    out = xs.nan_to_num(0.0).clamp(-fp8_max, fp8_max).to(fp8).view(torch.uint8)
    out[torch.isnan(xs) | torch.isinf(xs)] = 0x80 if fnuz else 0xFF
    return out.reshape(m, n), ref_scale


def _fp8_bytes(out):
    """fp8 output bytes with both e4m3fn NaNs (0x7f, 0xff) as 0xff: the NaN sign
    an Inf converts to differs between archs."""
    b = out.view(torch.uint8)
    return b if _device_arch() == "gfx942" else b.masked_fill(b == 0x7F, 0xFF)


def _fp4_bytes(out, x):
    """fp4x2 output bytes with the nibbles of NaN inputs cleared: FP4 has no NaN, and
    what a NaN converts to is up to the arch."""
    b = out.view(torch.uint8).clone()
    nan = torch.isnan(x).reshape(*b.shape, 2)
    b[nan[..., 0]] &= 0xF0
    b[nan[..., 1]] &= 0x0F
    return b


def test_quant_1x32_e8m0(m, n, q_dtype, h_dtype, shuffle, strided, pattern=None):
    """Byte-exact dynamic_per_group_scaled_quant, group 32, e8m0 scales.

    Shapes with M % 32 == 0 and (N/32) % 8 == 0 take the gfx1250 tile path
    (fp4 also needs at least one 32x8 scale tile per CU); the others stay on
    the generic kernel. `pattern` plants NaN/Inf (fp4: "snan" only).
    """
    G = 32
    fp4 = q_dtype == dtypes.fp4x2
    src = torch.randn((m, n + (G if strided else 0)), dtype=h_dtype)
    x = src[:, :n]
    if pattern is not None:
        _plant_nonfinite(x, G, pattern)
    n_out = n // 2 if fp4 else n
    out_buf, out = _guarded(m * n_out, q_dtype)
    out = out.view(m, n_out)
    scale_n = n // G
    if shuffle:
        # The MX shuffle tiles 32 rows x 8 groups, so rows are padded to 32.
        rows_pad = (m + 31) // 32 * 32
        scale_numel = rows_pad * ((scale_n + 7) // 8 * 8)
    else:
        scale_numel = m * scale_n
    scale_buf, scale = _guarded(scale_numel, torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, G, shuffle_scale=shuffle)
    _assert_guard(out_buf, "out")
    _assert_guard(scale_buf, "scale")

    ref_out, ref_scale = _mx_reference(x, G, q_dtype)
    got = scale[_mx32_indices(m, scale_n)] if shuffle else scale.view(m, scale_n)
    scale_err = int((got != ref_scale).sum())
    if fp4:
        ref_out = _fp4_bytes(ref_out, x)
    got_out = _fp4_bytes(out, x) if fp4 else _fp8_bytes(out)
    out_err = int((got_out != ref_out).sum())
    assert (
        scale_err == 0 and out_err == 0
    ), f"{m=} {n=} {shuffle=} {strided=} {pattern=} {scale_err=} {out_err=}"


def test_mxfp8_nonfinite_g128(m, n, h_dtype, shuffle, pattern):
    """NaN/Inf at shapes that reach the group-128 TDM tile path on gfx1250."""
    G = 128
    x = torch.randn((m, n), dtype=h_dtype)
    _plant_nonfinite(x, G, pattern)
    out = torch.empty((m, n), dtype=dtypes.fp8)
    scale_buf, scale = _guarded(m * (n // G), torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, G, shuffle_scale=shuffle)
    _assert_guard(scale_buf, "scale")
    ref_out, ref_scale = _mx_reference(x, G, dtypes.fp8)
    got = scale.view(n // G, m).t() if shuffle else scale.view(m, n // G)
    scale_err = int((got != ref_scale).sum())
    out_err = int((_fp8_bytes(out) != ref_out).sum())
    assert (
        scale_err == 0 and out_err == 0
    ), f"{m=} {n=} {shuffle=} {pattern=} {scale_err=} {out_err=}"


def test_mxfp8_m32k4(m, n, h_dtype, pattern=None):
    """scale_layout_m32k4 follows the same MXFP8 contract as the other layouts."""
    G = 32
    x = torch.randn((m, n), dtype=h_dtype)
    if pattern is not None:
        _plant_nonfinite(x, G, pattern)
    out = torch.empty((m, n), dtype=dtypes.fp8)
    rows_pad = (m + 31) // 32 * 32
    scale_buf, scale = _guarded(rows_pad * (n // G), torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, G, scale_layout_m32k4=True)
    _assert_guard(scale_buf, "scale")

    ref_out, ref_scale = _mx_reference(x, G, dtypes.fp8)
    expected = torch.full((rows_pad, n // G), 0x7F, dtype=torch.uint8)
    expected[:m] = ref_scale
    r = torch.arange(rows_pad)[:, None]
    g = torch.arange(n // G)[None, :]
    idx = (r // 32) * n + (g // 4) * 128 + (r % 32) * 4 + g % 4
    scale_err = int((scale[idx] != expected).sum())
    out_err = int((_fp8_bytes(out) != ref_out).sum())
    assert (
        scale_err == 0 and out_err == 0
    ), f"{m=} {n=} {pattern=} {scale_err=} {out_err=}"


def test_heterogeneous_devices():
    """Launch choices must follow each device's arch, not the first one queried.

    The fp8 encoding is fixed per process (dtypes.fp8 and the C++ torch_fp8), so
    only devices sharing gfx1250's OCP encoding are mixed; gfx942 (FNUZ) is not.
    """
    first = {}
    for i in range(torch.cuda.device_count()):
        arch = torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]
        if arch != "gfx942":
            first.setdefault(arch, i)
    if len(first) < 2 or "gfx1250" not in first or dtypes.fp8 != torch.float8_e4m3fn:
        aiter.logger.info("heterogeneous-device test skipped: archs %s", sorted(first))
        return
    order = [first["gfx1250"]] + [d for a, d in first.items() if a != "gfx1250"]
    for dev in order + order[::-1]:
        with torch.cuda.device(dev):
            test_mxfp8_nonfinite_g128(4096, 7168, dtypes.bf16, True, "nan")
            test_quant_1x32_e8m0(1024, 7168, dtypes.fp8, dtypes.bf16, True, False)
    aiter.logger.info("heterogeneous-device test passed on %s", sorted(first))


def _assert_tiled_paths_covered(g32_shapes, g128_shapes):
    """Fail if no case satisfies the gfx1250 tiled-path dispatch predicates."""
    if get_gfx() != "gfx1250":
        return
    cu = torch.cuda.get_device_properties(torch.cuda.current_device())
    cu = cu.multi_processor_count

    def pad8(v):
        return (v + 7) // 8 * 8

    # fp4 / fp8 32x512 TDM tiles, ahead of the mx32 tiles.
    def tdm32_path(m, n):
        return m % 32 == 0 and n % 512 == 0 and (m // 16) * (n // 512) * 2 >= cu

    # A single step (32x512 shuffled, 16x512 row-major) while every block is
    # resident: 4 blocks of 512 threads or 8 of 256 per CU. Row-major fp8 keeps
    # the generic kernel there.
    def tdm32_single(m, n, shuffle):
        if shuffle:
            return tdm32_path(m, n) and (n // 512) * (m // 32) <= 4 * cu
        return tdm32_path(m, n) and (n // 512) * pad8(m // 16) <= 8 * cu

    for shuffle in (True, False):
        layout = "shuffled" if shuffle else "row-major"
        assert any(
            tdm32_single(m, n, shuffle) for m, n in g32_shapes
        ), f"no group-32 {layout} case takes the single-step TDM tile on {cu} CUs"
        assert any(
            tdm32_path(m, n) and not tdm32_single(m, n, shuffle) for m, n in g32_shapes
        ), f"no group-32 {layout} case takes the two-step TDM tile on {cu} CUs"
    # Row-major bands are padded to a multiple of 8; the padding blocks exit early.
    assert any(
        tdm32_single(m, n, False) and (m // 16) % 8 for m, n in g32_shapes
    ), f"no fp4 group-32 row-major TDM case has padded bands on {cu} CUs"
    assert any(
        tdm32_path(m, n) and not tdm32_single(m, n, False) and (m // 32) % 8
        for m, n in g32_shapes
    ), f"no group-32 row-major two-step TDM case has padded bands on {cu} CUs"

    def mx32_tiles(m, n):
        return (m // 32) * (n // 256)

    # Shuffled group-32 shapes the TDM tiles leave to the mx32 tiles.
    def mx32_path(q, m, n):
        min_tiles = 2 * cu if q == "fp8" else cu // 2
        return (
            m % 32 == 0
            and (n // 32) % 8 == 0
            and mx32_tiles(m, n) >= min_tiles
            and not tdm32_path(m, n)
        )

    # fp4 takes a block per tile while all of them are resident (8 blocks of 256
    # threads per CU), two tiles along the row beyond that.
    def fp4_paired(m, n):
        return mx32_path("fp4", m, n) and mx32_tiles(m, n) > 8 * cu

    for q in ("fp8", "fp4"):
        assert any(
            mx32_path(q, m, n) for m, n in g32_shapes
        ), f"no {q} group-32 case reaches the mx32 tile path on {cu} CUs"
    assert any(
        mx32_path("fp4", m, n) and not fp4_paired(m, n) for m, n in g32_shapes
    ), f"no fp4 group-32 case takes a block per tile on {cu} CUs"
    # An odd tile count per row leaves the last paired block half full.
    assert any(
        fp4_paired(m, n) and (n // 256) % 2 for m, n in g32_shapes
    ), f"no fp4 group-32 paired-tile case has an odd tile count per row on {cu} CUs"

    def tdm_tile_path(m, n):
        if m % 16 or n % 512:
            return False
        tiles, bands = n // 512, m // 16
        return tiles * bands >= 4 * cu and all(
            tiles * bands * 8 >= launched * 7
            for launched in (pad8(tiles) * bands, tiles * pad8(bands))
        )

    # Single-step tiles (16x256 for fp8) while every block is resident: 16 per CU.
    def tdm_single_step(m, n):
        return tdm_tile_path(m, n) and pad8(n // 256) * (m // 16) <= 16 * cu

    assert any(
        tdm_tile_path(m, n) and not tdm_single_step(m, n) for m, n in g128_shapes
    ), f"no group-128 case reaches the two-step TDM tile path on {cu} CUs"
    assert any(
        tdm_single_step(m, n) for m, n in g128_shapes
    ), f"no group-128 case reaches the single-step TDM tile path on {cu} CUs"


def test_mxfp8_nonfinite(group_size, shuffle, h_dtype, pattern):
    """NaN/Inf groups give the same MXFP8 bytes on every arch up to the NaN sign;
    see _mx_reference."""
    x = torch.ones((32, group_size * 8), dtype=h_dtype)
    x[:, 1] = -1.0
    for row in range(32):
        col = (row * 3) % 8
        group = x[row, col * group_size : (col + 1) * group_size]
        if pattern == "all_nan":
            group.fill_(float("nan"))
        elif pattern == "snan":
            group.view(torch.int16)[-1] = 0x7F81 if h_dtype == dtypes.bf16 else 0x7C01
        elif pattern == "nan_inf":
            group[:3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        else:
            group[-1] = {"nan": float("nan"), "posinf": float("inf")}.get(
                pattern, -float("inf")
            )
    out = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((32, 8), dtype=torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, group_size, shuffle)

    expected, expected_scale = _mx_reference(x, group_size, dtypes.fp8)
    if shuffle and group_size == 32:
        shuffled = torch.empty_like(expected_scale).flatten()
        for row, col in itertools.product(range(32), range(8)):
            idx = (row % 16) * 4 + row // 16 + (col % 4) * 64 + (col // 4) * 2
            shuffled[idx] = expected_scale[row, col]
        expected_scale = shuffled
    elif shuffle:
        expected_scale = expected_scale.t().contiguous()
    assert torch.equal(_fp8_bytes(out), expected), pattern
    assert torch.equal(scale.flatten(), expected_scale.flatten()), pattern


def test_mxfp8_large_finite(group_size, shuffle, amax_bits):
    """Finite BF16 values up to the largest stay finite and follow the scale rule."""
    amax = torch.tensor([amax_bits], dtype=torch.int16).view(dtypes.bf16).float()
    group = (torch.linspace(-1, 1, group_size) * amax).to(dtypes.bf16)
    x = group.repeat(32, 8)
    out = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((32, 8), dtype=torch.uint8)
    dynamic_per_group_scaled_quant(out, x, scale, group_size, shuffle)

    exponent = torch.ceil(torch.log2(amax.double() / torch.finfo(dtypes.fp8).max))
    expected = (x.double() / torch.exp2(exponent).item()).to(dtypes.fp8)
    assert torch.isfinite(out.float()).all().item()
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8)), amax_bits
    assert torch.all(scale == int(exponent.item()) + 127).item(), amax_bits


def test_mx_quant_moe_sort_threshold(m, n, h_dtype, q_dtype, pattern=None):
    """fused_dynamic_mx_quant_moe_sort takes the fused kernel up to M = 2048 / topk at
    stage 1 and the split path above it; both must follow _mx_reference."""
    G, topk, block_m = 32, 8, 32
    fp4 = q_dtype == dtypes.fp4x2
    x = torch.randn((m, n), dtype=h_dtype)
    if pattern is not None:
        _plant_nonfinite(x, G, pattern)
    rows_pad = (m + block_m - 1) // block_m * block_m
    sorted_ids = torch.full((rows_pad,), m, dtype=torch.int32)
    sorted_ids[:m] = torch.arange(m, dtype=torch.int32)
    num_valid_ids = torch.tensor([rows_pad, m], dtype=torch.int32)
    out, scale = fused_dynamic_mx_quant_moe_sort(
        x, sorted_ids, num_valid_ids, m, topk, block_m, quant_dtype=q_dtype
    )

    ref_out, ref_scale = _mx_reference(x, G, q_dtype)
    got_scale = scale.view(torch.uint8).flatten()[_mx32_indices(m, n // G)]
    if fp4:
        ref_out = _fp4_bytes(ref_out, x)
    got_out = _fp4_bytes(out, x) if fp4 else _fp8_bytes(out)
    scale_err = int((got_scale != ref_scale).sum())
    out_err = int((got_out != ref_out).sum())
    assert (
        scale_err == 0 and out_err == 0
    ), f"{m=} {n=} {h_dtype=} {q_dtype=} {pattern=} {scale_err=} {out_err=}"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk (default 4096 8192; *_e8m0: see E8M0_SHAPES).
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk (default 1 .. 163840; *_e8m0: see E8M0_SHAPES).
    e.g.: -m 32""",
)
d_quant = {
    "fp8_tensor": (aiter.QuantType.per_Tensor, dtypes.fp8),
    "fp8_token": (aiter.QuantType.per_Token, dtypes.fp8),
    "fp8_1x128": (aiter.QuantType.per_1x128, dtypes.fp8),
    "i8_token": (aiter.QuantType.per_Token, dtypes.i8),
    # 'fp4x2-1x32': (aiter.QuantType.per_1x32, dtypes.fp4x2),
}
d_quant_e8m0 = {
    "fp8_1x128_e8m0": dtypes.fp8,
    "fp4_1x128_e8m0": dtypes.fp4x2,
}
d_quant_special = ["fp8_mx_nonfinite"]
# Odd M, the paired-store and TDM tile paths.
E8M0_SHAPES = [
    (1, 7168),
    (100, 1536),
    (1535, 7168),
    (1536, 1152),
    (1536, 7168),
    (1536, 16384),
    (1528, 16384),
    (2048, 12288),
    (16384, 7168),
]
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(d_quant) + list(d_quant_e8m0) + d_quant_special,
    nargs="*",
    default=list(d_quant) + list(d_quant_e8m0) + d_quant_special,
    help="""Quantization type.
    e.g.: -q fp8_tensor""",
)

args = parser.parse_args()
list_quant = [d_quant[key] for key in args.quant if key in d_quant]
list_n = args.n or [4096, 8192]
list_m = args.m or [1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 163840]

for (
    (q_type, q_dtype),
    h_dtype,
) in itertools.product(list_quant, args.dtype):
    df = []
    for n in list_n:
        for m in list_m:
            ret = test_quant(m, n, q_type, q_dtype, h_dtype)
            df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("quant summary (markdown):\n%s", df_md)

e8m0_shapes = (
    list(itertools.product(args.m or [1536], args.n or [7168]))
    if args.m or args.n
    else E8M0_SHAPES
)
_skip_fp4_e8m0 = get_gfx() == "gfx942"
for key in [k for k in args.quant if k in d_quant_e8m0]:
    if _skip_fp4_e8m0 and d_quant_e8m0[key] == dtypes.fp4x2:
        aiter.logger.info("Skipping %s on gfx942 (no HW fp4 convert)", key)
        continue
    df = []
    for h_dtype, (m, n), shuffle, strided in itertools.product(
        args.dtype, e8m0_shapes, [True, False], [False, True]
    ):
        df.append(
            test_quant_1x128_e8m0(m, n, d_quant_e8m0[key], h_dtype, shuffle, strided)
        )
    df = pd.DataFrame(df)
    aiter.logger.info("%s summary (markdown):\n%s", key, df.to_markdown(index=False))

if "fp8_mx_nonfinite" in args.quant and get_gfx() in ("gfx950", "gfx1250"):
    # Group-32 shapes on both sides of the tile-path gates, odd fp4 tile counts per
    # row band for both mx32 block shapes (4096, 2304) and (8192, 2304), padded TDM
    # bands for one and two steps (1056, 7168) and (3104, 7168), and narrow rows.
    g32_shapes = (
        (32, 256),
        (64, 768),
        (128, 7168),
        (40, 320),
        (32, 160),
        (1024, 7168),
        (1056, 7168),
        (3104, 7168),
        (4096, 7168),
        (4096, 1536),
        (4096, 2304),
        (8192, 2304),
        (16384, 1536),
        (32768, 1024),
    )
    g128_shapes = ((4096, 7168), (2048, 12288), (1536, 7168))
    _assert_tiled_paths_covered(g32_shapes, g128_shapes)
    for m, n in g32_shapes:
        for q_dtype, h_dtype, shuffle, strided in itertools.product(
            [dtypes.fp8, dtypes.fp4x2],
            [dtypes.bf16, dtypes.fp16],
            [True, False],
            [False, True],
        ):
            test_quant_1x32_e8m0(m, n, q_dtype, h_dtype, shuffle, strided)
    for (m, n), h_dtype, shuffle, pattern in itertools.product(
        ((1024, 7168), (40, 320)),
        [dtypes.bf16, dtypes.fp16],
        [True, False],
        ["nan", "posinf", "neginf", "mixed"],
    ):
        test_quant_1x32_e8m0(m, n, dtypes.fp8, h_dtype, shuffle, False, pattern)
    for (m, n), h_dtype, shuffle, pattern in itertools.product(
        g128_shapes,
        [dtypes.bf16, dtypes.fp16],
        [True, False],
        ["nan", "posinf", "neginf", "mixed"],
    ):
        test_mxfp8_nonfinite_g128(m, n, h_dtype, shuffle, pattern)
    # Largest M: the tiled grids would need grid.y > 65535 (2^21 + 32 rows / 32,
    # 2^20 rows / 16), so these take the generic kernel.
    for shuffle in (True, False):
        test_quant_1x32_e8m0(
            (1 << 21) + 32, 512, dtypes.fp8, dtypes.bf16, shuffle, False, "mixed"
        )
        test_quant_1x32_e8m0(
            (1 << 21) + 32, 512, dtypes.fp4x2, dtypes.bf16, shuffle, False
        )
        test_mxfp8_nonfinite_g128(1 << 20, 512, dtypes.bf16, shuffle, "mixed")
    for (m, n), h_dtype, pattern in itertools.product(
        ((40, 256), (1024, 7168)),
        [dtypes.bf16, dtypes.fp16],
        [None, "nan", "posinf", "neginf", "mixed"],
    ):
        test_mxfp8_m32k4(m, n, h_dtype, pattern)
    test_heterogeneous_devices()
    aiter.logger.info("fp8_1x32_e8m0 passed")
if "fp8_mx_nonfinite" in args.quant and get_gfx() in ("gfx942", "gfx950", "gfx1250"):
    for gs, shuffle in itertools.product([32, 64, 128], [True, False]):
        for h_dtype, pattern in itertools.product(
            [dtypes.bf16, dtypes.fp16],
            ["nan", "posinf", "neginf", "nan_inf", "all_nan", "snan"],
        ):
            test_mxfp8_nonfinite(gs, shuffle, h_dtype, pattern)
        for amax_bits in [0x7F60, 0x7F61, 0x7F69, 0x7F7F]:
            test_mxfp8_large_finite(gs, shuffle, amax_bits)
    # Stage-1 MoE quant + sort at topk 8: M = 256 is the last fused size, 288 splits.
    for m, h_dtype, pattern in itertools.product(
        (256, 288),
        [dtypes.bf16, dtypes.fp16],
        [None, "nan", "posinf", "neginf", "mixed", "snan"],
    ):
        test_mx_quant_moe_sort_threshold(m, 7168, h_dtype, dtypes.fp8, pattern)
        if pattern in (None, "snan") and get_gfx() != "gfx942":
            test_mx_quant_moe_sort_threshold(m, 7168, h_dtype, dtypes.fp4x2, pattern)
    if get_gfx() != "gfx942":
        # A large value then an sNaN, on the generic kernel and the gfx1250 tile paths.
        for (m, n), q_dtype, h_dtype, shuffle in itertools.product(
            ((40, 320), (1024, 7168)),
            [dtypes.fp4x2, dtypes.fp8],
            [dtypes.bf16, dtypes.fp16],
            [True, False],
        ):
            test_quant_1x32_e8m0(m, n, q_dtype, h_dtype, shuffle, False, "snan")
    aiter.logger.info("fp8_mx_nonfinite passed")
