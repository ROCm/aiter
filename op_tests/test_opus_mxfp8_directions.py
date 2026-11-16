# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 OPUS MXFP8 blockscale GEMM in the NN/TT/TN directions (CO kids 56-85).

Every direction computes the same logical C[M, N] = A[M, K] @ B[K, N] with an
E8M0 1x128 A scale and 128x128 B scale. Only the operand storage differs:

    dir  XQ storage    XQ view strides  WQ storage (pre-shuffle)  WQ view strides  w_scale strides
    NN   [M, K]        (K, 1)           [K, N]                    (1, N)           (1, N/128)
    TT   [K, M]        (1, M)           [N, K]                    (K, 1)           (K/128, 1)
    TN   [K, M]        (1, M)           [K, N]                    (1, N)           (1, N/128)

x_scale is always the contiguous [M, K/128] tensor whose bytes are
column-major ([K/128][M]), as per_group_quant_hip(transpose_scale=True)
emits; the strides (1, M) view of the same bytes is accepted too.

The kernels are called through the public exact-kid entry
opus_gemm(..., family="a8w8_blockscale_bpreshuffle", layout="bpreshuffle", kid=...);
the direction comes from the kid and its generated launcher checks the strides above.
"""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus import opus_gemm
from aiter.ops.shuffle import shuffle_weight

# kid per direction and variant: 256x256 / 256x256 PadM / 256x128 / 256x128 PadM,
# then GroupM 4 / 8 / 16 for each PadM-free tile (kids 68-85).
KIDS = {"NN": 56, "TT": 57, "TN": 58}
VARIANT_KIDS = {
    "256x256": {"NN": 56, "TT": 57, "TN": 58},
    "256x256_padm": {"NN": 59, "TT": 60, "TN": 61},
    "256x128": {"NN": 62, "TT": 63, "TN": 64},
    "256x128_padm": {"NN": 65, "TT": 66, "TN": 67},
}
for _i, (_tile, _option) in enumerate(
    (t, o) for t in ("256x256", "256x128") for o in ("gm4", "gm8", "gm16")
):
    VARIANT_KIDS[f"{_tile}_{_option}"] = dict(
        zip(("NN", "TT", "TN"), range(68 + 3 * _i, 71 + 3 * _i))
    )
# (M, N, K) per variant: PadM kids take M % 16 == 0, the 256x128 kids N % 128 == 0
VARIANT_SHAPES = {
    "256x256": [(256, 512, 256)],
    "256x256_padm": [(272, 512, 1024), (16, 256, 128)],
    "256x128": [(512, 384, 1024), (256, 128, 128)],
    "256x128_padm": [(400, 640, 2048), (16, 128, 256)],
}
# GroupM: 17 M tiles leave a partial last group for 4 / 8 / 16.
for _tile, _n in (("256x256", 768), ("256x128", 384)):
    for _option in ("gm4", "gm8", "gm16"):
        VARIANT_SHAPES[f"{_tile}_{_option}"] = [(17 * 256, _n, 256)]
SHAPES = [
    (256, 256, 128),
    (256, 512, 256),
    (512, 256, 1024),
    (768, 1280, 2048),
    (1024, 1024, 8320),  # crosses the 64 x K128 scale panel
]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="OPUS MXFP8 direction COs are gfx950 only",
)


def _operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=gen).to(dtypes.fp8)
    b = torch.randn(k, n, device="cuda", generator=gen).to(dtypes.fp8)
    # E8M0 exponents around 2^0; logical sa[m, k/128] and sb[k/128, n/128].
    sa = torch.randint(
        124, 131, (m, k // 128), device="cuda", generator=gen, dtype=torch.uint8
    )
    sb = torch.randint(
        124, 131, (k // 128, n // 128), device="cuda", generator=gen, dtype=torch.uint8
    )
    return a, b, sa, sb


def _reference(a, b, sa, sb):
    sa_f = torch.exp2(sa.float() - 127).repeat_interleave(128, dim=1)
    sb_f = (
        torch.exp2(sb.float() - 127)
        .repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
    )
    return (a.float() * sa_f) @ (b.float() * sb_f)


def _x_scale(sa, strided_view):
    # Bytes column-major ([K/128][M]); the public contract is the contiguous
    # [M, K/128] tensor holding those bytes, a strides (1, M) view also works.
    col_major = sa.t().contiguous()
    if strided_view:
        return col_major.t()
    return col_major.view(sa.shape)


def _run(xq, wq, x_scale, w_scale, y, kid):
    return opus_gemm(
        xq,
        wq,
        y,
        family="a8w8_blockscale_bpreshuffle",
        layout="bpreshuffle",
        kid=kid,
        x_scale=x_scale,
        w_scale=w_scale,
    )


def _direction_args(direction, a, b, sa, sb, strided_x_scale):
    if direction[0] == "N":
        xq = a.contiguous()
    else:
        xq = a.t().contiguous().t()  # storage [K, M], view strides (1, M)
    if direction[1] == "N":
        wq = shuffle_weight(b.contiguous(), layout=(16, 16)).t()  # storage [K, N]
        w_scale = sb.contiguous().t()  # storage [K/128, N/128]
    else:
        wq = shuffle_weight(b.t().contiguous(), layout=(16, 16))  # storage [N, K]
        w_scale = sb.t().contiguous()  # storage [N/128, K/128]
    return xq, wq, _x_scale(sa, strided_x_scale), w_scale


@pytest.mark.parametrize("direction", ["NN", "TT", "TN"])
@pytest.mark.parametrize("m,n,k", SHAPES)
@pytest.mark.parametrize("strided_x_scale", [False, True])
def test_mxfp8_direction_matches_reference(direction, m, n, k, strided_x_scale):
    a, b, sa, sb = _operands(m, n, k, seed=m * 7 + n * 13 + k)
    xq, wq, x_scale, w_scale = _direction_args(direction, a, b, sa, sb, strided_x_scale)
    y = torch.empty(m, n, device="cuda", dtype=dtypes.bf16)
    _run(xq, wq, x_scale, w_scale, y, KIDS[direction])
    torch.cuda.synchronize()
    ref = _reference(a, b, sa, sb)
    err = (y.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert (
        err <= 2e-2 * scale
    ), f"{direction} {m}x{n}x{k}: max err {err} vs |ref| {scale}"


@pytest.mark.parametrize("direction", ["NN", "TT", "TN"])
def test_mxfp8_direction_rejects_wrong_layout(direction):
    m, n, k = 256, 256, 128
    a, b, sa, sb = _operands(m, n, k, seed=1)
    xq, wq, x_scale, w_scale = _direction_args(direction, a, b, sa, sb, False)
    y = torch.empty(m, n, device="cuda", dtype=dtypes.bf16)
    other = next(d for d in KIDS if d != direction)
    # The launcher of another direction's kid rejects these operand strides.
    with pytest.raises(RuntimeError, match=f"{other} CO expects XQ strides"):
        _run(xq, wq, x_scale, w_scale, y, KIDS[other])


@pytest.mark.parametrize("direction", ["NN", "TT", "TN"])
@pytest.mark.parametrize(
    "variant,m,n,k",
    [(v, *shape) for v, shapes in VARIANT_SHAPES.items() for shape in shapes],
)
def test_mxfp8_direction_variants_match_reference(direction, variant, m, n, k):
    a, b, sa, sb = _operands(m, n, k, seed=m + 7 * n + 13 * k)
    xq, wq, x_scale, w_scale = _direction_args(direction, a, b, sa, sb, False)
    y = torch.empty(m, n, device="cuda", dtype=dtypes.bf16)
    _run(xq, wq, x_scale, w_scale, y, VARIANT_KIDS[variant][direction])
    torch.cuda.synchronize()
    ref = _reference(a, b, sa, sb)
    err = (y.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert (
        err <= 2e-2 * scale
    ), f"{direction} {variant} {m}x{n}x{k}: max err {err} vs |ref| {scale}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
