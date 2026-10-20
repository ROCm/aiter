# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 gluon MXFP8 GEMM (e4m3 x e4m3, 1x32 e8m0 scales on both): gemm_afp8wfp8_mx_preshuffle."""

import sys

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

import torch

from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    _is_gluon_available,
    gemm_afp8wfp8_mx_preshuffle,
)
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm_128row

SCALE_GROUP = 32

pytestmark = pytest.mark.skipif(
    not _is_gluon_available(),
    reason="gemm_afp8wfp8_mx_preshuffle is gfx1250 (gluon) only",
)


def generate_inputs(M, N, K, seed=0):
    """e4m3 x/w in [-2, 2) and e8m0 scales in [2^-3, 2^1]; returns raw tensors."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.rand((M, K), device="cuda", generator=g) * 4 - 2).to(torch.float8_e4m3fn)
    w = (torch.rand((N, K), device="cuda", generator=g) * 4 - 2).to(torch.float8_e4m3fn)
    xs = torch.randint(
        124, 129, (M, K // SCALE_GROUP), dtype=torch.uint8, device="cuda", generator=g
    )
    ws = torch.randint(
        124, 129, (N, K // SCALE_GROUP), dtype=torch.uint8, device="cuda", generator=g
    )
    return x, w, xs, ws


def layout_scales(s, scale_layout):
    if scale_layout == "128row":
        return shuffle_scale_gemm_128row(s)
    return shuffle_mxfp8fp4_scale(s.contiguous()).view(-1, s.shape[1] * SCALE_GROUP)


def prepare(x, w, xs, ws, scale_layout, x_preshuffled):
    """Kernel-side inputs: preshuffled w (N // 16, K * 16), optional preshuffled x, laid-out scales."""
    M, K = x.shape
    N = w.shape[0]
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(N // 16, K * 16)
    x_k = shuffle_weight(x.view(torch.uint8)) if x_preshuffled else x
    # x scales of a CTA with fewer than 32 rows are read un-shuffled (row-major)
    xs_k = xs.contiguous() if M < 32 else layout_scales(xs, scale_layout)
    return x_k, w_k, xs_k, layout_scales(ws, scale_layout)


def reference(x, w, xs, ws):
    xd = x.float() * torch.exp2(xs.float() - 127).repeat_interleave(SCALE_GROUP, 1)
    wd = w.float() * torch.exp2(ws.float() - 127).repeat_interleave(SCALE_GROUP, 1)
    return xd @ wd.T


def check(y, ref):
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert err < 1e-2, f"rel_l2 {err:.3e}"
    # elementwise: within a couple of bf16 ULPs of the value range
    tol = 2e-2 * ref.abs().max().item()
    assert (y.float() - ref).abs().max().item() <= tol


SHAPES = [
    (1, 1024, 256),
    (16, 4096, 1024),
    (31, 7168, 512),
    (32, 1024, 1024),
    (150, 2112, 1024),
    (256, 8192, 2048),
    (1024, 1024, 1024),
    (2048, 7168, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 1024),
]


@pytest.mark.parametrize("M, N, K", SHAPES)
@pytest.mark.parametrize("scale_layout", ["n32k4", "128row"])
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_mx(M, N, K, scale_layout, x_preshuffled):
    if x_preshuffled and M % 16:
        pytest.skip("x_preshuffled needs M % 16 == 0")
    x, w, xs, ws = generate_inputs(M, N, K)
    x_k, w_k, xs_k, ws_k = prepare(x, w, xs, ws, scale_layout, x_preshuffled)
    y = gemm_afp8wfp8_mx_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        x_preshuffled=x_preshuffled,
        scale_layout=scale_layout,
    )
    check(y, reference(x, w, xs, ws))


CLUSTER_CONFIG = dict(
    BLOCK_SIZE_M=1024,
    BLOCK_SIZE_N=1024,
    BLOCK_SIZE_K=128,
    num_warps=4,
    NUM_BUFFERS=3,
    num_ctas=16,
    GROUP_M=8,
    B2B_WMMA=1,
    NUM_KSUB=1,
    EPI_SPLIT=1,
)


@pytest.mark.parametrize("EPI_SPLIT", [1, 2])
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_mx_cluster_config(EPI_SPLIT, x_preshuffled):
    """Explicit 16-CTA cluster config, incl. the sliced epilogue and a ragged N."""
    M, N, K = 2048, 3072, 2048
    x, w, xs, ws = generate_inputs(M, N, K, seed=1)
    x_k, w_k, xs_k, ws_k = prepare(x, w, xs, ws, "n32k4", x_preshuffled)
    y = gemm_afp8wfp8_mx_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        config=dict(CLUSTER_CONFIG, EPI_SPLIT=EPI_SPLIT),
        x_preshuffled=x_preshuffled,
    )
    check(y, reference(x, w, xs, ws))


def test_gemm_afp8wfp8_mx_rejects_layout_config_keys():
    x, w, xs, ws = generate_inputs(256, 256, 256)
    x_k, w_k, xs_k, ws_k = prepare(x, w, xs, ws, "n32k4", False)
    with pytest.raises(ValueError):
        gemm_afp8wfp8_mx_preshuffle(
            x_k, w_k, xs_k, ws_k, config=dict(CLUSTER_CONFIG, SCALE_MODE=2)
        )


# --- fp8 element formats and coarser e8m0 scale granularities ------------------------

FP8_DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def generate_inputs_fmt(M, N, K, a_fmt, b_fmt, x_group, w_block, seed=0):
    """fp8 x/w of the given formats, x scales per (1, x_group), w scales per w_block."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.rand((M, K), device="cuda", generator=g) * 4 - 2).to(FP8_DTYPES[a_fmt])
    w = (torch.rand((N, K), device="cuda", generator=g) * 4 - 2).to(FP8_DTYPES[b_fmt])
    xs = torch.randint(
        124, 129, (M, K // x_group), dtype=torch.uint8, device="cuda", generator=g
    )
    bn, bk = w_block
    ws = torch.randint(
        124, 129, (-(-N // bn), K // bk), dtype=torch.uint8, device="cuda", generator=g
    )
    return x, w, xs, ws


def reference_fmt(x, w, xs, ws, x_group, w_block):
    N = w.shape[0]
    bn, bk = w_block
    xd = x.float() * torch.exp2(xs.float() - 127).repeat_interleave(x_group, 1)
    wsd = (
        torch.exp2(ws.float() - 127)
        .repeat_interleave(bn, 0)[:N]
        .repeat_interleave(bk, 1)
    )
    return xd @ (w.float() * wsd).T


def run_fmt(
    M, N, K, a_fmt, b_fmt, x_group, w_block, scale_layout="n32k4", x_preshuffled=False
):
    x, w, xs, ws = generate_inputs_fmt(M, N, K, a_fmt, b_fmt, x_group, w_block)
    if x_group == 128:
        xs_k = xs.T.contiguous()  # compact column-major (K // 128, M)
    elif M < 32:
        xs_k = xs.contiguous()
    else:
        xs_k = layout_scales(xs, scale_layout)
    # MX 1x32 w scales are laid out; coarse blocks go in compact, as in the checkpoint
    ws_k = layout_scales(ws, scale_layout) if w_block == (1, 32) else ws
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(N // 16, K * 16).view(w.dtype)
    x_k = shuffle_weight(x.view(torch.uint8)).view(x.dtype) if x_preshuffled else x
    y = gemm_afp8wfp8_mx_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        x_preshuffled=x_preshuffled,
        scale_layout=scale_layout,
        x_scale_group_size=x_group,
        w_scale_block_size=w_block,
    )
    check(y, reference_fmt(x, w, xs, ws, x_group, w_block))


@pytest.mark.parametrize(
    "a_fmt, b_fmt",
    [("e4m3", "e4m3"), ("e5m2", "e5m2"), ("e4m3", "e5m2"), ("e5m2", "e4m3")],
)
@pytest.mark.parametrize(
    "M, N, K", [(16, 4096, 1024), (256, 2112, 2048), (2048, 4096, 2048)]
)
def test_gemm_afp8wfp8_mx_formats(M, N, K, a_fmt, b_fmt):
    run_fmt(M, N, K, a_fmt, b_fmt, 32, (1, 32))


@pytest.mark.parametrize("w_block", [(1, 32), (32, 32), (128, 128)])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 1024, 256),
        (31, 7168, 512),
        (150, 2112, 1024),
        (256, 8192, 2048),
        (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_mx_x1x128(M, N, K, w_block, x_preshuffled):
    """DeepSeek-V3.1-style UE8M0: 1x128 x scales (compact column-major) x coarse w scales."""
    if x_preshuffled and M % 16:
        pytest.skip("x_preshuffled needs M % 16 == 0")
    run_fmt(M, N, K, "e4m3", "e4m3", 128, w_block, x_preshuffled=x_preshuffled)


@pytest.mark.parametrize("w_block", [(32, 32), (128, 128)])
@pytest.mark.parametrize("scale_layout", ["n32k4", "128row"])
@pytest.mark.parametrize("M, N, K", [(31, 1024, 1024), (2048, 7168, 4096)])
def test_gemm_afp8wfp8_mx_w_blocks(M, N, K, w_block, scale_layout):
    run_fmt(M, N, K, "e4m3", "e4m3", 32, w_block, scale_layout=scale_layout)


def test_gemm_afp8wfp8_mx_rejects_bad_args():
    x, w, xs, ws = generate_inputs_fmt(256, 256, 256, "e4m3", "e4m3", 128, (128, 128))
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(16, 256 * 16)
    xs_k = xs.T.contiguous()
    with pytest.raises(ValueError):
        gemm_afp8wfp8_mx_preshuffle(
            x, w_k, xs_k, ws, x_scale_group_size=64, w_scale_block_size=(128, 128)
        )
    with pytest.raises(AssertionError):  # 1x128 x scales must be (K // 128, M)
        gemm_afp8wfp8_mx_preshuffle(
            x, w_k, xs, ws, x_scale_group_size=128, w_scale_block_size=(128, 128)
        )
    with pytest.raises(ValueError):  # unsupported w block
        gemm_afp8wfp8_mx_preshuffle(
            x, w_k, xs_k, ws, x_scale_group_size=128, w_scale_block_size=(64, 64)
        )
    with pytest.raises(ValueError):  # compact w scales of the wrong shape
        gemm_afp8wfp8_mx_preshuffle(
            x,
            w_k,
            xs_k,
            ws.T.contiguous()[:1],
            x_scale_group_size=128,
            w_scale_block_size=(128, 128),
        )
