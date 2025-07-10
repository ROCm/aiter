# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import torch
import pytest
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_rope import ref_rope_sbhd_fwd, ref_rope_thd_fwd, RotateStyle, ref_rope_2d_fwd
from aiter.ops.triton.rope import (
    rope_fwd,
    rope_fwd_inplace,
    rope_bwd,
    rope_thd_fwd,
    rope_thd_fwd_inplace,
    rope_thd_bwd,
    rope_cached_fwd,
    rope_cached_fwd_inplace,
    rope_cached_positions_fwd,
    rope_cached_positions_fwd_inplace,
    rope_cached_positions_offsets_fwd,
    rope_cached_positions_offsets_fwd_inplace,
    rope_cached_bwd,
    rope_cached_positions_bwd,
    rope_cached_positions_offsets_bwd,
    rope_cached_thd_positions_2c_fwd,
    rope_cached_thd_positions_2c_fwd_inplace,
    rope_cached_thd_positions_offsets_2c_fwd,
    rope_cached_thd_positions_offsets_2c_fwd_inplace,
    rope_cached_thd_positions_2c_bwd,
    rope_cached_thd_positions_offsets_2c_bwd,
    rope_fwd_2d,
    rope_fwd_2d_inplace,
)

DEBUG_MODE = False


def generate_rope_inputs(
    B: int,
    S: int,
    H: int,
    Q: int,
    D: int,
    cached: bool,
    reuse_freqs_front_part: bool,
    nope: bool,
    pos: bool,
    offs: bool,
    two_inputs: bool,
    layout: str,
    dtype: torch.dtype,
    bwd: bool = False,
):
    torch.manual_seed(20)
    random.seed(20)

    device = "cuda"
    if layout == "thd":  # T == S
        assert B == 1, "B should always be 1 in THD layout"
        input_x_shape = (S, Q * H, D)
        input_y_shape = (S, H, D)
        pos_offs_shape = (S,)
    elif layout == "sbhd":
        input_x_shape = (S, B, Q * H, D)
        input_y_shape = (S, B, H, D)
        pos_offs_shape = (S, B)
    else:
        raise NotImplementedError(f"layout '{layout}' not supported")

    x = torch.randn(input_x_shape, dtype=dtype, device="cuda", requires_grad=bwd)
    y = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda", requires_grad=bwd)
        if two_inputs
        else None
    )
    gx = torch.randn(input_x_shape, dtype=dtype, device="cuda") if bwd else None
    gy = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda")
        if bwd and two_inputs
        else None
    )

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2

    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")
    positions = (
        torch.randint(
            max(0, int(S * 0.25) if offs else 0),
            max(1, int(S * 0.75) if offs else S),
            pos_offs_shape,
            device=device,
        )
        if pos
        else None
    )
    offsets = (
        torch.randint(
            max(0, int(S * -0.25)),
            max(1, int(S * 0.25)),
            pos_offs_shape,
            device="cuda",
        )
        if offs
        else None
    )

    cos = torch.cos(freqs) if cached else None
    sin = torch.sin(freqs) if cached else None

    if cached and layout == "thd":
        cos = cos.reshape(S, freqs_D)
        sin = sin.reshape(S, freqs_D)

    return x, y, gx, gy, freqs, positions, offsets, cos, sin


def ref_rope_cached_thd_positions_offsets_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: RotateStyle,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        positions: [num_tokens, ]
        offsets: [num_tokens, ]
    """
    if offsets is None:
        cos = cos[positions]
        sin = sin[positions]
    else:
        cos = cos[positions + offsets]
        sin = sin[positions + offsets]
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.GPTJ:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        elif rotate_style == RotateStyle.NEOX:
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
    else:
        x1 = x
        x2 = x

    # Compute out_x
    ox1 = x1 * cos - x2 * sin
    ox2 = x2 * cos + x1 * sin

    if rotate_style == RotateStyle.GPTJ:
        ox = torch.stack((ox1, ox2), dim=-1).flatten(-2)
    elif rotate_style == RotateStyle.NEOX:
        ox = torch.cat((ox1, ox2), dim=-1)

    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.GPTJ:
            y1 = y[..., ::2]
            y2 = y[..., 1::2]
        elif rotate_style == RotateStyle.NEOX:
            y1 = y[..., : y.shape[-1] // 2]
            y2 = y[..., y.shape[-1] // 2 :]

    # Compute out_x
    oy1 = y1 * cos - y2 * sin
    oy2 = y2 * cos + y1 * sin

    if rotate_style == RotateStyle.GPTJ:
        oy = torch.stack((oy1, oy2), dim=-1).flatten(-2)
    elif rotate_style == RotateStyle.NEOX:
        oy = torch.cat((oy1, oy2), dim=-1)

    return ox, oy


@pytest.mark.parametrize("B", [1, 2, 15, 32, 57])
@pytest.mark.parametrize("S", [2, 10, 32])
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
def test_rope_sbhd_fwd(
    B: int,
    S: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    inplace: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B,
        S,
        H,
        1,
        D,
        cached=False,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=False,
        offs=False,
        two_inputs=False,
        layout="sbhd",
        dtype=dtype,
    )

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")
    torch_out = ref_rope_sbhd_fwd(
        x,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if inplace:
        triton_out = rope_fwd_inplace(
            x,
            freqs,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    else:
        triton_out = rope_fwd(
            x,
            freqs,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("B", [1, 2, 15, 32, 57])
@pytest.mark.parametrize("S", [2, 10, 32])
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_sbhd_bwd(
    B: int,
    S: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B,
        S,
        H,
        1,
        D,
        cached=False,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=False,
        offs=False,
        two_inputs=False,
        layout="sbhd",
        dtype=dtype,
        bwd=True,
    )

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")

    triton_out = rope_bwd(
        gx,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
        transpose_output=False,
    )

    torch_fwd = ref_rope_sbhd_fwd(
        x,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    torch_fwd.backward(gx)
    torch_out = x.grad

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "B, T", [(1, 1), (1, 4), (2, 6), (4, 100), (32, 320), (57, 500)]
)
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.NEOX, RotateStyle.GPTJ])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
def test_rope_thd_fwd(
    B: int,
    T: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    inplace: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        H,
        1,
        D,
        cached=False,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=False,
        offs=False,
        two_inputs=False,
        layout="thd",
        dtype=dtype,
    )

    if B > 1:
        seqlens = random.sample(range(1, T), k=B - 1)
        seqlens = sorted(seqlens)
        seqlens = [0] + seqlens + [T]
    else:
        seqlens = [0, T]
    cu_seqlens = torch.Tensor(seqlens).to(torch.int).to(freqs.device)

    if DEBUG_MODE:
        print(f"cu_seqlens={cu_seqlens}")
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")

    torch_out = ref_rope_thd_fwd(
        x,
        cu_seqlens,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if inplace:
        triton_out = rope_thd_fwd_inplace(
            x,
            cu_seqlens,
            freqs,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    else:
        triton_out = rope_thd_fwd(
            x,
            cu_seqlens,
            freqs,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "B, T", [(1, 1), (1, 4), (2, 6), (4, 100), (32, 320), (57, 500)]
)
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.NEOX, RotateStyle.GPTJ])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_thd_bwd(
    B: int,
    T: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        H,
        1,
        D,
        cached=False,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=False,
        offs=False,
        two_inputs=False,
        layout="thd",
        dtype=dtype,
        bwd=True,
    )

    if B > 1:
        seqlens = random.sample(range(1, T), k=B - 1)
        seqlens = sorted(seqlens)
        seqlens = [0] + seqlens + [T]
    else:
        seqlens = [0, T]
    cu_seqlens = torch.Tensor(seqlens).to(torch.int).to(freqs.device)

    if DEBUG_MODE:
        print(f"cu_seqlens={cu_seqlens}")
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")

    triton_out = rope_thd_bwd(
        gx,
        cu_seqlens,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
        transpose_output=False,
    )

    torch_fwd = ref_rope_thd_fwd(
        x,
        cu_seqlens,
        freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    torch_fwd.backward(gx)
    torch_out = x.grad

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("B", [1, 2, 15, 32, 57])
@pytest.mark.parametrize("S", [4, 10, 32])
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("pos, offs", [(False, False), (True, False), (True, True)])
def test_rope_cached_fwd(
    B: int,
    S: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    pos: bool,
    offs: bool,
    inplace: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B,
        S,
        H,
        1,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=pos,
        offs=offs,
        two_inputs=False,
        layout="sbhd",
        dtype=dtype,
    )

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    torch_out = ref_rope_sbhd_fwd(
        x,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if pos:
        if offs:
            if inplace:
                triton_out = rope_cached_positions_offsets_fwd_inplace(
                    x,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style=rotate_style,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    nope_first=nope_first,
                    transpose_output=False,
                )
            else:
                triton_out = rope_cached_positions_offsets_fwd(
                    x,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style=rotate_style,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    nope_first=nope_first,
                    transpose_output=False,
                )
        else:
            if inplace:
                triton_out = rope_cached_positions_fwd_inplace(
                    x,
                    cos,
                    sin,
                    positions,
                    rotate_style=rotate_style,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    nope_first=nope_first,
                    transpose_output=False,
                )
            else:
                triton_out = rope_cached_positions_fwd(
                    x,
                    cos,
                    sin,
                    positions,
                    rotate_style=rotate_style,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    nope_first=nope_first,
                    transpose_output=False,
                )
    else:
        if inplace:
            triton_out = rope_cached_fwd_inplace(
                x,
                cos,
                sin,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )
        else:
            triton_out = rope_cached_fwd(
                x,
                cos,
                sin,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("B", [1, 2, 15, 32, 57])
@pytest.mark.parametrize("S", [4, 10, 32])
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pos, offs", [(False, False), (True, False), (True, True)])
def test_rope_cached_bwd(
    B: int,
    S: int,
    H: int,
    D: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    pos: bool,
    offs: bool,
    dtype: torch.dtype,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B,
        S,
        H,
        1,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=pos,
        offs=offs,
        two_inputs=False,
        layout="sbhd",
        dtype=dtype,
        bwd=True,
    )

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    if pos:
        if offs:
            triton_out = rope_cached_positions_offsets_bwd(
                gx,
                cos,
                sin,
                positions,
                offsets,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )
        else:
            triton_out = rope_cached_positions_bwd(
                gx,
                cos,
                sin,
                positions,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )
    else:
        triton_out = rope_cached_bwd(
            gx,
            cos,
            sin,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )

    torch_fwd = ref_rope_sbhd_fwd(
        x,
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    )
    torch_fwd.backward(gx)
    torch_out = x.grad

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("T", [(4), (6), (100), (320), (500), (8192)])
@pytest.mark.parametrize("QH_per_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 8, 16, 16])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.NEOX, RotateStyle.GPTJ])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("pos, offs", [(True, False), (True, True)])
def test_rope_cached_thd_2c_fwd(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    rotate_style: RotateStyle,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    dtype: torch.dtype,
    pos: bool,
    offs: bool,
    inplace: bool,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    torch_out_x = ref_rope_sbhd_fwd(
        x.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    ).squeeze(0)
    torch_out_y = ref_rope_sbhd_fwd(
        y.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    ).squeeze(0)

    if DEBUG_MODE:
        print(f"torch_out_x={torch_out_x}")
        print(f"torch_out_y={torch_out_y}")

    if offs:
        if inplace:
            triton_out_x, triton_out_y = (
                rope_cached_thd_positions_offsets_2c_fwd_inplace(
                    x,
                    y,
                    cos,
                    sin,
                    positions,
                    offsets,
                    rotate_style=rotate_style,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    nope_first=nope_first,
                    transpose_output=False,
                )
            )
        else:
            triton_out_x, triton_out_y = rope_cached_thd_positions_offsets_2c_fwd(
                x,
                y,
                cos,
                sin,
                positions,
                offsets,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )
    else:
        if inplace:
            triton_out_x, triton_out_y = rope_cached_thd_positions_2c_fwd_inplace(
                x,
                y,
                cos,
                sin,
                positions,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )
        else:
            triton_out_x, triton_out_y = rope_cached_thd_positions_2c_fwd(
                x,
                y,
                cos,
                sin,
                positions,
                rotate_style=rotate_style,
                reuse_freqs_front_part=reuse_freqs_front_part,
                nope_first=nope_first,
                transpose_output=False,
            )

    if DEBUG_MODE:
        print(f"triton_out_x={triton_out_x}")
        print(f"triton_out_y={triton_out_y}")

    torch.testing.assert_close(triton_out_x, torch_out_x, atol=1e-3, rtol=1e-1)
    torch.testing.assert_close(triton_out_y, torch_out_y, atol=1e-3, rtol=1e-1)


@pytest.mark.parametrize("T", [(4), (6), (100), (320), (500), (8192)])
@pytest.mark.parametrize("QH_per_KH", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("KH", [1, 8, 16, 16])
@pytest.mark.parametrize("D", [4, 64, 128])  # For now, D is power of 2.
@pytest.mark.parametrize("rotate_style", [RotateStyle.NEOX, RotateStyle.GPTJ])
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("pos, offs", [(True, False), (True, True)])
def test_rope_cached_thd_2c_bwd(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    rotate_style: RotateStyle,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    dtype: torch.dtype,
    pos: bool,
    offs: bool,
):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=nope,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
        bwd=True,
    )

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    if offs:
        triton_out_x, triton_out_y = rope_cached_thd_positions_offsets_2c_bwd(
            gx,
            gy,
            cos,
            sin,
            positions,
            offsets,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    else:
        triton_out_x, triton_out_y = rope_cached_thd_positions_2c_bwd(
            gx,
            gy,
            cos,
            sin,
            positions,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )

    torch_fwd_x = ref_rope_sbhd_fwd(
        x.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    ).squeeze(0)
    torch_fwd_y = ref_rope_sbhd_fwd(
        y.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=nope_first,
    ).squeeze(0)
    torch_fwd_x.backward(gx)
    torch_fwd_y.backward(gy)
    torch_out_x = x.grad
    torch_out_y = y.grad

    if DEBUG_MODE:
        print(f"torch_out_x={torch_out_x}")
        print(f"torch_out_y={torch_out_y}")

    if DEBUG_MODE:
        print(f"triton_out_x={triton_out_x}")
        print(f"triton_out_y={triton_out_y}")

    torch.testing.assert_close(triton_out_x, torch_out_x, atol=1e-3, rtol=1e-1)
    torch.testing.assert_close(triton_out_y, torch_out_y, atol=1e-3, rtol=1e-1)


@pytest.mark.parametrize("B", [1, 2, 15, 32, 57])
@pytest.mark.parametrize("H", [1, 8, 32])
@pytest.mark.parametrize("D", [4, 128])  # TODO 256 with height/width =64 is too slow.
@pytest.mark.parametrize("height, width", [(32, 32), (64, 32), (32, 64)])
@pytest.mark.parametrize("margin", [0])
@pytest.mark.parametrize(
    "rotate_style", [RotateStyle.NEOX]
)  # GPTJ case us off in CK/HIP test case
@pytest.mark.parametrize(
    "nope, nope_first", [(False, False)]
)  # Other cases are off in CK/HIP test case
@pytest.mark.parametrize(
    "reuse_freqs_front_part", [False]
)  # Other cases are off in CK/HIP test case
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16]) #TODO bf16 results in accuracy issues
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("inplace", [True, False])
def test_rope_2d_fwd(
    B: int,
    H: int,
    D: int,
    height: int,
    width: int,
    margin: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope: bool,
    nope_first: bool,
    inplace: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(20)

    x = torch.randn((B, height * width, H, D), dtype=dtype, device="cuda")

    freqs_h = torch.randn((1, height + margin, 1, D // 2), dtype=dtype, device="cuda")
    freqs_w = torch.randn((1, width + margin, 1, D // 2), dtype=dtype, device="cuda")

    cos_h = torch.cos(freqs_h)  # [1, height, 1, d // 2]
    sin_h = torch.sin(freqs_h)  # [1, height, 1, d // 2]
    cos_w = torch.cos(freqs_w)  # [1, width, 1, d // 2]
    sin_w = torch.sin(freqs_w)  # [1, width, 1, d // 2]

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")
        print(
            f"freqs_h.shape={freqs_h.shape} freqs_h.strides={freqs_h.stride()} freqs_h={freqs_h}"
        )
        print(
            f"freqs_w.shape={freqs_w.shape} freqs_w.strides={freqs_w.stride()} freqs_w={freqs_w}"
        )
        print(f"cos_h.shape={cos_h.shape} cos_h.strides={cos_h.stride()} cos_h={cos_h}")
        print(f"sin_h.shape={sin_h.shape} sin_h.strides={sin_h.stride()} sin_h={sin_h}")
        print(f"cos_w.shape={cos_w.shape} cos_w.strides={cos_w.stride()} cos_w={cos_w}")
        print(f"sin_w.shape={sin_w.shape} sin_w.strides={sin_w.stride()} sin_w={sin_w}")

    torch_out = ref_rope_2d_fwd(
        x, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style=rotate_style
    )
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    if inplace:
        triton_out = rope_fwd_2d_inplace(
            x,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            img_height=height,
            img_width=width,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    else:
        triton_out = rope_fwd_2d(
            x,
            cos_h,
            sin_h,
            cos_w,
            sin_w,
            img_height=height,
            img_width=width,
            rotate_style=rotate_style,
            reuse_freqs_front_part=reuse_freqs_front_part,
            nope_first=nope_first,
            transpose_output=False,
        )
    if DEBUG_MODE:
        print(f"triton_out={triton_out}")

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
