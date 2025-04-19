import os
import sys
import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_rope import ref_rope_sbhd_fwd, RotateStyle
from aiter.ops.triton.rope import rope_fwd, rope_fwd_cached

DEBUG_MODE = True

@pytest.mark.parametrize('B', [1, 2, 15, 32, 57])
@pytest.mark.parametrize('S', [2, 10, 32])
@pytest.mark.parametrize('H', [1, 8, 32])
@pytest.mark.parametrize('D', [4, 128, 256])  #For now, D is power of 2. 
@pytest.mark.parametrize('rotate_style', [RotateStyle.GPTJ , RotateStyle.NEOX])
@pytest.mark.parametrize('nope, nope_first', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('reuse_freqs_front_part', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_rope_fwd(B: int, S: int, H: int, D: int, rotate_style: int, reuse_freqs_front_part: bool, nope: bool, nope_first: bool, dtype: torch.dtype):

    torch.manual_seed(20)

    x = torch.randn((S, B, H, D), dtype=dtype, device="cuda")

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2

    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")
    triton_out = rope_fwd(x, freqs, rotate_style=rotate_style, reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first, transpose_output=False)

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch_out = ref_rope_sbhd_fwd(x, freqs, rotate_style=rotate_style, reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first)
    
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    torch.testing.assert_close(triton_out, torch_out,atol=1e-1, rtol=1e-1)

@pytest.mark.parametrize('B', [1, 2, 15, 32, 57])
@pytest.mark.parametrize('S', [2, 10, 32])
@pytest.mark.parametrize('H', [1, 8, 32])
@pytest.mark.parametrize('D', [4, 128, 256])  #For now, D is power of 2. 
@pytest.mark.parametrize('rotate_style', [RotateStyle.GPTJ , RotateStyle.NEOX])
@pytest.mark.parametrize('nope, nope_first', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('reuse_freqs_front_part', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_rope_fwd_cached(B: int, S: int, H: int, D: int, rotate_style: int, reuse_freqs_front_part: bool, nope: bool, nope_first: bool, dtype: torch.dtype):
    torch.manual_seed(20)

    x = torch.randn((S, B, H, D), dtype=dtype, device="cuda")

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2

    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    if DEBUG_MODE:
        print(f"x.shape={x.shape} x={x}")
        print(f"freqs.shape={freqs.shape} freqs.strides={freqs.stride()} freqs={freqs}")
    triton_out = rope_fwd_cached(x, cos, sin, rotate_style=rotate_style, reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first, transpose_output=False)

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
    torch_out = ref_rope_sbhd_fwd(x, freqs, rotate_style=rotate_style, reuse_freqs_front_part=reuse_freqs_front_part, nope_first=nope_first)
    
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")

    torch.testing.assert_close(triton_out, torch_out,atol=1e-1, rtol=1e-1)