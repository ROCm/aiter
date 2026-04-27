# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch


def _get_compiled(fn):
    return torch.compile(fn, backend="inductor", options={"max_autotune": True})


def generate_cos_sin(seq_len, dim, device, dtype):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def torch_rope_neox(x, cos, sin):
    dim = x.shape[-1]
    half = dim // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_part = cos[:, :half].unsqueeze(1).unsqueeze(1)
    sin_part = sin[:, :half].unsqueeze(1).unsqueeze(1)
    out1 = x1 * cos_part - x2 * sin_part
    out2 = x2 * cos_part + x1 * sin_part
    return torch.cat([out1, out2], dim=-1)


def torch_rope_gptj(x, cos, sin):
    dim = x.shape[-1]
    half = dim // 2
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_part = cos[:, :half].unsqueeze(1).unsqueeze(1)
    sin_part = sin[:, :half].unsqueeze(1).unsqueeze(1)
    out1 = x1 * cos_part - x2 * sin_part
    out2 = x1 * sin_part + x2 * cos_part
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rotate_style", ["neox", "gptj"])
@pytest.mark.parametrize(
    "S, B, H, D", [(32, 2, 8, 64), (64, 4, 16, 128)]
)
def test_compile_rope(S, B, H, D, dtype, rotate_style):
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    device = "cuda"
    x = torch.randn(S, B, H, D, device=device, dtype=dtype)
    cos, sin = generate_cos_sin(S, D, device, dtype)

    rope_fn = torch_rope_neox if rotate_style == "neox" else torch_rope_gptj
    out_eager = rope_fn(x, cos, sin)

    def fn(x, cos, sin):
        return rope_fn(x, cos, sin)

    compiled_fn = _get_compiled(fn)
    out_compiled = compiled_fn(x, cos, sin)
    torch.cuda.synchronize()

    assert not torch.isnan(out_compiled).any(), "torch.compile produced NaN"
    torch.testing.assert_close(out_compiled, out_eager, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
