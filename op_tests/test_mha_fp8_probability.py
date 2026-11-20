# SPDX-License-Identifier: MIT
"""gfx942 FP8 attention must preserve probability mass when V is constant."""

import sys

import pytest
import torch


@pytest.fixture
def fp8_attention():
    if not torch.cuda.is_available():
        pytest.skip("Requires a gfx942 GPU")
    architecture = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if architecture.split(":")[0] != "gfx942":
        pytest.skip("Requires native gfx942 FNUZ FP8")
    # Delay the optional AITER import until after the CPU skip.
    mha = pytest.importorskip("aiter.ops.mha")
    if not mha.ENABLE_CK:
        pytest.skip("Requires ENABLE_CK=1 for the ASM dispatch")
    return mha.flash_attn_fp8_pertensor_func


def inputs(seqlen, score, max_position):
    shape = (1, seqlen, 24, 128)
    q = torch.zeros(shape, dtype=torch.float32, device="cuda")
    k = torch.zeros_like(q)
    if score == "spread":
        # Each FP8 input and resulting logit -7 - code/64 is exact.
        q[..., :7] = torch.tensor(
            [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625], device=q.device
        )
        codes = torch.linspace(0, 128, seqlen - 1, device=q.device).round().int()
        shifts = torch.arange(5, -1, -1, device=q.device)
        encoded_k = torch.cat(
            (-7 - codes[:, None] // 64, -((codes[:, None] >> shifts) & 1)), dim=1
        )
        start = 1 if max_position == "first" else 0
        k[:, start : start + seqlen - 1, :, :7] = encoded_k[:, None, :]
    else:
        q[..., 0] = 1
        k[..., 0] = score
        k[:, 0 if max_position == "first" else -1, :, 0] = 0
    dtype = torch.float8_e4m3fnuz
    return q.to(dtype), k.to(dtype)


def run_attention(function, q, k, v):
    descale = torch.ones((1,), dtype=torch.float32, device=q.device)
    return function(q, k, v, descale, descale, descale, softmax_scale=1.0, causal=False)


@pytest.mark.parametrize("seqlen", [4096, 4117])
@pytest.mark.parametrize("score", [0.0, -7.0, -8.0, "spread"])
@pytest.mark.parametrize("max_position", ["first", "last"])
def test_constant_v(fp8_attention, seqlen, score, max_position):
    q, k = inputs(seqlen, score, max_position)
    v = torch.ones(q.shape, device=q.device, dtype=torch.float32).to(q.dtype)
    # FP64 reference uses the actual, exactly representable input scores.
    logits = q[0, 0, 0].double() @ k[0, :, 0].double().T
    truth = torch.softmax(logits, dim=0).sum().item()
    # Identical tails round in the same direction in E4M3 even after scaling.
    # A 5% bound covers FP8 rounding across probability scales, including
    # spread tails; exactly uniform probabilities retain the 1% check.
    atol = 0.01 if score == 0.0 else 0.05
    for _ in range(5):
        output = run_attention(fp8_attention, q, k, v)
        assert torch.isfinite(output).all().item()
        torch.testing.assert_close(
            output.float(), torch.full_like(output.float(), truth), atol=atol, rtol=0
        )


def test_nonconstant_v(fp8_attention):
    seqlen = 4117
    q, k = inputs(seqlen, -7.0, "last")
    # Identical Q rows permit a direct FP64 reference without an S-by-S matrix.
    # The max key carries +2; background keys carry -1 or +0.5 by channel.
    values = torch.full(q.shape, -1.0, device=q.device, dtype=torch.float32)
    values[..., 1::2] = 0.5
    values[:, -1] = 2.0
    v = values.to(q.dtype)
    logits = q[0, 0, 0].double() @ k[0, :, 0].double().T
    truth = torch.softmax(logits, dim=0) @ v[0, :, 0].double()
    for _ in range(5):
        output = run_attention(fp8_attention, q, k, v)
        assert torch.isfinite(output).all().item()
        expected = truth.float().view(1, 1, 1, 128).expand_as(output)
        torch.testing.assert_close(output.float(), expected, atol=0.01, rtol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
