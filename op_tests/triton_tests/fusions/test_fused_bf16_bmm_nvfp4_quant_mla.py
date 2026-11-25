# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.fusions.fused_bf16_bmm_nvfp4_quant_mla import (
    D_NOPE,
    KV_LORA,
    fused_bf16_bmm_nvfp4_quant_mla,
)
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in ("gfx1250", "gfx1250-strict"), reason="gfx1250 only"
)

HEADS = 128
Q_GAIN = 6.72


def generate_inputs(batch, heads=HEADS, q_width=D_NOPE, seed=0):
    """``q_nope`` (a view of ``(B, H, q_width)`` when q_width > 128), ``w_uk``
    and the two output planes. ``q_aux`` holds random bytes."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (batch, heads, q_width), dtype=torch.float32, device="cuda", generator=gen
    )
    # Outliers, zeros and rows of tiny values.
    big = torch.rand(q.shape, dtype=torch.float32, device="cuda", generator=gen) < 0.01
    q = torch.where(big, q * 50, q)
    q[torch.rand(q.shape, dtype=torch.float32, device="cuda", generator=gen) < 0.01] = (
        0.0
    )
    tiny = (
        torch.rand((batch, heads, 1), dtype=torch.float32, device="cuda", generator=gen)
        < 0.02
    )
    q = torch.where(tiny, q * 1e-33, q)
    # All-zero blocks (amax 0, ideal scale 1): zero rows zero every block of
    # their output, and a zero W_UK slice zeroes one block of every row.
    zero = (
        torch.rand((batch, heads, 1), dtype=torch.float32, device="cuda", generator=gen)
        < 0.02
    )
    q = torch.where(zero, torch.zeros_like(q), q)
    q_nope = q.to(torch.bfloat16)[..., :D_NOPE]
    w_uk = (
        torch.randn(
            (heads, KV_LORA, D_NOPE), dtype=torch.float32, device="cuda", generator=gen
        )
        * 0.05
    ).to(torch.bfloat16)
    w_uk[::7, 32:48] = 0
    q_nope_fp4 = torch.empty(
        (batch, heads, KV_LORA // 2), dtype=torch.uint8, device="cuda"
    )
    q_aux = torch.randint(
        0, 256, (batch, heads, 128), dtype=torch.uint8, device="cuda", generator=gen
    )
    return q_nope, w_uk, q_nope_fp4, q_aux


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_codes(x):
    """Nearest-even E2M1 codes of fp32 ``x`` (saturating at 6), sign in bit 3."""
    table = torch.tensor(_E2M1, dtype=torch.float32, device=x.device)
    a = x.abs().contiguous()
    hi = torch.searchsorted(table, a).clamp(max=len(_E2M1) - 1)
    lo = (hi - 1).clamp(min=0)
    da, db = a - table[lo], table[hi] - a
    code = torch.where((da < db) | ((da == db) & (lo % 2 == 0)), lo, hi)
    return code | (torch.signbit(x).to(torch.int64) << 3)


def quantize_reference(out):
    batch, heads, _ = out.shape
    x = out.float().view(batch, heads, KV_LORA // 16, 16)
    amax = x.abs().amax(dim=-1)
    gain = torch.full_like(amax, Q_GAIN)
    ideal = torch.where(
        amax == 0, torch.ones_like(amax), amax / torch.full_like(amax, 6.0)
    )
    s8 = (ideal * gain).clamp(2**-9, 448).to(torch.float8_e4m3fn)
    div = s8.float() / gain
    codes = _e2m1_codes(x / div[..., None]).view(batch, heads, KV_LORA // 2, 2)
    packed = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return packed, s8.view(torch.uint8)


def reference(q_nope, w_uk):
    """BF16 BMM per head, then 16-value blocks: scale E4M3(amax / 6 * gain),
    codes E2M1(x / (scale / gain))."""
    batch, heads, _ = q_nope.shape
    out = torch.empty((batch, heads, KV_LORA), dtype=torch.bfloat16, device="cuda")
    batched_gemm_bf16(q_nope.transpose(0, 1), w_uk, YQ=out.transpose(0, 1))
    return quantize_reference(out)


@pytest.mark.parametrize(
    "batch", [1, 37, 64, 65, 300, 1536, 2047, 2048, 2049, 2200, 2240, 2300]
)
@pytest.mark.parametrize("q_width", [D_NOPE, 192])
def test_fused_bf16_bmm_nvfp4_quant_mla(batch, q_width):
    q_nope, w_uk, q_nope_fp4, q_aux = generate_inputs(
        batch, q_width=q_width, seed=batch
    )
    stale = q_aux.clone()
    got_codes, got_aux = fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, q_nope_fp4, q_aux)
    want_codes, want_scales = reference(q_nope, w_uk)
    assert got_codes is q_nope_fp4 and got_aux is q_aux
    assert torch.equal(got_codes, want_codes), "codes differ"
    assert torch.equal(got_aux[..., 64:96], want_scales), "NoPE scales differ"
    assert torch.equal(got_aux[..., :64], stale[..., :64])
    assert torch.equal(got_aux[..., 96:], stale[..., 96:])
    # The block the zero W_UK slice leaves in every row: ideal scale 1, zero codes.
    unit = torch.tensor(Q_GAIN, dtype=torch.float32, device="cuda")
    unit = unit.to(torch.float8_e4m3fn).view(torch.uint8).item()
    assert (got_aux[:, ::7, 64 + 2] == unit).all()
    assert (got_codes[:, ::7, 16:24] == 0).all()


def test_allocates_outputs_and_empty_batch():
    q_nope, w_uk, _, _ = generate_inputs(37)
    codes, aux = fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk)
    assert torch.equal(codes, reference(q_nope, w_uk)[0])
    assert codes.device == q_nope.device and aux.shape == (37, HEADS, 128)
    q_nope, w_uk, codes, aux = generate_inputs(0)
    fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)
    q_nope, w_uk, codes, aux = generate_inputs(37, heads=0)
    fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_launches_on_the_input_device():
    with torch.cuda.device(1):
        q_nope, w_uk, codes, aux = generate_inputs(37)
        want = reference(q_nope, w_uk)[0]
    fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)
    torch.cuda.synchronize(1)
    assert torch.equal(codes, want)


def test_graph_replay():
    q_nope, w_uk, codes, aux = generate_inputs(300, seed=7)
    fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)
    want = codes.clone(), aux.clone()
    codes.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(codes, want[0]) and torch.equal(aux, want[1])


def test_rejects_bad_inputs():
    q_nope, w_uk, codes, aux = generate_inputs(4)
    with pytest.raises(AssertionError):
        fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk[:, :256], codes, aux)
    with pytest.raises(AssertionError):
        fused_bf16_bmm_nvfp4_quant_mla(q_nope.float(), w_uk, codes, aux)
    with pytest.raises(AssertionError):
        fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes[:, :, :128], aux)
    # Contiguous, but off a 16-byte boundary.
    raw = torch.empty(aux.numel() + 1, dtype=torch.uint8, device="cuda")
    with pytest.raises(AssertionError):
        fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, raw[1:].view(aux.shape))
    # Off q_nope's device.
    with pytest.raises(AssertionError, match="one GPU"):
        fused_bf16_bmm_nvfp4_quant_mla(
            q_nope, w_uk, codes, torch.empty_like(aux, device="cpu")
        )
    if torch.cuda.device_count() > 1:
        with pytest.raises(AssertionError, match="one GPU"):
            fused_bf16_bmm_nvfp4_quant_mla(
                q_nope, torch.empty_like(w_uk, device="cuda:1"), codes, aux
            )


def test_codes_saturate_past_the_scale_range():
    """Absorbed values of +-2**39 overflow the block scale: it saturates at 448
    (E4M3 0x7E) and every code at +-6.0 (7 and 15)."""
    q_nope = torch.full(
        (1, HEADS, D_NOPE), 65536.0, dtype=torch.bfloat16, device="cuda"
    )
    w_uk = torch.full(
        (HEADS, KV_LORA, D_NOPE), 65536.0, dtype=torch.bfloat16, device="cuda"
    )
    w_uk[:, 1::2] = -65536.0
    codes, aux = fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk)
    assert (aux[..., 64:96] == 0x7E).all()
    assert (codes == (7 | (15 << 4))).all()
    assert torch.equal(codes, reference(q_nope, w_uk)[0])


def test_runs_under_the_strict_arch_alias(monkeypatch):
    import aiter.ops.triton.fusions.fused_bf16_bmm_nvfp4_quant_mla as op
    from aiter.ops.triton.utils._triton import arch_info

    q_nope, w_uk, codes, aux = generate_inputs(37)
    want = reference(q_nope, w_uk)[0]
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250-strict")
    monkeypatch.setattr(op, "get_arch", lambda: "gfx1250-strict")
    fused_bf16_bmm_nvfp4_quant_mla(q_nope, w_uk, codes, aux)
    assert torch.equal(codes, want)


@pytest.mark.parametrize("batch", [37, 2049, 2300])
@pytest.mark.parametrize("heads", [1, 8, 128])
def test_exact_bmm_against_torch(batch, heads):
    # Dyadic inputs make the FP32 dot sums exact across reduction orders.
    gen = torch.Generator(device="cuda").manual_seed(batch + heads)
    q = torch.randint(-4, 5, (batch, heads, 192), device="cuda", generator=gen)
    q = q.to(torch.bfloat16) * (1 / 8)
    w = torch.randint(-4, 5, (heads, 512, 128), device="cuda", generator=gen)
    w = w.to(torch.bfloat16) * (1 / 64)
    want = torch.bmm(q[..., :128].transpose(0, 1).float(), w.transpose(1, 2).float())
    want = want.transpose(0, 1).to(torch.bfloat16)
    codes = torch.full((batch, heads, 256), 0xA5, device="cuda", dtype=torch.uint8)
    aux = torch.full((batch, heads, 128), 0xA5, device="cuda", dtype=torch.uint8)
    fused_bf16_bmm_nvfp4_quant_mla(q[..., :128], w, codes, aux)
    want_codes, want_scales = quantize_reference(want)
    assert torch.equal(codes, want_codes)
    assert torch.equal(aux[..., 64:96], want_scales)
    assert (aux[..., :64] == 0xA5).all() and (aux[..., 96:] == 0xA5).all()
