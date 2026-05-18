"""Test fp8_einsum('bhr,hdr->bhd') in fp32 scale mode against a DeepGEMM-
equivalent reference.

The kernel takes bf16 X and pre-quantized fp8 Y + fp32 per-(128,128) block
scales; it quantizes X online per-row over each 128-K chunk. The reference
replicates exactly that contract:

  sx  = (x_amax_per_(B*H, 128-K-block) / 448).clamp_min(1e-4)
  x_q = (x / sx).to(fp8_e4m3)
  sy, y_q = per_block_cast_to_fp8(y, block=(128,128))
  z_ref  = einsum_fp32_path(x_q, sx, y_q, sy)
  assert calc_diff(z_kernel, z_ref) < 1e-3

This test only runs on gfx950 hardware (the kernel guards against other
archs); on other GPUs it is skipped.
"""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight


def _import_kernel():
    from aiter.ops.flydsl.kernels.fp8_einsum import (  # noqa: WPS433
        compile_fp8_einsum_bhr_hdr_bhd,
    )

    return compile_fp8_einsum_bhr_hdr_bhd


def per_token_cast_to_fp8_x(x_bf16: torch.Tensor):
    """Per-(B,H, K=128-block) quant of X. Matches kernel's online quant."""
    B, H, R = x_bf16.shape
    BR = 128
    assert R % BR == 0
    x_f32 = x_bf16.float()
    x_blk = x_f32.view(B, H, R // BR, BR)
    amax = x_blk.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 448.0).clamp_min(1e-4)
    x_q = (x_blk / scale).to(torch.float8_e4m3fn)
    x_q = x_q.view(B, H, R)
    scale = scale.view(B, H, R // BR)
    return x_q, scale


def per_block_cast_to_fp8_y(y_bf16: torch.Tensor):
    """Per-(128,128) block quant of Y over (D, R) within each head."""
    H, D, R = y_bf16.shape
    BD, BR = 128, 128
    assert D % BD == 0 and R % BR == 0
    y_f32 = y_bf16.float()
    y_blk = y_f32.view(H, D // BD, BD, R // BR, BR)
    amax = y_blk.abs().amax(dim=(2, 4), keepdim=True)
    scale = (amax / 448.0).clamp_min(1e-4)
    y_q = (y_blk / scale).to(torch.float8_e4m3fn)
    y_q = y_q.view(H, D, R)
    scale = scale.view(H, D // BD, R // BR)
    return y_q, scale


def einsum_fp32_path(
    x_fp8: torch.Tensor,
    sx: torch.Tensor,
    y_fp8: torch.Tensor,
    sy: torch.Tensor,
) -> torch.Tensor:
    """Bit-aligned to kernel's scratch-then-promote scheme.

    For each K=128 block:
      local = sum_r (x_fp8[b,h,r].f32 * y_fp8[h,d,r].f32)   # 128 fma's
      z    += local * sx[b,h,k] * sy[h, d//128, k]
    """
    B, H, R = x_fp8.shape
    _, D, _ = y_fp8.shape
    BR = 128
    K128 = R // BR
    z = torch.zeros((B, H, D), dtype=torch.float32, device=x_fp8.device)
    for k in range(K128):
        x_blk = x_fp8[:, :, k * BR : (k + 1) * BR].float()  # (B,H,128)
        y_blk = y_fp8[:, :, k * BR : (k + 1) * BR].float()  # (H,D,128)
        local = torch.einsum("bhr,hdr->bhd", x_blk, y_blk)  # fp32
        sx_blk = sx[:, :, k].unsqueeze(-1)                  # (B,H,1)
        sy_blk = sy[:, :, k].repeat_interleave(BR, dim=-1)  # (H, D)
        z = z + local * sx_blk * sy_blk.unsqueeze(0)
    return z


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """DeepGEMM-style 1 - cosine_similarity."""
    af = a.flatten().double()
    bf = b.flatten().double()
    denom = af.pow(2).sum().sqrt() * bf.pow(2).sum().sqrt()
    return float(1.0 - (af * bf).sum() / denom)


def preshuffle_y(y_HDR_fp8: torch.Tensor) -> torch.Tensor:
    """Reshape Y[h,:,:] into (n0,k0,4,16,16) layout, per head.

    shuffle_weight(layout=(16, 32)) over a (-1, D, R) input gives BN=16,
    BK=64 fp8 = matches the kernel's preshuffle convention.
    """
    return shuffle_weight(y_HDR_fp8, layout=(16, 32))


@pytest.mark.parametrize(
    "H,D,R,B",
    [
        (8, 1024, 4096, 4),
        (8, 1024, 4096, 32),
        (8, 1024, 4096, 128),
    ],
)
@pytest.mark.parametrize("tile_m,tile_n,tile_k", [(64, 128, 128)])
def test_fp8_einsum_bhr_hdr_bhd_fp32_mode(
    H: int, D: int, R: int, B: int,
    tile_m: int, tile_n: int, tile_k: int,
):
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    if get_gfx() != "gfx950":
        pytest.skip(f"fp8_einsum kernel requires gfx950, got {get_gfx()}")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn unavailable")

    torch.manual_seed(0)
    device = torch.device("cuda")

    # ── Inputs ────────────────────────────────────────────────────────────────
    x_bf16 = torch.randn(B, H, R, dtype=torch.bfloat16, device=device) * 4.0
    y_bf16 = torch.randn(H, D, R, dtype=torch.bfloat16, device=device) * 0.5

    # Pre-quantize Y per (128, 128) block — matches kernel sy contract.
    y_fp8, sy_fp32 = per_block_cast_to_fp8_y(y_bf16)
    sy_fp32 = sy_fp32.to(torch.float32).contiguous()

    # Preshuffle Y per head into (n0, k0, 4, 16, 16).
    y_pre = preshuffle_y(y_fp8).contiguous()

    # Output buffer.
    z_out = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)

    # ── Compile + launch kernel ───────────────────────────────────────────────
    compile_fn = _import_kernel()
    kernel = compile_fn(
        H=H, D=D, R=R,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_mode="fp32",
    )
    stream = torch.cuda.current_stream()
    kernel(z_out, x_bf16, y_pre, sy_fp32, B, stream)
    torch.cuda.synchronize()

    # ── Reference ─────────────────────────────────────────────────────────────
    # Reuse the same Y quant the kernel sees; quantize X the same way the kernel
    # does online (so both paths share the FP8 lossy step and only the GEMM
    # math is being compared).
    x_fp8, sx_fp32 = per_token_cast_to_fp8_x(x_bf16)
    z_ref_fp32 = einsum_fp32_path(x_fp8, sx_fp32, y_fp8, sy_fp32)
    z_ref_bf16 = z_ref_fp32.to(torch.bfloat16)

    diff = calc_diff(z_out, z_ref_bf16)
    assert diff < 1e-3, f"calc_diff={diff:.6f} above tolerance"


@pytest.mark.parametrize("H,D,R", [(8, 1024, 4096)])
def test_fp8_einsum_ue8m0_mode_not_implemented(H: int, D: int, R: int):
    """UE8M0 mode is currently stubbed; this guards against accidental enable."""
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    compile_fn = _import_kernel()
    with pytest.raises(NotImplementedError):
        compile_fn(
            H=H, D=D, R=R,
            tile_m=64, tile_n=128, tile_k=128,
            scale_mode="ue8m0",
        )


@pytest.mark.parametrize(
    "H,D,R,tile_m,tile_n,tile_k",
    [
        (8, 1024, 4096, 64, 128, 128),
        (8, 1024, 4096, 64, 128, 256),
        (8, 1024, 4096, 128, 128, 128),
        (8, 1024, 4096, 32, 128, 128),
    ],
)
def test_fp8_einsum_compile_trace(
    H: int, D: int, R: int, tile_m: int, tile_n: int, tile_k: int,
):
    """Pure MLIR compile trace — runs on any host (no GPU needed for IR build).

    This catches IR-level regressions without requiring gfx950 hardware. We
    monkey-patch the arch guard since this test does not launch the kernel.
    """
    if not is_flydsl_available():
        pytest.skip("FlyDSL is not available")
    from aiter.ops.flydsl.kernels import fp8_einsum as fp8_einsum_mod

    orig = fp8_einsum_mod.get_hip_arch
    fp8_einsum_mod.get_hip_arch = lambda: "gfx950"
    try:
        kernel = fp8_einsum_mod.compile_fp8_einsum_bhr_hdr_bhd(
            H=H, D=D, R=R,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            scale_mode="fp32",
        )
    finally:
        fp8_einsum_mod.get_hip_arch = orig
    assert kernel is not None
