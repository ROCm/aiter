import torch

from aiter.ops.shuffle import shuffle_weight as _shuffle_weight_base
from aiter.ops.triton.utils._triton.arch_info import get_arch

# =============================================================================
# WEIGHTS
# =============================================================================


def _shuffle_weight_gfx1250(w: torch.Tensor) -> torch.Tensor:
    """gfx1250 WMMA weight preshuffle
    Callers wanting the flattened (N//16, K*16) / transposed (E, K*16, N//16) TDM
    view reshape it.
    """
    x_type = w.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        w = w.view(torch.uint8)
    if w.ndim == 2:
        N, K = w.shape
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        w = w.view(N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 2, 3, 1, 4).contiguous()
        w = w.view(N, K)
    elif w.ndim == 3:
        E, K, N = w.shape
        assert K % 32 == 0, f"K={K} must be divisible by 32"
        assert N % 16 == 0, f"N={N} must be divisible by 16"
        w = w.transpose(-1, -2)  # (E, N, K)
        w = w.view(E, N // 16, 16, K // 32, 2, 16)
        w = w.permute(0, 1, 3, 4, 2, 5).contiguous()
        w = w.view(E, K, N)
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {w.ndim}D")
    w = w.view(x_type)
    w.is_shuffled = True
    return w


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 16),
    use_int4=False,
    is_guinterleave=False,
    gate_up: bool = False,
    pad_k_to: int = 0,
    arch=None,
) -> torch.Tensor:
    """Arch-aware weight preshuffle.

    On gfx1250 the WMMA TDM layout (``_shuffle_weight_gfx1250``) is used; on every
    other arch this delegates to the base ``aiter.ops.shuffle.shuffle_weight``.
    """
    if (arch or get_arch()) == "gfx1250":
        if use_int4 or is_guinterleave or gate_up or pad_k_to:
            raise NotImplementedError(
                "shuffle_weight on gfx1250 does not support use_int4 / is_guinterleave / gate_up / pad_k_to "
            )
        return _shuffle_weight_gfx1250(x)

    return _shuffle_weight_base(
        x,
        layout=layout,
        use_int4=use_int4,
        is_guinterleave=is_guinterleave,
        gate_up=gate_up,
        pad_k_to=pad_k_to,
    )


# =============================================================================
# SCALES
# =============================================================================


# --- shared gfx1250 scale tile (GEMM + MoE) ---
def _shuffle_scale_tile_gfx1250(scales, preshuffle_factor, scale_kwidth):
    """Shared gfx1250 scale tile-permute over the last two dims.

    row = the output M/N axis, packed into stripes of ``preshuffle_factor`` lanes
    col = the scale-K axis (K_groups / K_SCALE), packed into ``scale_kwidth`` groups

    Shared by the GEMM ((M, K_groups)) and MoE ((E, N, K_SCALE), transposed) gfx1250 scale shuffles.
    """
    # rows and cols grab the last two dims, and *batch collects everything before them into a list (possibly empty)
    *batch, rows, cols = scales.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scales.reshape(-1, rows, cols)  # fold batch/expert dims into one axis
    x = x.view(-1, num_stripes, preshuffle_factor, num_kchunks, scale_kwidth)
    x = x.permute(0, 1, 3, 2, 4).contiguous()  # swap lanes <-> k-chunks
    out = x.view(-1, num_stripes, cols * preshuffle_factor)
    return out.reshape(*batch, num_stripes, cols * preshuffle_factor)


# --- shared gfx950 scale tile (GEMM + MoE) ---
def _shuffle_scale_tile_gfx950(scales, preshuffle_factor, scale_kwidth):
    """Shared gfx950 (CDNA4) scale tile-permute over the last two dims.

    row = the output M/N axis, packed into stripes of ``preshuffle_factor`` lanes (split 2 x preshuffle_factor//2)
    col = the scale-K axis (K_groups / K_SCALE), packed into ``scale_kwidth`` groups (split 2 x scale_kwidth//2)

    Shared by the GEMM ((M, K_groups)) and MoE ((E, N, K_SCALE), transposed) gfx950 scale shuffles.
    """
    # rows and cols grab the last two dims, and *batch collects everything before them into a list (possibly empty)
    *batch, rows, cols = scales.shape
    num_stripes = rows // preshuffle_factor
    num_kchunks = cols // scale_kwidth
    x = scales.reshape(-1, rows, cols)  # fold batch/expert dims into one axis
    x = x.view(
        -1, num_stripes, 2, preshuffle_factor // 2, num_kchunks, 2, scale_kwidth // 2, 1
    )
    x = x.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    out = x.view(-1, num_stripes, cols * preshuffle_factor)
    return out.reshape(*batch, num_stripes, cols * preshuffle_factor)


# --- GEMM scales (afp4wfp4) ---


def shuffle_scale_gemm(
    scales: torch.Tensor,
    arch=None,
    preshuffle_factor: int = 16,
    scale_kwidth: int = 4,
) -> torch.Tensor:
    """Arch-aware GEMM scale shuffle.

    Inverse: ``unshuffle_scale_gemm`` (gfx950 only).
    gfx950: preshuffle_factor = 32, scale_kwidth = 8
    gfx1250: preshuffle_factor = 16, scale_kwidth = 4
    """
    if (arch or get_arch()) == "gfx1250":
        return _shuffle_scale_tile_gfx1250(scales, preshuffle_factor, scale_kwidth)

    if (arch or get_arch()) == "gfx950":
        return _shuffle_scale_tile_gfx950(scales, preshuffle_factor, scale_kwidth)
    raise ValueError(f"Unsupported arch: {arch or get_arch()}")


def unshuffle_scale_gemm(scales_shuffled: torch.Tensor, arch=None) -> torch.Tensor:
    """Inverse of ``shuffle_scale_gemm`` (gfx950 layout). gfx1250 has no consumer."""
    if (arch or get_arch()) == "gfx1250":
        raise NotImplementedError("unshuffle_scale_gemm is not implemented for gfx1250")
    scales = scales_shuffled.clone()
    sm, sn = scales.shape
    scales = scales.view(sm * 32, sn // 32)
    sm, sn = scales.shape
    scales = scales.view(sm // 32, sn // 8, 4, 16, 2, 2, 1)
    scales = scales.permute(0, 5, 3, 1, 4, 2, 6).contiguous()
    scales = scales.view(sm, sn)
    return scales


# --- MoE MX scales (a8w4 / a8w8 / a16w4 / a4w4) ---
def shuffle_scale_moe(
    data: torch.Tensor,
    arch=None,
    preshuffle_factor: int = 32,
    scale_kwidth: int = 8,
) -> torch.Tensor:
    """Arch-aware MoE scale shuffle (a8w4 / a8w8 / a16w4 / a4w4 family).

    Returns the shuffled scale tensor; the caller supplies the matching
    ``SWIZZLE_MX_SCALE`` label ("GFX1250_SCALE" for gfx1250, "CDNA4_SCALE" for gfx950).
    gfx950: preshuffle_factor = 32, scale_kwidth = 8
    gfx1250: preshuffle_factor = 32, scale_kwidth = 8
    """

    arch = arch or get_arch()
    if arch == "gfx1250":
        tiled = _shuffle_scale_tile_gfx1250(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        return tiled.transpose(-1, -2)
    if (arch or get_arch()) == "gfx950":
        tiled = _shuffle_scale_tile_gfx950(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
    return tiled.transpose(-1, -2)


# --- batched scales (FP4 blockscale16, attention) ---
def shuffle_scale_batched(data: torch.Tensor, scale_k_width=None) -> torch.Tensor:
    """Batched shuffle scales for the FP4 blockscale16 format.

    Single-layout permute, no arch branch: the blockscale16 layout is
    arch-independent and is consumed by the FP4 MLA KV-cache path on both gfx950
    and gfx1250
    https://github.com/triton-lang/triton/blob/main/third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py#L1014
    """
    data_shape = data.shape
    N = data_shape[-2]
    SCALE_K = data_shape[-1]
    PRESHUFFLE_FACTOR = 128
    if scale_k_width is None:
        SCALE_KWIDTH = (
            min(16, 1 << (SCALE_K - 1).bit_length()) if SCALE_K >= 4 else SCALE_K
        )
    else:
        assert scale_k_width in [4, 8, 16]
        SCALE_KWIDTH = scale_k_width if SCALE_K >= 4 else SCALE_K
    data = data.view(
        -1,
        N // PRESHUFFLE_FACTOR,
        4,
        PRESHUFFLE_FACTOR // 4,
        SCALE_K // SCALE_KWIDTH,
        SCALE_KWIDTH,
    )
    data = data.permute(0, 1, 4, 3, 2, 5).contiguous()
    data = data.view(
        *data_shape[:-2], N // PRESHUFFLE_FACTOR, SCALE_K * PRESHUFFLE_FACTOR
    )
    return data
