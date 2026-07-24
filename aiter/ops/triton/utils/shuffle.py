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


def moe_weight_decode_view(w: torch.Tensor) -> torch.Tensor:
    """zero-copy fn
    Input: ``(E, N, K)`` (K byte-packed for mxfp4). Output shares storage.
    """
    w_u8 = w if w.dtype == torch.uint8 else w.view(torch.uint8)
    E, N, K = w_u8.shape
    assert N % 16 == 0, f"N={N} must be divisible by 16"
    return w_u8.view(E, N // 16, K * 16).transpose(-1, -2)


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


def shuffle_scale_gemm_e8m0(scales: torch.Tensor, arch="gfx950") -> torch.Tensor:
    """MXFP4 e8m0 block-scale preshuffle, built on the ``shuffle_scale_gemm``
    reshape/permute structure. Drop-in for ``aiter.ops.shuffle.shuffle_scale``
    (non-``guinterleave``) / ``fp4_utils.e8m0_shuffle``.

    Arch-aware.

    * gfx950:  ``preshuffle_factor = 32``, ``scale_kwidth = 8`` -- byte-identical
      to ``fp4_utils.e8m0_shuffle`` (row-``(2,16)`` / K-``(2,4)``).
    * gfx1250: ``preshuffle_factor = 16``, ``scale_kwidth = 4`` -- the gfx1250
      WMMA GEMM scale tile.

    The column padding is a multiple of ``scale_kwidth`` (8 on gfx950; 4 on gfx1250). 
    The row padding is a multiple of 256, which is
    the reference gfx950 M-tile and is divisible by both preshuffle factors, so
    the tile reshape is valid on either arch. Returns the padded
    ``(M_pad, K_groups_pad)`` view; the consumer (``gemm_a4w4_quant``) re-collapses
    it by ``preshuffle_factor``, so load-time tile and forward collapse now agree.
    """
    if scales is None:
        return scales
    if scales.dtype == torch.float32:
        return scales
    assert scales.ndim == 2, "scale must be a 2D tensor"

    arch = arch or get_arch()

    if arch == "gfx1250":
        preshuffle_factor, scale_kwidth = 16, 4
    else:
        preshuffle_factor, scale_kwidth = 32, 8

    m, n = scales.shape
    m_pad = (m + 255) // 256 * 256
    n_pad = (n + scale_kwidth - 1) // scale_kwidth * scale_kwidth
    padded = torch.empty((m_pad, n_pad), dtype=scales.dtype, device=scales.device)
    padded[:m, :n] = scales

    tiled = shuffle_scale_gemm(
        padded,
        arch=arch,
        preshuffle_factor=preshuffle_factor,
        scale_kwidth=scale_kwidth,
    )
    return tiled.reshape(m_pad, n_pad)


def mxfp4_scale_preshuffle_factor(arch=None) -> int:
    """Per-arch e8m0 scale preshuffle (collapse) factor for the MXFP4 preshuffle
    GEMM: 16 on gfx1250 (WMMA tiles both M and N in 16-lane groups), 32 elsewhere
    (gfx950 CDNA4). Applies to BOTH the activation (M-axis) and weight (N-axis)
    scales, and must match the tile emitted by ``shuffle_scale_gemm_e8m0``.
    """
    return 16 if (arch or get_arch()) in ("gfx1250",) else 32


def collapse_mxfp4_gemm_scale(scale, block_size, rows_valid=None, arch=None):
    """Re-collapse an un-collapsed ``(rows_pad, kgroups_pad)`` e8m0 scale (as
    produced by ``shuffle_scale_gemm_e8m0``) into the packed layout the preshuffle
    GEMM consumes, by the arch preshuffle factor (``mxfp4_scale_preshuffle_factor``).

    ``rows_valid`` (the activation M): when provided and smaller than the MXFP4
    block size, the padded rows are sliced to ``[:rows_valid]`` rather than
    collapsed (the small-M activation case). Weight scales pass ``rows_valid=None``
    to always collapse.
    """
    scale = scale.view(torch.uint8)
    if rows_valid is not None and rows_valid < block_size:
        return scale[:rows_valid, ...]
    pf = mxfp4_scale_preshuffle_factor(arch)
    return scale.view(scale.shape[0] // pf, -1)


def quant_mxfp4_act_preshuffle(x, params_dtype, m, block_size, arch=None):
    """Arch-aware online MXFP4 activation quant for the preshuffle GEMM path.

    Returns ``(x, x_scale)`` with ``x_scale`` already collapsed by the arch
    preshuffle factor, ready for ``gemm_afp4wfp4_preshuffle``.

    ``get_hip_quant``'s built-in scale shuffle is gfx950-pinned
    (``per_1x32_mx_quant_hip`` pads to the 32/8 tile, no arch branch). On gfx1250
    we quantize WITHOUT that shuffle and apply the arch-aware e8m0 tile (16/4) via
    ``shuffle_scale_gemm_e8m0``, so the activation scale matches the gfx1250 GEMM
    instead of arriving in the gfx950 layout.
    """
    from aiter import QuantType, get_hip_quant

    arch = arch or get_arch()
    quant_func = get_hip_quant(QuantType.per_1x32)
    if arch in ("gfx1250",):
        # shuffle is pinned at gfx950, quant then shuffle manually
        x, x_scale = quant_func(x, quant_dtype=params_dtype, shuffle=False)
        if m >= block_size:
            x_scale = shuffle_scale_gemm_e8m0(x_scale.view(torch.uint8), arch=arch)
    else:
        x, x_scale = quant_func(
            x,
            quant_dtype=params_dtype,
            shuffle=(m >= block_size),
        )
    return x, collapse_mxfp4_gemm_scale(x_scale, block_size, rows_valid=m, arch=arch)


# --- MoE MX scales (a8w4 / a8w8 / a16w4 / a4w4) ---
def shuffle_scale_moe(
    data: torch.Tensor,
    arch=None,
    preshuffle_factor: int = 32,
    scale_kwidth: int = 8,
    return_layout: bool = False,
):
    """Arch-aware MoE scale shuffle (a8w4 / a8w8 / a16w4 / a4w4 family).

    gfx950 / gfx1250: preshuffle_factor = 32, scale_kwidth = 8.

    With ``return_layout=True`` also returns the matching ``SWIZZLE_MX_SCALE``
    label ("GFX1250_SCALE" for gfx1250, "CDNA4_SCALE" for gfx950) as
    ``(scale, label)``, so callers stay arch-agnostic; otherwise returns just
    the shuffled scale tensor.
    """
    arch = arch or get_arch()
    layout = None
    if arch == "gfx1250":
        tiled = _shuffle_scale_tile_gfx1250(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "GFX1250_SCALE"
    elif (arch or get_arch()) == "gfx950":
        tiled = _shuffle_scale_tile_gfx950(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "CDNA4_SCALE"
    scale = tiled.transpose(-1, -2)
    return (scale, layout) if return_layout else scale


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
