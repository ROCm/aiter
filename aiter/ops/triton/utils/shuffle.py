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


# --- int4 weights (gemm_a16w4) ---
# The column of each nibble in an AWQ int32 word: column c is in nibble _AWQ_NIBBLE[c].
_AWQ_NIBBLE = (0, 4, 1, 5, 2, 6, 3, 7)


def _unpack_int32_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 words (..., C) to uint8 values (..., 8 * C), nibble i of word c to 8 * c + i."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    values = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * 8).to(torch.uint8)


def pack_gemm_a16w4_weight(q: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values q (N, K), 0 to 15, to the gemm_a16w4 weight layout.

    Returns uint8 (N, K // 2). Byte j of row n holds q[n, 2j] in bits 0-3 and
    q[n, 2j + 1] in bits 4-7.
    """
    assert q.dim() == 2 and q.shape[1] % 2 == 0, "q must have shape (N, K), K even"
    q = q.to(torch.uint8) & 0xF
    return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()


def gptq_to_gemm_a16w4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    checkpoint_format: str = "gptq",
):
    """Convert 4-bit GPTQ checkpoint tensors to the gemm_a16w4 inputs.

    Args:
        qweight: int32 (K // 8, N). Nibble i of qweight[j, n] is q[8j + i, n].
        scales: (K // group_size, N).
        qzeros: int32 (K // group_size, N // 8), packed like qweight along N.
            None for symmetric checkpoints (zero point 8).
        g_idx: int32 (K,), the group of each row of K. Act-order (desc_act)
            checkpoints sort K by group. Each group must have group_size rows.
        checkpoint_format: "gptq" stores zero - 1 in qzeros, "gptq_v2" stores zero.

    Returns:
        (w, w_scales, w_zeros, x_perm) for gemm_a16w4. w_zeros is None when all
        zero points are 8. x_perm is None when K is not reordered.
    """
    assert checkpoint_format in ("gptq", "gptq_v2"), checkpoint_format
    K, N = qweight.shape[0] * 8, qweight.shape[1]
    num_groups = scales.shape[0]
    assert scales.shape == (num_groups, N) and K % num_groups == 0
    group_size = K // num_groups
    q = _unpack_int32_uint4(qweight.t().contiguous())

    x_perm = None
    if g_idx is not None:
        perm = torch.argsort(g_idx.to(torch.int64), stable=True)
        rows = torch.arange(K, device=perm.device)
        if not torch.equal(g_idx.to(torch.int64)[perm], rows // group_size):
            raise ValueError("each group in g_idx must have K // num_groups rows")
        if not torch.equal(perm, rows):
            q = q[:, perm]
            x_perm = perm.to(torch.int32)

    w_zeros = None
    if qzeros is not None:
        zeros = _unpack_int32_uint4(qzeros)
        if checkpoint_format == "gptq":
            zeros = (zeros + 1) & 0xF
        if not bool((zeros == 8).all()):
            w_zeros = zeros.t().contiguous()

    return pack_gemm_a16w4_weight(q), scales.t().contiguous(), w_zeros, x_perm


def awq_to_gemm_a16w4(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor
):
    """Convert 4-bit AWQ (GEMM format) checkpoint tensors to the gemm_a16w4 inputs.

    Args:
        qweight: int32 (K, N // 8). Each word holds 8 columns of N in the AWQ
            nibble order (0, 2, 4, 6, 1, 3, 5, 7).
        scales: (K // group_size, N).
        qzeros: int32 (K // group_size, N // 8), packed like qweight.

    Returns:
        (w, w_scales, w_zeros) for gemm_a16w4.
    """
    K, N = qweight.shape[0], qweight.shape[1] * 8
    num_groups = scales.shape[0]
    assert scales.shape == (num_groups, N) and qzeros.shape == (num_groups, N // 8)
    order = torch.tensor(_AWQ_NIBBLE, device=qweight.device)
    q = _unpack_int32_uint4(qweight).view(K, N // 8, 8)[:, :, order].reshape(K, N)
    zeros = _unpack_int32_uint4(qzeros).view(num_groups, N // 8, 8)[:, :, order]
    w_zeros = zeros.reshape(num_groups, N).t().contiguous()
    return pack_gemm_a16w4_weight(q.t()), scales.t().contiguous(), w_zeros


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
    return_layout: bool = False,
):
    """Arch-aware MoE scale shuffle (a8w4 / a8w8 / a16w4 / a4w4 family).

    gfx950 / gfx1250: preshuffle_factor = 32, scale_kwidth = 8.

    With ``return_layout=True`` also returns the matching ``SWIZZLE_MX_SCALE``
    label ("GFX1250_SCALE" for gfx1250, "CDNA4_SCALE" for gfx950) as
    ``(scale, label)``, so callers stay arch-agnostic; otherwise returns just
    the shuffled scale tensor.

    On archs with no native preshuffled MX scale layout (i.e. not gfx950 /
    gfx1250, e.g. gfx942), this is a no-op: returns ``data`` unchanged and
    ``layout=None``, so callers can invoke this unconditionally without an
    arch check of their own.
    """
    arch = arch or get_arch()
    if arch == "gfx1250":
        tiled = _shuffle_scale_tile_gfx1250(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "GFX1250_SCALE"
    elif arch == "gfx950":
        tiled = _shuffle_scale_tile_gfx950(
            data.transpose(-1, -2), preshuffle_factor, scale_kwidth
        )
        layout = "CDNA4_SCALE"
    else:
        return (data, None) if return_layout else data
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
