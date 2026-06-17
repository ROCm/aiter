# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL dynamic quantization APIs.

Matches the call signatures used by the CUDA reference in
``aiter/csrc/kernels/quant_kernels.cu`` (``dynamic_per_tensor_quant``) so it
can be plugged into the same precision and perf tests.
"""

from __future__ import annotations

import torch

from .kernels.dynamic_quant import (
    GROUP_QUANT_BLOCK_SIZE as _FP4_GROUP_QUANT_BLOCK_SIZE,
    FP4_GROUP_SIZE as _FP4_GROUP_SIZE,
    build_dynamic_per_tensor_quant_module,
    build_per_1x32_fp4_quant_module,
    build_per_1x32_fp4_quant_hadamard_module,
    build_per_1x32_fp4_quant_block_rotation_module,
    build_per_1x32_fp4_quant_block_rotation_mfma_module,
    build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module,
    build_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_module,
)
from .kernels.tensor_shim import get_dtype_str

__all__ = [
    "flydsl_dynamic_per_tensor_quant",
    "flydsl_per_1x32_fp4_quant",
    "flydsl_per_1x32_fp4_quant_hadamard",
    "flydsl_per_1x32_fp4_quant_block_rotation",
    "flydsl_per_1x32_fp4_quant_block_rotation_mfma",
    "flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace",
    "flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort",
    "mxfp4_singrot_R_preshuffle",
]


def mxfp4_singrot_R_preshuffle(
    rot_R: torch.Tensor,
    *,
    rot_transposed: bool = False,
) -> torch.Tensor:
    """Permute a single ``(32, 32)`` rotation matrix into MFMA B-fragment order.

    The single-shared-R rotation+quant kernel needs, for output tile
    ``(n_tile, k_tile)`` and lane ``L``, the B-fragment
    ``R[n = n_tile*16 + L%16][k = k_tile*16 + (L//16)*4 + 0..3]`` (4 contiguous
    K). Reading that straight from a row-major ``[n][k]`` ``R`` is a strided
    per-lane gather (adjacent lanes 64 B apart). Pre-laying ``R`` so the
    fragment lands contiguously at flat element
    ``((n_tile*NUM_K_TILES + k_tile)*64 + L)*4`` lets the kernel read it with a
    single lane-contiguous coalesced load (adjacent lanes 8 B apart) -- exactly
    the MoE preshuffled-weight trick.

    Parameters
    ----------
    rot_R : ``(32, 32)`` / ``(1, 32, 32)`` single matrix, or a per-block
        ``(scale_N, 32, 32)`` stack (the permutation is applied to each block).
    rot_transposed : if True, ``rot_R`` is stored ``[k][n]``; it is transposed
        to the canonical ``[n][k]`` before shuffling (so the kernel can always
        treat the shuffled tensor as non-transposed).

    Returns
    -------
    A contiguous ``(32, 32)`` tensor (same dtype/device) whose flat layout is
    the fragment order, carrying a ``rot_preshuffled = True`` marker attribute.
    Pass it to :func:`flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace`
    (single-R); the dispatch detects the marker and selects the shuffled load.
    """
    G = _FP4_GROUP_SIZE          # 32
    TILE = 16
    NT = G // TILE               # 2 n-tiles
    KT = G // TILE               # 2 k-tiles
    FV = 4                       # WMMA_FRAG_VALS
    orig_shape = tuple(rot_R.shape)
    # Accept a single (32,32)/(1,32,32) matrix or (B,32,32) stack; shuffle each block.
    R = rot_R.reshape(-1, G, G)              # (B, 32, 32)
    if rot_transposed:
        R = R.transpose(-1, -2)
    B = R.shape[0]
    Rf = R.contiguous().reshape(B, G * G)    # [b, n*32 + k]
    L = torch.arange(64, device=R.device)
    n_r = L % TILE                           # n within tile (0..15)
    g = L // TILE                            # lane group 0..3
    out = torch.empty(B, NT * KT * 64 * FV, dtype=Rf.dtype, device=R.device)
    for nt in range(NT):
        for kt in range(KT):
            n = nt * TILE + n_r              # (64,)
            k0 = kt * TILE + g * FV          # (64,)
            blk = (nt * KT + kt) * 64 * FV
            for i in range(FV):
                src = n * G + (k0 + i)       # (64,)
                out[:, blk + L * FV + i] = Rf[:, src]
    out = out.reshape(orig_shape).contiguous()
    out.rot_preshuffled = True
    return out


def _out_dtype_str(dtype: torch.dtype) -> str:
    if dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        return "fp8"
    if dtype == torch.int8:
        return "i8"
    raise ValueError(f"unsupported out dtype: {dtype}")


def flydsl_dynamic_per_tensor_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place dynamic per-tensor scaled quantization.

    Parameters
    ----------
    out : torch.Tensor
        Output buffer, same shape as ``input``, dtype fp8 (E4M3 FN/FNUZ).
    input : torch.Tensor
        bf16 or fp16 input, last dim is the quantization axis.
    scale : torch.Tensor
        Length-1 fp32 buffer holding ``max(|input|) / dtype_max`` on return.
        The kernel zeros it internally before computing.
    stream : optional ``torch.cuda.Stream``
        Defaults to the current CUDA stream.

    Notes
    -----
    Matches the semantics of ``aiter.dynamic_per_tensor_quant`` (CUDA): the
    stored scale is the *dequant* scale, i.e. ``y = round(x / scale)``.
    """
    assert input.is_contiguous(), "FlyDSL dynamic_per_tensor_quant requires contiguous input"
    assert out.is_contiguous(), "FlyDSL dynamic_per_tensor_quant requires contiguous out"
    assert input.shape == out.shape, (
        f"shape mismatch: input {tuple(input.shape)} vs out {tuple(out.shape)}"
    )
    assert scale.numel() == 1 and scale.dtype == torch.float32, (
        f"scale must be a 1-elem fp32 tensor, got {tuple(scale.shape)} {scale.dtype}"
    )

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    # tensor_shim uses "f16"; the kernel module wants "fp16".
    if in_dtype == "f16":
        in_dtype = "fp16"

    out_dtype = _out_dtype_str(out.dtype)

    cols = int(input.shape[-1])
    rows = int(input.numel() // cols)

    # Zero the global accumulator the kernel atomically max-reduces into,
    # enqueued on the active stream (stays async w.r.t. the host).
    if stream is None:
        scale.zero_()
    else:
        with torch.cuda.stream(stream):
            scale.zero_()

    launcher = build_dynamic_per_tensor_quant_module(
        cols=cols,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        use_ptr64=bool((rows * cols) >= (1 << 31)),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(input, out, scale, rows, stream)


def flydsl_per_1x32_fp4_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place MXFP4 per-1x32 dynamic quantization.

    Mirrors ``aiter.dynamic_per_group_scaled_quant_fp4`` (CUDA, ``group_size=32,
    shuffle_scale=False``): per 32-elem group, OCP MX scale, fp4 E2M1 packed 2/byte.

    Parameters
    ----------
    out : torch.Tensor
        Output buffer of shape ``(*input.shape[:-1], input.shape[-1] // 2)``,
        dtype ``uint8`` or ``fp4x2`` (which is a uint8 view).
    input : torch.Tensor
        bf16 or fp16 contiguous input; last-dim must be a multiple of 32.
    scale : torch.Tensor
        ``uint8``/``fp8_e8m0`` buffer of shape ``(*input.shape[:-1],
        input.shape[-1] // 32)`` holding the per-group E8M0 scale exponents.
    stream : optional ``torch.cuda.Stream``
        Defaults to the current CUDA stream.
    """
    assert input.is_contiguous(), "FlyDSL per_1x32_fp4_quant requires contiguous input"
    assert out.is_contiguous(), "FlyDSL per_1x32_fp4_quant requires contiguous out"

    cols = int(input.shape[-1])
    rows = int(input.numel() // cols)
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )

    scale_n = cols // _FP4_GROUP_SIZE
    total_groups = rows * scale_n
    num_blocks = (
        total_groups + _FP4_GROUP_QUANT_BLOCK_SIZE - 1
    ) // _FP4_GROUP_QUANT_BLOCK_SIZE

    # Shape sanity (cheap host-side check; HW would error otherwise).
    expected_out_last = cols // 2
    assert out.shape[-1] == expected_out_last, (
        f"out last dim {out.shape[-1]} != cols/2 = {expected_out_last}"
    )
    assert int(out.numel()) == rows * expected_out_last, (
        f"out numel {out.numel()} != rows*cols/2 = {rows * expected_out_last}"
    )
    assert int(scale.numel()) == total_groups, (
        f"scale numel {scale.numel()} != total_groups = {total_groups}"
    )

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    if in_dtype == "f16":
        in_dtype = "fp16"

    launcher = build_per_1x32_fp4_quant_module(
        cols=cols,
        in_dtype=in_dtype,
        shuffle_scale=False,
        use_ptr64=bool((rows * cols) >= (1 << 31)),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(input, out, scale, num_blocks, total_groups, stream)


def flydsl_per_1x32_fp4_quant_hadamard(
    out: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    *,
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place MXFP4 per-1x32 dynamic quant **with fused H_32 rotation**.

    Like ``flydsl_per_1x32_fp4_quant`` but applies ``H_32 / sqrt(32)`` per group
    before amax; the scale dequants rotated values (Walsh-Hadamard is self-inverse).
    """
    assert input.is_contiguous(), "FlyDSL per_1x32_fp4_quant_hadamard requires contiguous input"
    assert out.is_contiguous(), "FlyDSL per_1x32_fp4_quant_hadamard requires contiguous out"

    cols = int(input.shape[-1])
    rows = int(input.numel() // cols)
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )

    scale_n = cols // _FP4_GROUP_SIZE
    total_groups = rows * scale_n
    num_blocks = (
        total_groups + _FP4_GROUP_QUANT_BLOCK_SIZE - 1
    ) // _FP4_GROUP_QUANT_BLOCK_SIZE

    expected_out_last = cols // 2
    assert out.shape[-1] == expected_out_last, (
        f"out last dim {out.shape[-1]} != cols/2 = {expected_out_last}"
    )
    assert int(out.numel()) == rows * expected_out_last, (
        f"out numel {out.numel()} != rows*cols/2 = {rows * expected_out_last}"
    )
    assert int(scale.numel()) == total_groups, (
        f"scale numel {scale.numel()} != total_groups = {total_groups}"
    )

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    if in_dtype == "f16":
        in_dtype = "fp16"

    launcher = build_per_1x32_fp4_quant_hadamard_module(
        cols=cols,
        in_dtype=in_dtype,
        shuffle_scale=False,
        use_ptr64=bool((rows * cols) >= (1 << 31)),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(input, out, scale, num_blocks, total_groups, stream)


def flydsl_per_1x32_fp4_quant_block_rotation(
    out: torch.Tensor,
    input: torch.Tensor,
    rot_R: torch.Tensor,
    scale: torch.Tensor,
    *,
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place MXFP4 per-1x32 dynamic quant **with fused per-block rotation**.

    Single-kernel ``y = einsum("...bg,bhg->...bh", x, rot_R); quant_mxfp4_per_1x32(y)``
    over ``input (rows, cols)``, ``rot_R (cols//32, 32, 32)``; scale dequants rotated
    values, so consumers must apply ``R[b]^T`` on top of dequant.
    """
    assert input.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation requires contiguous input"
    assert rot_R.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation requires contiguous rot_R"
    assert out.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation requires contiguous out"
    assert input.dim() == 2, f"input must be 2-D (rows, cols), got shape {input.shape}"
    assert rot_R.dim() == 3, f"rot_R must be 3-D (B, g, g), got shape {rot_R.shape}"

    rows = int(input.shape[0])
    cols = int(input.shape[-1])
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )
    scale_n = cols // _FP4_GROUP_SIZE
    expected_R_shape = (scale_n, _FP4_GROUP_SIZE, _FP4_GROUP_SIZE)
    assert tuple(rot_R.shape) == expected_R_shape, (
        f"rot_R shape {tuple(rot_R.shape)} != expected {expected_R_shape}"
    )

    expected_out_last = cols // 2
    assert out.shape == (rows, expected_out_last), (
        f"out shape {tuple(out.shape)} != expected ({rows}, {expected_out_last})"
    )
    assert scale.shape == (rows, scale_n), (
        f"scale shape {tuple(scale.shape)} != expected ({rows}, {scale_n})"
    )

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    if in_dtype == "f16":
        in_dtype = "fp16"

    rot_dtype = get_dtype_str(rot_R.dtype)
    if rot_dtype not in ("bf16", "f16", "fp32", "f32"):
        raise ValueError(
            f"unsupported rot_R dtype {rot_R.dtype}; only bf16/fp16/fp32 supported"
        )
    if rot_dtype == "f16":
        rot_dtype = "fp16"
    if rot_dtype == "f32":
        rot_dtype = "fp32"

    num_m_blocks = (rows + _FP4_GROUP_QUANT_BLOCK_SIZE - 1) // _FP4_GROUP_QUANT_BLOCK_SIZE

    launcher = build_per_1x32_fp4_quant_block_rotation_module(
        cols=cols,
        in_dtype=in_dtype,
        rot_dtype=rot_dtype,
        shuffle_scale=False,
        use_ptr64=bool((rows * cols) >= (1 << 31)),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(input, rot_R, out, scale, num_m_blocks, rows, stream)


def flydsl_per_1x32_fp4_quant_block_rotation_mfma(
    out: torch.Tensor,
    input: torch.Tensor,
    rot_R: torch.Tensor,
    scale: torch.Tensor,
    *,
    rot_transposed: bool = False,
    stream: torch.cuda.Stream = None,
) -> None:
    """MFMA-accelerated variant of :func:`flydsl_per_1x32_fp4_quant_block_rotation`.

    Uses ``v_mfma_f32_16x16x16_{bf16,f16}_1k`` instead of the scalar FMA chain;
    ``input.dtype`` must match ``rot_R.dtype`` (bf16/fp16, no fp32).

    Parameters
    ----------
    rot_transposed : bool, default False
        ``False``: ``rot_R[b,h,g]`` is ``R`` (``Y[m] = X[m] @ R.T``); ``True``:
        ``rot_R[b,g,h]`` is ``R.T`` (``Y[m] = X[m] @ R``). Compile-time flag.

    See :func:`flydsl_per_1x32_fp4_quant_block_rotation` for the full contract.
    """
    assert input.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation_mfma requires contiguous input"
    assert rot_R.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation_mfma requires contiguous rot_R"
    assert out.is_contiguous(), "FlyDSL per_1x32_fp4_quant_block_rotation_mfma requires contiguous out"
    assert input.dim() == 2, f"input must be 2-D (rows, cols), got shape {input.shape}"
    assert rot_R.dim() == 3, f"rot_R must be 3-D (B, g, g), got shape {rot_R.shape}"

    rows = int(input.shape[0])
    cols = int(input.shape[-1])
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )
    scale_n = cols // _FP4_GROUP_SIZE
    expected_R_shape = (scale_n, _FP4_GROUP_SIZE, _FP4_GROUP_SIZE)
    assert tuple(rot_R.shape) == expected_R_shape, (
        f"rot_R shape {tuple(rot_R.shape)} != expected {expected_R_shape}"
    )

    expected_out_last = cols // 2
    assert out.shape == (rows, expected_out_last), (
        f"out shape {tuple(out.shape)} != expected ({rows}, {expected_out_last})"
    )
    assert scale.shape == (rows, scale_n), (
        f"scale shape {tuple(scale.shape)} != expected ({rows}, {scale_n})"
    )

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    if in_dtype == "f16":
        in_dtype = "fp16"

    rot_dtype = get_dtype_str(rot_R.dtype)
    if rot_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported rot_R dtype {rot_R.dtype} for MFMA path; "
            "only bf16/fp16 supported (use flydsl_per_1x32_fp4_quant_block_rotation for fp32 R)"
        )
    if rot_dtype == "f16":
        rot_dtype = "fp16"

    # MFMA requires A and B to share the same fp type.
    if in_dtype != rot_dtype:
        raise ValueError(
            f"MFMA path requires input.dtype == rot_R.dtype "
            f"(got input={input.dtype}, rot_R={rot_R.dtype}); "
            "use flydsl_per_1x32_fp4_quant_block_rotation for mixed dtypes"
        )

    num_m_blocks = (rows + _FP4_GROUP_QUANT_BLOCK_SIZE - 1) // _FP4_GROUP_QUANT_BLOCK_SIZE

    launcher = build_per_1x32_fp4_quant_block_rotation_mfma_module(
        cols=cols,
        in_dtype=in_dtype,
        rot_dtype=rot_dtype,
        shuffle_scale=False,
        rot_transposed=bool(rot_transposed),
        use_ptr64=bool((rows * cols) >= (1 << 31)),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(input, rot_R, out, scale, num_m_blocks, rows, stream)


def _read_rot_R_transpose_attr(rot_R: torch.Tensor) -> bool:
    """Return user-set ``rot_R.transpose`` bool (default ``False``).

    Only honoured when a ``bool`` is bound (else it's the built-in method).
    """
    attr = getattr(rot_R, "transpose", False)
    return attr if isinstance(attr, bool) else False


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is ``<= cap`` (>= 1)."""
    cap = max(1, min(cap, n))
    for d in range(cap, 0, -1):
        if n % d == 0:
            return d
    return 1


def _select_single_rot_blocks_per_wg(
    blocks_per_wg, scale_n: int, num_sorted_blocks: int, device,
) -> int:
    """Pick ``K`` (blocks per workgroup) for the single-shared-R kernel.

    ``"auto"`` resolves to ``K = 1``: merging blocks is net-negative on MI355
    (it kills inter-WG latency hiding). An explicit int is a power-user knob.
    """
    if isinstance(blocks_per_wg, int):
        return _largest_divisor_leq(scale_n, max(1, blocks_per_wg))
    if blocks_per_wg != "auto":
        raise ValueError(
            f"blocks_per_wg must be an int or 'auto', got {blocks_per_wg!r}"
        )
    # Measured best: no merging.
    return 1


# Measured-best (persist_m, blocks_per_wg, waves_per_wg) per token count M
# (single-shared-R). Auto-tune picks the first bucket whose M_max >= M.
_SINGLE_ROT_AUTO_TABLE = (
    # (M_max, persist_m, blocks_per_wg, waves_per_wg)
    (1 << 10,  1, 1, 1),   # <= 1024
    (1 << 11,  2, 1, 2),   # <= 2048
    (1 << 12,  2, 1, 4),   # <= 4096
    (1 << 13,  1, 1, 1),   # <= 8192
    (1 << 14,  2, 1, 4),   # <= 16384
    (1 << 15, 16, 2, 2),   # <= 32768
    (40960,    1, 1, 1),   # <= 40960
    (1 << 16, 32, 2, 2),   # <= 65536 and beyond
)


def _single_rot_auto_config(m_rows: int):
    """Return (persist_m, blocks_per_wg, waves_per_wg) tuned for M = ``m_rows``
    (smallest bucket whose M_max >= M)."""
    for m_max, pm, k, w in _SINGLE_ROT_AUTO_TABLE:
        if m_rows <= m_max:
            return pm, k, w
    return _SINGLE_ROT_AUTO_TABLE[-1][1:]


# Measured-best (persist_m, blocks_per_wg, waves_per_wg) per token count M for
# the PER-BLOCK R[b] kernel; K is always 1 (per-block K>1 serializes R[b] loads).
_PERBLOCK_ROT_AUTO_TABLE = (
    # (M_max, persist_m, blocks_per_wg, waves_per_wg)
    (1 << 13,  1, 1, 2),   # <= 8192   : W=2 hides the per-row latency
    (1 << 14,  2, 1, 1),   # <= 16384  : persist_m=2, single wave
    (1 << 31,  1, 1, 1),   # large M   : plain pm1/K1/W1
)


def _perblock_rot_auto_config(m_rows: int):
    """Return (persist_m, blocks_per_wg, waves_per_wg) tuned for M = ``m_rows``
    on the per-block R[b] path (smallest covering bucket)."""
    for m_max, pm, k, w in _PERBLOCK_ROT_AUTO_TABLE:
        if m_rows <= m_max:
            return pm, k, w
    return _PERBLOCK_ROT_AUTO_TABLE[-1][1:]


def flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace(
    out: torch.Tensor,
    input: torch.Tensor,
    rot_R: torch.Tensor,
    out_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    *,
    rot_transposed: bool = False,
    blocks_per_wg: "int | str" = "auto",
    waves_per_wg: "int | None" = None,
    preload_b: "bool | None" = None,
    persist_m: "int | None" = None,
    xcd_remap: bool = False,
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place rotation + per-1x32 MXFP4 quant + MoE scale-sort.

    Single-kernel MFMA rotation + per-1x32 MXFP4 quant, then a MoE scale-sort;
    mirrors CUDA ``fused_dynamic_mxfp4_quant_moe_sort_hip``. ``topk`` is inferred
    as ``rows // token_num``. Only routed rows of ``out`` are written.

    Parameters
    ----------
    out : ``(rows, cols // 2)`` ``uint8`` (``fp4x2``) packed output. Contiguous.
    input : ``(rows, cols)`` ``bf16`` / ``fp16``. Contiguous.
    rot_R : ``(cols // 32, 32, 32)`` ``bf16`` / ``fp16`` matching ``input``. A
        single ``(32, 32)`` / ``(1, 32, 32)`` matrix takes the shared-R fast path.
    blocks_per_wg : int or ``"auto"``, default ``"auto"``
        Blocks per workgroup (single-R path); ``"auto"`` is the measured-best ``K``
        for ``M = rows``. Per-block path always uses ``K = 1``.
    waves_per_wg : int or ``None``, default ``None``
        Waves per workgroup; ``None`` takes the measured-best ``W`` for ``M``.
    persist_m : int or ``None``, default ``None``
        Persistent row-block groups; ``None`` takes the measured-best value.
    out_scale : ``(((sorted_ids.shape[0]+31)//32)*32, cols//32)`` ``fp8_e8m0``
        MoE-sorted, MXFP4-shuffled scale dest (as ``mxfp4_moe_sort_fwd`` returns).
    sorted_ids, num_valid_ids, token_num
        MoE sort inputs; same semantics as ``aiter.ops.quant.mxfp4_moe_sort_fwd``.
    rot_transposed : ``bool``, default ``False``. Rotation tensor layout (see
        :func:`flydsl_per_1x32_fp4_quant_block_rotation_mfma`).
    stream : optional ``torch.cuda.Stream`` -- defaults to the current stream.
    """
    assert input.dim() == 2, f"input must be 2-D (rows, cols), got {input.shape}"
    rows = int(input.shape[0])
    cols = int(input.shape[-1])
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )
    scale_n = cols // _FP4_GROUP_SIZE

    # Fast path: a single 32x32 rot_R shares one R across blocks (LDS once);
    # a (scale_N, 32, 32) tensor uses the per-block kernel.
    _g = _FP4_GROUP_SIZE
    single_R = (
        (rot_R.dim() == 2 and tuple(rot_R.shape) == (_g, _g))
        or (rot_R.dim() == 3 and tuple(rot_R.shape) == (1, _g, _g))
    )
    # Read the host-applied preshuffle marker before reshape/contiguous drops it.
    _rot_pshuf = bool(getattr(rot_R, "rot_preshuffled", False))
    if not single_R:
        assert rot_R.dim() == 3, f"rot_R must be 3-D (B, g, g), got {rot_R.shape}"
        expected_R_shape = (scale_n, _g, _g)
        assert tuple(rot_R.shape) == expected_R_shape, (
            f"rot_R shape {tuple(rot_R.shape)} != expected {expected_R_shape}"
        )

    expected_out_shape = (rows, cols // 2)
    assert tuple(out.shape) == expected_out_shape, (
        f"out shape {tuple(out.shape)} != expected {expected_out_shape}"
    )
    expected_out_scale_rows = (sorted_ids.shape[0] + 31) // 32 * 32
    assert tuple(out_scale.shape) == (expected_out_scale_rows, scale_n), (
        f"out_scale shape {tuple(out_scale.shape)} != expected "
        f"({expected_out_scale_rows}, {scale_n})"
    )

    if token_num <= 0 or rows % token_num != 0:
        raise ValueError(
            f"rows ({rows}) must be a positive multiple of token_num "
            f"({token_num}); topk is inferred as rows // token_num"
        )
    topk = rows // token_num

    in_dtype = get_dtype_str(input.dtype)
    if in_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported input dtype {input.dtype}; only bf16/fp16 supported"
        )
    if in_dtype == "f16":
        in_dtype = "fp16"
    rot_dtype = get_dtype_str(rot_R.dtype)
    if rot_dtype not in ("bf16", "f16"):
        raise ValueError(
            f"unsupported rot_R dtype {rot_R.dtype} for MFMA+sort path; "
            "only bf16/fp16 supported"
        )
    if rot_dtype == "f16":
        rot_dtype = "fp16"
    if in_dtype != rot_dtype:
        raise ValueError(
            f"MFMA+sort path requires input.dtype == rot_R.dtype "
            f"(got input={input.dtype}, rot_R={rot_R.dtype})"
        )

    if not input.is_contiguous():
        input = input.contiguous()
    if not rot_R.is_contiguous():
        rot_R = rot_R.contiguous()
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if not out_scale.is_contiguous():
        raise ValueError("out_scale must be contiguous")
    if sorted_ids.dtype != torch.int32:
        sorted_ids = sorted_ids.to(torch.int32)
    if not sorted_ids.is_contiguous():
        sorted_ids = sorted_ids.contiguous()
    if num_valid_ids.dtype != torch.int32:
        num_valid_ids = num_valid_ids.to(torch.int32)

    # Both kernels traverse contiguous source rows, so the grid is sized by real
    # source-row blocks (not the padded sorted space).
    num_row_blocks = (
        rows + _FP4_GROUP_QUANT_BLOCK_SIZE - 1
    ) // _FP4_GROUP_QUANT_BLOCK_SIZE

    # Use the slower 64-bit GEP only when the i32 offset src_row*cols would
    # overflow at rows*cols >= 2**31; else keep the cheap buffer load.
    use_ptr64 = (rows * cols) >= (1 << 31)

    if single_R:
        # Single shared (32, 32) R; normalize (1, 32, 32) and pick K.
        rot_R = rot_R.reshape(_g, _g)
        # Auto-tune (persist_m, K, W) for M = rows; explicit args override.
        auto_pm, auto_k, auto_w = _single_rot_auto_config(rows)
        k = _select_single_rot_blocks_per_wg(
            auto_k if blocks_per_wg == "auto" else blocks_per_wg,
            scale_n, num_row_blocks, input.device,
        )
        # Multi-wave factor: explicit arg, else tuned default.
        w = auto_w if waves_per_wg is None else int(waves_per_wg)
        # Preload R B-fragments into registers: explicit arg, else on.
        pb = True if preload_b is None else bool(preload_b)
        # Preshuffle (default on): trust host marker or shuffle here; implies
        # preload_b and folds in rot_transposed.
        if not _rot_pshuf:
            rot_R = mxfp4_singrot_R_preshuffle(
                rot_R, rot_transposed=bool(rot_transposed),
            )
            _rot_pshuf = True
        if _rot_pshuf:
            pb = True  # shuffled load goes straight HBM->VGPR
        launcher = (
            build_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_module(
                cols=cols,
                in_dtype=in_dtype,
                rot_dtype=rot_dtype,
                topk=topk,
                # rot_transposed is absorbed by the host shuffle when preshuffled
                rot_transposed=bool(rot_transposed) and not _rot_pshuf,
                use_ptr64=bool(use_ptr64),
                blocks_per_wg=k,
                waves_per_wg=w,
                preload_b=pb,
                rot_preshuffled=_rot_pshuf,
                persist_m=auto_pm if persist_m is None else int(persist_m),
                xcd_remap=bool(xcd_remap),
            )
        )
    else:
        # Per-block R[b]: auto-tune (persist_m, K, W) for M = rows; explicit args
        # override. The table always returns K = 1 (per-block K>1 serializes loads).
        auto_pm, auto_k, auto_w = _perblock_rot_auto_config(rows)
        # Multi-wave amortizes the one R[b] load over W waves. Explicit arg, else default.
        w = auto_w if waves_per_wg is None else int(waves_per_wg)
        # K: "auto" takes the tuned K (always 1); explicit int clamped to divide scale_n.
        if isinstance(blocks_per_wg, str):
            kb = auto_k
        else:
            kb = int(blocks_per_wg)
            if kb < 1 or scale_n % kb != 0:
                kb = 1
        # Preshuffle (default on) per block; trust host marker or shuffle here,
        # folding in rot_transposed.
        if not _rot_pshuf:
            rot_R = mxfp4_singrot_R_preshuffle(
                rot_R, rot_transposed=bool(rot_transposed),
            )
            _rot_pshuf = True
        # Preshuffle implies preload_b; persist_m defaults to 1 for this path.
        pb = True if (preload_b is None or _rot_pshuf) else bool(preload_b)
        launcher = build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module(
            cols=cols,
            in_dtype=in_dtype,
            rot_dtype=rot_dtype,
            topk=topk,
            # rot_transposed absorbed by the host shuffle when preshuffled
            rot_transposed=bool(rot_transposed) and not _rot_pshuf,
            use_ptr64=bool(use_ptr64),
            waves_per_wg=w,
            blocks_per_wg=kb,
            preload_b=pb,
            rot_preshuffled=_rot_pshuf,
            persist_m=auto_pm if persist_m is None else int(persist_m),
            xcd_remap=bool(xcd_remap),
        )

    if stream is None:
        stream = torch.cuda.current_stream()

    # Step 1: rotation + per-1x32 MXFP4 quant (NO sort). Contiguous source rows
    # -> plain unsorted scale temp + natural-order fp4x2 ``out`` (no gather).
    from aiter import dtypes as _dtypes

    scale_unsorted = torch.empty(
        rows, scale_n, dtype=_dtypes.fp8_e8m0, device=input.device,
    )
    launcher(
        input, rot_R, out, scale_unsorted.view(torch.uint8),
        num_row_blocks, token_num, stream,
    )

    # Step 2: MoE scale-sort + MXFP4 shuffle -> out_scale in the MoE-sorted,
    # MXFP4-shuffled layout (bit-identical to ``mxfp4_moe_sort_fwd``).
    from aiter.ops.quant import mxfp4_moe_sort_hip

    with torch.cuda.stream(stream):
        mxfp4_moe_sort_hip(
            out_scale, scale_unsorted, sorted_ids, num_valid_ids,
            token_num, cols,
        )


def flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort(
    input: torch.Tensor,
    rot_R: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    *,
    rot_transposed: "bool | None" = None,
    stream: torch.cuda.Stream = None,
):
    """Alloc-and-return wrapper around
    :func:`flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace`.

    API matches ``aiter.ops.quant.fused_dynamic_mxfp4_quant_moe_sort``; see the
    in-place function for kernel-level details.

    Parameters
    ----------
    input : ``(rows, cols)`` ``bf16`` / ``fp16``.
    rot_R : ``(cols // 32, 32, 32)`` ``bf16`` / ``fp16``. Set ``rot_R.transpose``
        or pass ``rot_transposed=True`` for the ``[b, g, h]`` layout.
    sorted_ids, num_valid_ids, token_num
        Same semantics as :func:`aiter.ops.quant.mxfp4_moe_sort_fwd`.
    rot_transposed : optional ``bool`` override; ``None`` reads
        ``rot_R.transpose``.

    Returns
    -------
    out_fp4x2 : ``(rows, cols // 2)`` viewed as ``fp4x2``.
    out_scale : ``(((sorted_ids.shape[0] + 31) // 32) * 32, cols // 32)``
        ``fp8_e8m0``, in the MXFP4-shuffled MoE-sorted layout.
    """
    from aiter import dtypes

    if rot_transposed is None:
        rot_transposed = _read_rot_R_transpose_attr(rot_R)

    assert input.dim() == 2, f"input must be 2-D (rows, cols), got {input.shape}"
    rows = int(input.shape[0])
    cols = int(input.shape[-1])
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )

    out_u8 = torch.empty(rows, cols // 2, dtype=torch.uint8, device=input.device)
    out_scale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32,
        (cols + _FP4_GROUP_SIZE - 1) // _FP4_GROUP_SIZE,
        dtype=dtypes.fp8_e8m0,
        device=input.device,
    )

    flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace(
        out_u8,
        input,
        rot_R,
        out_scale,
        sorted_ids,
        num_valid_ids,
        token_num,
        rot_transposed=bool(rot_transposed),
        stream=stream,
    )

    return out_u8.view(dtypes.fp4x2), out_scale
