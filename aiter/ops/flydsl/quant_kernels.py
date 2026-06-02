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
]


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
    # tensor_shim uses "f16" for half but the kernel module uses "fp16"; normalize.
    if in_dtype == "f16":
        in_dtype = "fp16"

    out_dtype = _out_dtype_str(out.dtype)

    cols = int(input.shape[-1])
    rows = int(input.numel() // cols)

    # Host-side init: zero the global accumulator that data_to_scale_kernel
    # atomically max-reduces into. ``zero_`` is enqueued on the active stream
    # so this stays async w.r.t. the host.
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

    Mirrors the semantics of ``aiter.dynamic_per_group_scaled_quant_fp4``
    (CUDA) with ``group_size=32, shuffle_scale=False``:

        for each contiguous group of 32 elements in the last dim:
            scale = round_up_pow2(amax) * 0.25              # OCP MX scale
            out[group] = HW-quantize(input[group] / scale)  # fp4 E2M1, packed 2-per-byte
            scale_e8m0 = (bitcast<u32>(scale) >> 23) & 0xFF

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

    Equivalent of ``flydsl_per_1x32_fp4_quant`` but applies the orthonormal
    Walsh-Hadamard transform ``H_32 / sqrt(32)`` to each group of 32 elements
    immediately after loading and before computing the per-group amax.

    The stored E8M0 scale is the dequant scale for the **rotated** values, so
    a downstream consumer must apply the inverse rotation
    (``H_32 / sqrt(32)`` since Walsh-Hadamard is its own inverse up to
    normalization) after the dequant step.

    See ``flydsl_per_1x32_fp4_quant`` for the rest of the contract (shape /
    layout / dtype rules).
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

    Equivalent of running
    ``y = apply_block_rotation(input, rot_R); quant_mxfp4_per_1x32(y)``
    in a single kernel:

      *   ``input``         : ``(rows, cols)``               bf16 / fp16, contiguous
      *   ``rot_R``         : ``(cols // 32, 32, 32)``       bf16 / fp16 / f32, contiguous
      *   ``out``           : ``(rows, cols // 2)``          uint8 (fp4x2)
      *   ``scale``         : ``(rows, cols // 32)``         uint8 (fp8 e8m0)

    The rotation is the same block-diagonal map as the host reference

    .. code-block:: python

        def apply_block_rotation(x, R):
            *lead, N = x.shape
            B, g, _ = R.shape                   # g == 32
            xb = x.reshape(*lead, B, g)
            return torch.einsum("...bg,bhg->...bh", xb, R).reshape(*lead, N)

    LDS reuse of ``R[b]`` makes the kernel cheap when many tokens share the
    same group: ``rot_R`` is read from VRAM only ``rows / 64`` times per b.

    The stored E8M0 scale is the dequant scale of the *rotated* values.
    Downstream consumers must invert by applying ``R[b]^T`` (assumed
    orthogonal) on top of dequant.
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

    Implemented with ``v_mfma_f32_16x16x16_{bf16,f16}_1k`` instead of the
    1024-FMA scalar chain. Restrictions:

    *   ``input.dtype`` must match ``rot_R.dtype`` (both bf16 or both fp16).
    *   ``rot_R`` cannot be fp32; use the scalar wrapper for fp32 R.

    Parameters
    ----------
    rot_transposed : bool, default False
        How to interpret the last two dims of ``rot_R`` (shape
        ``(scale_N, 32, 32)``):

        - ``False``: ``rot_R[b, h, g]`` is the rotation matrix ``R``.
          The kernel computes
          ``y[m, b*32 + h] = sum_g x[m, b*32 + g] * rot_R[b, h, g]``,
          equivalent to ``Y = einsum("...bg, bhg -> ...bh", X, rot_R)``
          (per-block ``Y[m] = X[m] @ R.T``).
        - ``True``: ``rot_R[b, g, h]`` is the rotation matrix stored
          transposed along its last two dims (i.e. the caller passes
          ``R.transpose(-1, -2)``). The kernel computes
          ``y[m, b*32 + h] = sum_g x[m, b*32 + g] * rot_R[b, g, h]``,
          equivalent to ``Y = einsum("...bg, bgh -> ...bh", X, rot_R)``
          (per-block ``Y[m] = X[m] @ R``).

        Both modes yield mathematically equivalent results when fed
        corresponding (transposed vs. non-transposed) ``rot_R`` tensors.
        The flag is compile-time (selects a different cached kernel via
        ``lru_cache``); there is no runtime branch. The transposed mode
        pays a small one-time per-workgroup overhead in the cooperative
        LDS load (~8 scalar stores per thread instead of 1 vec8), then
        proceeds with an identical MFMA hot path.

    All other shape / contiguity constraints are identical to the scalar
    wrapper. See :func:`flydsl_per_1x32_fp4_quant_block_rotation` for the
    full contract.
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

    # MFMA requires both A and B same fp type at the hardware level.
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

    ``Tensor.transpose`` resolves to a built-in method by default, so we
    only honour the attribute when the caller has explicitly bound a
    ``bool`` to it.
    """
    attr = getattr(rot_R, "transpose", False)
    return attr if isinstance(attr, bool) else False


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
    stream: torch.cuda.Stream = None,
) -> None:
    """In-place rotation + per-1x32 MXFP4 quant + MoE scale-sort.

    **Single-kernel** fusion: launches
    :func:`build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module`,
    which iterates over destination ``sorted_row`` blocks, gathers each
    routed token via ``sorted_ids``, runs the MFMA-accelerated per-block
    rotation + per-1x32 MXFP4 quant, and writes:

      * the packed fp4x2 bytes to ``out`` at row
        ``token_idx * topk + topk_id`` (the topk-expanded source row),
      * the E8M0 scale byte directly into ``out_scale`` at the
        ``fp4_scale_shuffle_id(scaleN_pad, sorted_row, b)`` address.

    This mirrors the CUDA ``fused_dynamic_mxfp4_quant_moe_sort_hip``
    behaviour. ``topk`` is inferred as ``rows // token_num`` (stage1
    activations give ``topk == 1``; stage2 activations give the real
    ``topk``).

    NOTE on coverage: only ``out`` rows that appear in ``sorted_ids``
    (routed tokens) are written; unrouted rows are left untouched. This
    matches what the downstream tile-quantised CK GEMM consumes (it only
    reads routed positions). The previous two-kernel chain
    (``flydsl_per_1x32_fp4_quant_block_rotation_mfma`` +
    ``mxfp4_moe_sort_hip``) instead wrote every source row of ``out``;
    the MoE-sorted ``out_scale`` is identical between the two.

    Parameters
    ----------
    out : ``(rows, cols // 2)`` ``uint8`` (caller views as ``fp4x2``).
        Packed fp4x2 quant output. Must be contiguous.
    input : ``(rows, cols)`` ``bf16`` / ``fp16``. Contiguous.
    rot_R : ``(cols // 32, 32, 32)`` ``bf16`` / ``fp16``, dtype must
        match ``input``. Set ``rot_R.transpose = True`` (instance
        attribute) or pass ``rot_transposed=True`` to use the
        ``[b, g, h]`` storage layout; see
        :func:`flydsl_per_1x32_fp4_quant_block_rotation_mfma` for the
        full convention.
    out_scale : ``(((sorted_ids.shape[0] + 31) // 32) * 32, cols // 32)``
        ``fp8_e8m0`` -- MoE-sorted, MXFP4-shuffled scale destination.
        Allocated by the caller (matches what
        :func:`aiter.ops.quant.mxfp4_moe_sort_fwd` returns).
    sorted_ids, num_valid_ids, token_num
        MoE sort inputs as produced by ``moe_sort_block_fwd``; same
        semantics as :func:`aiter.ops.quant.mxfp4_moe_sort_fwd`.
    rot_transposed : ``bool``, default ``False``. Selects the kernel
        build for the rotation tensor layout (see
        :func:`flydsl_per_1x32_fp4_quant_block_rotation_mfma`).
    stream : optional ``torch.cuda.Stream`` -- defaults to the current
        CUDA stream.
    """
    assert input.dim() == 2, f"input must be 2-D (rows, cols), got {input.shape}"
    assert rot_R.dim() == 3, f"rot_R must be 3-D (B, g, g), got {rot_R.shape}"
    rows = int(input.shape[0])
    cols = int(input.shape[-1])
    if cols % _FP4_GROUP_SIZE != 0:
        raise ValueError(
            f"input last dim must be a multiple of {_FP4_GROUP_SIZE}, got {cols}"
        )
    scale_n = cols // _FP4_GROUP_SIZE

    expected_out_shape = (rows, cols // 2)
    assert tuple(out.shape) == expected_out_shape, (
        f"out shape {tuple(out.shape)} != expected {expected_out_shape}"
    )
    expected_R_shape = (scale_n, _FP4_GROUP_SIZE, _FP4_GROUP_SIZE)
    assert tuple(rot_R.shape) == expected_R_shape, (
        f"rot_R shape {tuple(rot_R.shape)} != expected {expected_R_shape}"
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

    num_sorted_blocks = (
        int(sorted_ids.shape[0]) + _FP4_GROUP_QUANT_BLOCK_SIZE - 1
    ) // _FP4_GROUP_QUANT_BLOCK_SIZE

    # The input gather computes an i32 element offset ``src_row * cols``; once
    # ``rows * cols >= 2**31`` that product overflows (and the bf16 byte offset
    # leaves the 4 GB buffer window). Only then pay for the slower 64-bit GEP +
    # global load; otherwise keep the cheap hardware buffer load. This mirrors
    # the wptr64 selection in the moe GEMM kernels.
    use_ptr64 = (rows * cols) >= (1 << 31)

    launcher = build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module(
        cols=cols,
        in_dtype=in_dtype,
        rot_dtype=rot_dtype,
        topk=topk,
        rot_transposed=bool(rot_transposed),
        use_ptr64=bool(use_ptr64),
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    # The kernel addresses ``out_scale`` by byte; pass a uint8 view so the
    # 1-byte E8M0 stores land regardless of the caller's scalar type.
    out_scale_u8 = out_scale.view(torch.uint8)

    launcher(
        input,
        rot_R,
        sorted_ids,
        num_valid_ids,
        out,
        out_scale_u8,
        num_sorted_blocks,
        token_num,
        stream,
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

    User-facing API matches
    :func:`aiter.ops.quant.fused_dynamic_mxfp4_quant_moe_sort`, so this
    is a drop-in upgrade when a per-block rotation has to be folded
    into the per-1x32 MXFP4 quant + MoE scale-sort pipeline. See the
    underlying in-place function for the kernel-level details.

    Parameters
    ----------
    input : ``(rows, cols)`` ``bf16`` / ``fp16``.
    rot_R : ``(cols // 32, 32, 32)`` ``bf16`` / ``fp16``. Set
        ``rot_R.transpose = True`` (instance attribute) or pass
        ``rot_transposed=True`` to switch to the ``[b, g, h]`` storage
        layout.
    sorted_ids, num_valid_ids, token_num
        Same semantics as :func:`aiter.ops.quant.mxfp4_moe_sort_fwd`.
    rot_transposed : optional ``bool`` override. When ``None`` (the
        default), the value is read from the ``rot_R.transpose``
        attribute.

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
