# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public launcher for the pinned vLLM AMD QSA expand."""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.qsa_vllm_amd import (
    VLLM_AMD_QSA_PIN,
    _expand_qsa_indices_kernel,
)


def _unit_stride_int32(
    name: str, tensor: torch.Tensor, length: int, device: torch.device
) -> torch.Tensor:
    """Return a contiguous int32 vector of ``length``.

    The expand kernel loads these as ``ptr + index`` and does not take a
    stride. A view such as ``base[::2]`` would otherwise read the neighboring
    element.
    """
    if tensor.dtype != torch.int32 or tensor.shape != (length,):
        raise ValueError(
            f"{name} must be int32 [{length}], got {tensor.dtype} {tuple(tensor.shape)}"
        )
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    return tensor.contiguous()


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if token_topk <= 0:
        raise ValueError("QSA token top-k must be positive")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    rows = block_indices.shape[0]
    query_positions = _unit_stride_int32(
        "query_positions", query_positions, rows, block_indices.device
    )
    token_to_req = _unit_stride_int32(
        "token_to_req", token_to_req, rows, block_indices.device
    )
    if sequence_lengths.dim() != 1:
        raise ValueError(
            "sequence_lengths must be a 1-D int32 vector, "
            f"got {sequence_lengths.dtype} {tuple(sequence_lengths.shape)}"
        )
    sequence_lengths = _unit_stride_int32(
        "sequence_lengths",
        sequence_lengths,
        sequence_lengths.shape[0],
        block_indices.device,
    )
    if block_indices.shape != (rows, block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if not block_indices.is_cuda:
        raise RuntimeError("QSA index expansion requires a GPU")
    if out is None:
        out = torch.empty(
            (rows, output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (rows, output_width) or out.dtype != torch.int32:
        raise ValueError(
            "QSA expand out must be int32 "
            f"[{rows}, {output_width}], got {out.dtype} {tuple(out.shape)}"
        )
    elif out.device != block_indices.device:
        raise ValueError(
            f"QSA expand out must be on {block_indices.device}, got {out.device}"
        )
    if not rows:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        sequence_lengths.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


__all__ = [
    "VLLM_AMD_QSA_PIN",
    "expand_qsa_block_indices_cuda",
]
