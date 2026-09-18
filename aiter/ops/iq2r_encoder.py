# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Importance-aware reference encoder for AITER's native-basis IQ2R format."""

from __future__ import annotations

import base64

import torch
from torch import Tensor

from .iq2r_format import (
    IQ2R_ATOMS_PER_TRIPLET,
    IQ2R_BASE_PADDING_BYTES,
    IQ2R_CODEBOOK_BYTES,
    IQ2R_CODEBOOK_ENTRIES,
    IQ2R_SCALE_BLOCK,
    IQ2R_TILE_K,
    IQ2R_TILE_N,
    IQ2R_VECTOR_SIZE,
    iq2r_packed_sizes,
    iq2r_physical_n_blocks,
)
from .iq2r_reference import (
    iq2r_metadata_word,
    iq2r_set_metadata_word,
    iq2r_tile_views,
)

_MXFP4_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# The standard IQ2_XXS positive-magnitude grid serialized as 256 eight-byte
# vectors.  Two E4M3 phases seed the 512-entry learned IQ2R codebook.
_IQ2_XXS_GRID_BYTES = base64.b64decode(
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgICCsICAgICCsIKwgICAgICCsrCAgICAgrKysICAgICBkICBkICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgrKwgrCAgICCsIKysICAgIGQgICBkICAgIGQgIGQgICAgIGQgZCAgIGRkZCBkICAgICAgZGQgICAgZCCsZCAgICCsZKxkICAgICAgIKwgICCsICAgrCAgIKwgrCCsICAgrCAgrKwgICBkICAgIGQgICBkICAgZCAgICBkICBkICBkIKwgIGQgICBkrCAgZCAgICAgZCBkICCsICBkIGQgICCsIGQgZCAgICCsZCBkICBkICCsIGQgICBkIKwgZCAgICBkrCBkICAgZKysIGQgICAgICBkZCAgrCAgIGRkICAgrCAgZGQgICAgrCBkZCAgrGQgZGRkICBkrKxkZGQgICAgIKxkZCAgZCBkrGRkICBkrCAgrGQgICAgZCCsZCAgICAgZKxkICAgZCCsrGQgICBkrKysZCAgICAgICCsICBkZCAgIKwgICCsICAgrCAgIGRkICCsICAgrKwgIKwgIGQgIGQgrCAgIGQgZCCsICAgIGRkIKwgIKwgZGQgrCAgIKwgrCCsICAgZCAgZKwgICAgIGRkrCAgrCAgIKysICAgZGQgrKwgIGQgICAgIGQgIGQgICAgZCAgIGQgICBkIGQgrCAgIGQgICAgZCAgZCAgIKxkICBkICBkIKwgIGQgICBkrCAgZCBkZGSsICBkICAgICBkIGQgIKwgIGQgZCAgIKwgZCBkICAgZGRkIGQgrKxkZGQgZCAgICCsZCBkICBkrCCsIGQgZGQgZKwgZCAgICAgIGRkICCsICAgZGQgICCsICBkZCBkZKwgIGRkIGSsIGQgZGQgICAgrCBkZCAgrGQgZGRkIKwgrGRkZGQgICAgIKxkZCCsZGQgrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgZCAgrCCsZCAgICAgZKxkIGRkICBkrGQgICCsrGSsZCBkIGRkrKxkICAgICAgIKwgrCAgICAgrCCsrCAgICCsICBkIGQgIKwgZCCsZCAgrCAgICCsICCsIKwgIKwgIKwgZKysIGQgrCAgrCBkZCCsICAgICCsIKwgrCAgIKwgrCBkICAgIGSsICBkICAgZKwgICBkICBkrCAgICBkIGSsIKxkZGQgZKwgICAgIGRkrCBkICBkZGSsICBkrGRkZKwgICBkrKxkrCAgrCAgIKysICAgrCAgrKwgIGRkrCCsrCAgZCBkrKysIGQgICAgICBkIGQgICAgIGQgIGQgICAgZCCsZCAgICBkZCCsICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGSsZGRkICAgZCAgrGQgICBkZCAgrCAgIGQgZCCsICAgZCAgZKwgICBkICAgIGQgIGQgIKwgZCAgZGQgrGRkICBkICAgrGQgIGRkZCCsZCAgZGQgICCsICBkICBkIKwgIGQgrCBkrCAgZKxkZGSsICBkIKysZKwgIGQgICAgIGQgZCCsICAgZCBkICCsICBkIGQgICCsIGQgZGSsZKwgZCBkrCBkIGRkIGQgZKwgZGQgZCAgICCsZCBkZCAgICCsIGQgZCAgIKwgZCAgZCAgrCBkICAgZCCsIGRkZCBkIKwgZCAgICBkrCBkIKxkZGSsIGRkIKxkZKwgZKwgIKxkrCBkZGQgZKysIGQgIGSsrKwgZCAgICAgIGRkIKwgICAgZGRkIGQgICBkZGSsZCAgIGRkICCsICAgZGQgICCsICBkZCCsIKwgIGRkIGQgIGQgZGSsICBkZCBkZCBkrKxkIGRkZCBkrKwgZGQgIGSsIGRkZKwgZKwgZGRkrKwgIGRkZGRkICAgrGRkZCBkZGSsZGRkICAgICCsZGRkIGQgIKxkZGSsZCAgrGRkIGSsZCCsZGQgICBkZKxkZCCsICCsrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkIKysZCAgrGQgICAgZCCsZGRkZGRkIKxkIKxkIKwgrGQgIKxkrCCsZCAgICAgZKxkZGQgICBkrGQgIGQgZGSsZKwgZCBkZKxkIGQgrGRkrGSsICBkIKysZCAgICAgICCsrCAgICAgIKysrCAgICAgrGQgIGQgICCsrCAgrCAgIKwgZCAgZCAgrCCsZCBkICCsICAgZGQgIKxkIGQgrCAgrGQgICAgZCCsIGQgICBkIKwgIGQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrCAgICBkZCCsrGQgZGRkIKwgZGSsZGQgrGSsICCsZCCsICAgZKxkIKwgIKxkrGQgrKwgICAgrCCsIGQgIGSsIKxkIGQgrKwgrCBkICAgIGSsICBkICAgZKwgZKwgICBkrCAgIGQgIGSsZCCsrCAgZKysZGQgZCBkrCAgIKxkIGSsZGQgZKwgZKwgICAgIGRkrKwgrCAgZGSsIGQgZCBkZKxkIGRkZGRkrGQgIKwgrGSsICCsIGSsZKysICAgICCsrCAgZGQgIKysZGQgrCAgrKxkrCAgZCCsrCAgICCsIKysIKxkICBkrKwgIGRkIKysrCBkICBkrKys="
)


def _pad_k(values: Tensor, padded_k: int, value: float = 0.0) -> Tensor:
    if values.shape[-1] > padded_k:
        raise ValueError(f"cannot pad K={values.shape[-1]} to smaller K={padded_k}")
    if values.shape[-1] == padded_k:
        return values
    return torch.nn.functional.pad(
        values, (0, padded_k - values.shape[-1]), value=value
    )


def iq2r_dequantize_mxfp4(
    blocks: Tensor,
    scales: Tensor,
    *,
    output_dtype: torch.dtype = torch.float16,
) -> Tensor:
    """Dequantize GPT-OSS native MXFP4 weights without changing row order.

    ``blocks`` has shape ``[..., groups, packed_values]`` and stores the low
    FP4 nibble before the high nibble. ``scales`` has the matching
    ``[..., groups]`` E8M0 exponent bytes.  GPT-OSS gate/up rows are already
    adjacent ``[gate_0, up_0, gate_1, up_1, ...]`` in the source checkpoint;
    this routine deliberately performs no gate/up permutation.
    """

    if not isinstance(blocks, Tensor) or not isinstance(scales, Tensor):
        raise TypeError("blocks and scales must be torch.Tensor instances")
    if blocks.dtype != torch.uint8:
        raise TypeError(f"blocks must have dtype uint8, got {blocks.dtype}")
    if scales.dtype != torch.uint8:
        raise TypeError(f"scales must have dtype uint8, got {scales.dtype}")
    if blocks.ndim < 2:
        raise ValueError(
            "blocks must have shape [..., groups, packed_values], "
            f"got {tuple(blocks.shape)}"
        )
    if tuple(scales.shape) != tuple(blocks.shape[:-1]):
        raise ValueError(
            f"scales shape must be {tuple(blocks.shape[:-1])}, "
            f"got {tuple(scales.shape)}"
        )
    if blocks.device != scales.device:
        raise ValueError("blocks and scales must be on the same device")
    if not output_dtype.is_floating_point:
        raise TypeError(f"output_dtype must be floating point, got {output_dtype}")

    values = torch.tensor(_MXFP4_E2M1_VALUES, device=blocks.device, dtype=output_dtype)
    unpacked = torch.empty((*blocks.shape, 2), device=blocks.device, dtype=output_dtype)
    unpacked[..., 0] = values[(blocks & 0x0F).to(torch.int64)]
    unpacked[..., 1] = values[(blocks >> 4).to(torch.int64)]
    unpacked = unpacked.reshape(*blocks.shape[:-2], blocks.shape[-2], -1)
    exponents = scales.to(torch.int32) - 127
    torch.ldexp(unpacked, exponents.unsqueeze(-1), out=unpacked)
    return unpacked.reshape(*blocks.shape[:-2], -1)


def iq2r_scale_blocks(values: Tensor) -> Tensor:
    """Map row-major values to scaled-MFMA scale-lane order."""

    n, k = values.shape
    if n % IQ2R_TILE_N or k % IQ2R_TILE_K:
        raise ValueError(f"IQ2R scale blocks require N%16==0,K%128==0, got {n}x{k}")
    dense = values.reshape(n, k // IQ2R_TILE_K, 4, IQ2R_SCALE_BLOCK)
    return (
        dense.reshape(n // IQ2R_TILE_N, IQ2R_TILE_N, k // IQ2R_TILE_K, 4, 32)
        .permute(0, 2, 3, 1, 4)
        .reshape(n // IQ2R_TILE_N, k // IQ2R_TILE_K, 64, 32)
        .contiguous()
    )


def _scale_importance(importance: Tensor, n: int) -> Tensor:
    return iq2r_scale_blocks(importance.reshape(1, -1).expand(n, -1))


def iq2r_initial_codebook(device: torch.device | str = "cpu") -> Tensor:
    values = torch.tensor(
        list(_IQ2_XXS_GRID_BYTES), dtype=torch.float32, device=device
    ).reshape(256, IQ2R_VECTOR_SIZE)
    base = values * (1.0 / 8.0)
    codebook = torch.cat((base, base * 1.5), dim=0).to(torch.float8_e4m3fn).float()
    codebook[0].zero_()
    return codebook.contiguous()


def iq2r_reserve_zero_codeword(codebook: Tensor) -> Tensor:
    if tuple(codebook.shape) != (IQ2R_CODEBOOK_ENTRIES, IQ2R_VECTOR_SIZE):
        raise ValueError("IQ2R codebook must have shape [512,8]")
    result = codebook.clone()
    result[0].zero_()
    return result


def _assign_codewords(
    vectors: Tensor,
    vector_importance: Tensor,
    codebook: Tensor,
    *,
    chunk: int = 4096,
) -> tuple[Tensor, Tensor]:
    indices = torch.empty(vectors.shape[0], dtype=torch.int64, device=vectors.device)
    errors = torch.empty(vectors.shape[0], dtype=torch.float32, device=vectors.device)
    for start in range(0, vectors.shape[0], chunk):
        stop = min(start + chunk, vectors.shape[0])
        delta = vectors[start:stop, None, :] - codebook[None, :, :]
        distance = (delta.square() * vector_importance[start:stop, None, :]).sum(dim=-1)
        errors[start:stop], indices[start:stop] = distance.min(dim=-1)
    return indices, errors


def _e8m0_candidates(fragments: Tensor, codebook_max: Tensor, radius: int) -> Tensor:
    maximum = fragments.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    center = torch.round(torch.log2(maximum / codebook_max.clamp_min(1e-30))).to(
        torch.int32
    )
    offsets = torch.arange(
        -radius, radius + 1, device=fragments.device, dtype=torch.int32
    )
    return (center[..., None] + offsets).clamp(-126, 127)


@torch.no_grad()
def iq2r_learn_codebook(
    weight: Tensor,
    importance: Tensor,
    *,
    iterations: int = 4,
    sample_vectors: int = 65536,
    seed: int = 0x10A0,
) -> Tensor:
    """Learn one non-negative 512x8 E4M3 codebook for an expert matrix."""

    if weight.ndim != 2 or importance.shape != (weight.shape[1],):
        raise ValueError("weight and importance must have shapes [N,K] and [K]")
    padded_k = ((weight.shape[1] + IQ2R_TILE_K - 1) // IQ2R_TILE_K) * IQ2R_TILE_K
    padded_weight = _pad_k(weight.float(), padded_k)
    padded_importance = _pad_k(importance.float(), padded_k)
    fragments = iq2r_scale_blocks(padded_weight).reshape(-1, 4, IQ2R_VECTOR_SIZE)
    objective_weight = _scale_importance(padded_importance, weight.shape[0]).reshape(
        -1, 4, IQ2R_VECTOR_SIZE
    )
    sample_blocks = max(1, sample_vectors // 4)
    if fragments.shape[0] > sample_blocks:
        generator = torch.Generator(device=weight.device).manual_seed(seed)
        chosen = torch.randperm(
            fragments.shape[0], generator=generator, device=weight.device
        )[:sample_blocks]
        fragments = fragments[chosen]
        objective_weight = objective_weight[chosen]

    codebook = iq2r_initial_codebook(weight.device)
    for _ in range(iterations):
        block_max = fragments.abs().amax(dim=(-1, -2)).clamp_min(1e-30)
        center = torch.round(torch.log2(block_max / codebook.max().clamp_min(1e-30)))
        scales = torch.exp2(center)
        vectors = (fragments.abs() / scales[:, None, None]).reshape(
            -1, IQ2R_VECTOR_SIZE
        )
        vector_importance = objective_weight.reshape(-1, IQ2R_VECTOR_SIZE)
        indices, _ = _assign_codewords(vectors, vector_importance, codebook)
        numerator = torch.zeros_like(codebook)
        denominator = torch.zeros_like(codebook)
        scatter = indices[:, None].expand(-1, IQ2R_VECTOR_SIZE)
        numerator.scatter_add_(0, scatter, vectors * vector_importance)
        denominator.scatter_add_(0, scatter, vector_importance)
        updated = numerator / denominator.clamp_min(1e-12)
        occupied = denominator.sum(dim=-1) > 0
        occupied[0] = False
        codebook = torch.where(occupied[:, None], updated, codebook)
        codebook = codebook.clamp(0.0, 448.0).to(torch.float8_e4m3fn).float()
        codebook[0].zero_()
    return codebook.contiguous()


@torch.no_grad()
def iq2r_encode_reference(
    weight: Tensor,
    importance: Tensor,
    codebook: Tensor,
    *,
    exponent_radius: int = 0,
) -> tuple[Tensor, Tensor]:
    """Encode one matrix into the exact IQ2R production byte layout."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape [N,K]")
    n, k = weight.shape
    if n % IQ2R_TILE_N or k % IQ2R_SCALE_BLOCK:
        raise ValueError("IQ2R requires N%16==0 and K%32==0")
    if importance.shape != (k,):
        raise ValueError(f"importance must have shape [{k}]")
    if not 0 <= exponent_radius <= 16:
        raise ValueError("exponent_radius must be in [0,16]")
    codebook = iq2r_reserve_zero_codeword(codebook).float()
    data_bytes, auxiliary_bytes = iq2r_packed_sizes(n, k)
    padded_k = ((k + IQ2R_TILE_K - 1) // IQ2R_TILE_K) * IQ2R_TILE_K
    padded_weight = _pad_k(weight.float(), padded_k)
    padded_importance = _pad_k(importance.float(), padded_k)
    fragments = iq2r_scale_blocks(padded_weight)
    objective_weight = _scale_importance(padded_importance, n)
    candidates = _e8m0_candidates(fragments, codebook.max(), exponent_radius)
    block_shape = fragments.shape[:-1]
    best_error = torch.full(block_shape, torch.inf, device=weight.device)
    best_exp = torch.zeros(block_shape, dtype=torch.int32, device=weight.device)
    best_indices = torch.zeros(
        (*block_shape, 4), dtype=torch.int64, device=weight.device
    )

    vectors = fragments.reshape(-1, 4, IQ2R_VECTOR_SIZE)
    vector_importance = objective_weight.reshape(-1, 4, IQ2R_VECTOR_SIZE)
    candidate_exp = candidates.reshape(-1, candidates.shape[-1])
    for candidate in range(candidate_exp.shape[-1]):
        exponent = candidate_exp[:, candidate]
        scale = torch.exp2(exponent.float())
        normalized = vectors.abs() / scale[:, None, None]
        group_indices = []
        group_errors = []
        for group in range(4):
            indices, error = _assign_codewords(
                normalized[:, group], vector_importance[:, group], codebook
            )
            group_indices.append(indices)
            group_errors.append(error * scale.square())
        total = torch.stack(group_errors, dim=-1).sum(dim=-1)
        improve = total < best_error.reshape(-1)
        best_error.reshape(-1)[improve] = total[improve]
        best_exp.reshape(-1)[improve] = exponent[improve]
        stacked = torch.stack(group_indices, dim=-1)
        best_indices.reshape(-1, 4)[improve] = stacked[improve]

    signs = torch.zeros((*block_shape, 4), dtype=torch.uint8, device=weight.device)
    sign_bits = fragments.signbit().reshape(*block_shape, 4, IQ2R_VECTOR_SIZE)
    for bit in range(IQ2R_VECTOR_SIZE):
        signs |= sign_bits[..., bit].to(torch.uint8) << bit

    data = torch.zeros(data_bytes, dtype=torch.uint8)
    logical_tile_count = (n // IQ2R_TILE_N) * (padded_k // IQ2R_TILE_K)
    block_indices = best_indices.reshape(logical_tile_count, 4, 16, 4)
    block_signs = signs.reshape(logical_tile_count, 4, 16, 4)
    physical_indices = torch.empty_like(block_indices)
    physical_signs = torch.empty_like(block_signs)
    for group in range(4):
        half = group % 2
        low_block = group // 2
        physical_indices[:, group, :, :2] = block_indices[
            :, low_block, :, half * 2 : half * 2 + 2
        ]
        physical_indices[:, group, :, 2:] = block_indices[
            :, low_block + 2, :, half * 2 : half * 2 + 2
        ]
        physical_signs[:, group, :, :2] = block_signs[
            :, low_block, :, half * 2 : half * 2 + 2
        ]
        physical_signs[:, group, :, 2:] = block_signs[
            :, low_block + 2, :, half * 2 : half * 2 + 2
        ]
    indices_cpu = physical_indices.reshape(logical_tile_count, 64, 4).cpu()
    signs_cpu = physical_signs.reshape(logical_tile_count, 64, 4).cpu()
    k_tiles = padded_k // IQ2R_TILE_K
    n_blocks = n // IQ2R_TILE_N
    for n_block in range(n_blocks):
        atom = n_block % IQ2R_ATOMS_PER_TRIPLET
        for k_tile in range(k_tiles):
            logical_tile = n_block * k_tiles + k_tile
            records, atom_metadata = iq2r_tile_views(data, n_block, k_tile, k_tiles)
            records[:, :4] = indices_cpu[logical_tile].to(torch.uint8)
            records[:, 4:] = signs_cpu[logical_tile]
            for lane in range(64):
                high_nibble = 0
                for codeword in range(4):
                    high_nibble |= (
                        (int(indices_cpu[logical_tile, lane, codeword]) >> 8) & 1
                    ) << codeword
                word = iq2r_metadata_word(atom_metadata, lane)
                iq2r_set_metadata_word(
                    atom_metadata, lane, word | (high_nibble << (atom * 8))
                )

    logical_scales = (
        (best_exp + 127)
        .clamp(0, 254)
        .to(torch.uint8)
        .cpu()
        .reshape(logical_tile_count, 64)
    )
    valid_blocks = k // IQ2R_SCALE_BLOCK
    base_exponents = torch.full(
        (iq2r_physical_n_blocks(n) + IQ2R_BASE_PADDING_BYTES,),
        127,
        dtype=torch.uint8,
    )
    for n_block in range(n_blocks):
        valid_exponents = []
        for block_k in range(valid_blocks):
            logical_tile = n_block * k_tiles + block_k // 4
            logical_block = block_k % 4
            valid_exponents.append(
                logical_scales[
                    logical_tile,
                    logical_block * 16 : (logical_block + 1) * 16,
                ]
            )
        base_exponent = int(torch.cat(valid_exponents).min())
        base_exponents[n_block] = base_exponent
        for k_tile in range(k_tiles):
            logical_tile = n_block * k_tiles + k_tile
            _, atom_metadata = iq2r_tile_views(data, n_block, k_tile, k_tiles)
            atom = n_block % IQ2R_ATOMS_PER_TRIPLET
            for lane in range(64):
                block_k = k_tile * 4 + lane // 16
                if block_k < valid_blocks:
                    delta = int(logical_scales[logical_tile, lane]) - base_exponent
                    if delta > 15:
                        raise ValueError(
                            "IQ2R scale exponent range exceeds the 4-bit delta "
                            f"format for output block {n_block}: delta={delta}"
                        )
                    word = iq2r_metadata_word(atom_metadata, lane)
                    iq2r_set_metadata_word(
                        atom_metadata,
                        lane,
                        word | (max(delta, 0) << (atom * 8 + 4)),
                    )

    auxiliary = torch.empty(auxiliary_bytes, dtype=torch.uint8)
    auxiliary[:IQ2R_CODEBOOK_BYTES] = (
        codebook.to(torch.float8_e4m3fn).view(torch.uint8).cpu().reshape(-1)
    )
    auxiliary[IQ2R_CODEBOOK_BYTES:] = base_exponents
    return data.to(weight.device), auxiliary.to(weight.device)


__all__ = [
    "iq2r_dequantize_mxfp4",
    "iq2r_encode_reference",
    "iq2r_initial_codebook",
    "iq2r_learn_codebook",
    "iq2r_reserve_zero_codeword",
    "iq2r_scale_blocks",
]
