# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validated Python owner for the optional MK1 native decoder handle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

import torch

from ._native import load_native_extension
from .checkpoint import (
    CheckpointInfo,
    load_persistent_weights,
    persistent_checkpoint_bytes,
    validate_checkpoint,
)
from .config import MK1Config, is_supported
from .errors import BackendLaunchError, CheckpointError, PrelaunchError

ATOM_PHYSICAL_PAGE_SIZE = 16


@dataclass(frozen=True, slots=True)
class AtomCacheBinding:
    """Zero-copy references and geometry for ATOM's shuffled K/V planes."""

    key_planes: tuple[torch.Tensor, ...]
    value_planes: tuple[torch.Tensor, ...]
    block_counts: tuple[int, ...]
    block_strides: tuple[int, ...]
    pools: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuantumRequest:
    """One frozen persistent-decode command from the serving scheduler."""

    pending_token: int
    committed_kv_length: int
    max_sequence_length: int
    max_tokens: int
    eos_token_id: int
    ignore_eos: bool
    full_block_map: torch.Tensor
    rope_cosine: torch.Tensor
    rope_sine: torch.Tensor
    sliding_block_map: torch.Tensor | None = None
    cancellation_flag: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class QuantumResult:
    finish_reason: int
    emitted_tokens: tuple[int, ...]
    committed_kv_length: int
    pending_token: int
    completed_epochs: int


def write_atom_shuffled_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write K/V into ATOM's native 16-token SHUFFLE cache layout."""

    if key_cache.element_size() != 2 or value_cache.element_size() != 2:
        raise ValueError("ATOM SHUFFLE cache writer requires a two-byte scalar")
    if key_cache.dtype != value_cache.dtype:
        raise ValueError("K and V cache dtypes must match")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError("key and value must have [token, head, dim] shape")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if slot_mapping.numel() != key.shape[0]:
        raise ValueError("slot mapping length must match the token count")

    pack = 16 // key_cache.element_size()
    if key.shape[2] % pack:
        raise ValueError("head dimension must be divisible by the cache pack width")

    slots = slot_mapping.to(device=key.device, dtype=torch.int64)
    blocks = torch.div(slots, ATOM_PHYSICAL_PAGE_SIZE, rounding_mode="floor")
    tokens = torch.remainder(slots, ATOM_PHYSICAL_PAGE_SIZE)
    key_rows = key.to(key_cache.dtype).reshape(
        key.shape[0], key.shape[1], key.shape[2] // pack, pack
    )
    key_cache[blocks, :, :, tokens, :] = key_rows
    token_groups = torch.div(tokens, pack, rounding_mode="floor")
    token_lanes = torch.remainder(tokens, pack)
    value_cache[blocks, :, token_groups, :, token_lanes] = value.to(value_cache.dtype)


write_atom_fp16_cache = write_atom_shuffled_cache


class PersistentDecoder:
    """Own one compiled checkpoint, native decoder, and its bound GPU state."""

    def __init__(
        self,
        *,
        device: int,
        max_sequence_length: int,
        cache_scalar: str,
        batch_size: int,
        backend: str,
        extension: ModuleType | None = None,
    ) -> None:
        config = MK1Config(
            device=device,
            max_sequence_length=max_sequence_length,
            cache_scalar=cache_scalar,
            batch_size=batch_size,
            backend=backend,
        )
        is_supported(config).require()
        provider = extension or load_native_extension()
        self.config = config
        self._checkpoint_info: CheckpointInfo | None = None
        self._checkpoint_weights: list[torch.Tensor] = []
        self._closed = False
        try:
            self.native = provider.PersistentDecoder(
                device=device,
                max_sequence_length=max_sequence_length,
                cache_scalar=cache_scalar,
                batch_size=batch_size,
                backend=backend,
            )
        except Exception as error:  # noqa: BLE001
            raise PrelaunchError(
                f"cannot create MK1 native decoder: {error}"
            ) from error

    @classmethod
    def from_checkpoint(
        cls,
        config: MK1Config,
        checkpoint_path: str | Path,
        *,
        extension: ModuleType | None = None,
    ) -> "PersistentDecoder":
        """Create a decoder and load only its persistent tensor catalog."""

        is_supported(config).require()
        info = validate_checkpoint(checkpoint_path, config)
        decoder = cls(
            device=config.device,
            max_sequence_length=config.max_sequence_length,
            cache_scalar=config.cache_scalar,
            batch_size=config.batch_size,
            backend=config.backend,
            extension=extension,
        )
        try:
            target_device = (
                torch.device("cuda", config.device)
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            weights = load_persistent_weights(info, device=target_device)
            decoder.bind_weights(weights)
        except Exception as error:  # noqa: BLE001
            decoder.close()
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError(
                f"persistent checkpoint loading failed: {error}"
            ) from error
        decoder._checkpoint_weights = weights
        decoder._checkpoint_info = info
        return decoder

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise PrelaunchError("persistent decoder is closed")

    def bind_atom_cache(self, binding: AtomCacheBinding) -> None:
        self._require_open()
        lengths = {
            len(binding.key_planes),
            len(binding.value_planes),
            len(binding.block_counts),
            len(binding.block_strides),
            len(binding.pools),
        }
        if len(lengths) != 1 or not binding.key_planes:
            raise ValueError("ATOM cache binding arrays must have equal nonzero length")
        for index, plane in enumerate((*binding.key_planes, *binding.value_planes)):
            if (
                plane.dtype != torch.uint8
                or plane.ndim != 1
                or not plane.is_contiguous()
            ):
                raise ValueError(
                    f"ATOM cache plane {index} must be contiguous one-dimensional uint8"
                )
        if any(int(count) <= 0 for count in binding.block_counts):
            raise ValueError("ATOM cache block counts must be positive")
        if any(int(stride) <= 0 for stride in binding.block_strides):
            raise ValueError("ATOM cache block strides must be positive")
        try:
            self.native.bind_atom_cache(
                list(binding.key_planes),
                list(binding.value_planes),
                [int(value) for value in binding.block_counts],
                [int(value) for value in binding.block_strides],
                [int(value) for value in binding.pools],
            )
        except Exception as error:  # noqa: BLE001
            raise PrelaunchError(f"native cache binding failed: {error}") from error

    bind_cache = bind_atom_cache

    def bind_weights(self, weights: Iterable[torch.Tensor]) -> None:
        """Internal checkpoint-loader primitive for the current native ABI."""

        self._require_open()
        tensors = list(weights)
        if not tensors:
            raise ValueError("persistent decoder requires bound weights")
        try:
            self.native.bind_weights(tensors)
        except Exception as error:  # noqa: BLE001
            raise PrelaunchError(f"native weight binding failed: {error}") from error

    def run_quantum(self, request: QuantumRequest) -> QuantumResult:
        self._require_open()
        if request.full_block_map.dtype != torch.int32:
            raise ValueError("full block map must use torch.int32")
        if (
            request.full_block_map.ndim != 1
            or not request.full_block_map.is_contiguous()
        ):
            raise ValueError("full block map must be contiguous and one-dimensional")
        try:
            result = self.native.run_quantum(
                pending_token=int(request.pending_token),
                committed_kv_length=int(request.committed_kv_length),
                max_sequence_length=int(request.max_sequence_length),
                max_tokens=int(request.max_tokens),
                eos_token_id=int(request.eos_token_id),
                ignore_eos=bool(request.ignore_eos),
                full_block_map=request.full_block_map,
                sliding_block_map=request.sliding_block_map,
                cancellation_flag=request.cancellation_flag,
                rope_cosine=request.rope_cosine,
                rope_sine=request.rope_sine,
            )
        except Exception as error:  # noqa: BLE001
            raise BackendLaunchError(
                f"native persistent quantum failed: {error}"
            ) from error
        emitted = tuple(int(token) for token in result.emitted_tokens)
        output = QuantumResult(
            finish_reason=int(result.finish_reason),
            emitted_tokens=emitted,
            committed_kv_length=int(result.committed_kv_length),
            pending_token=int(result.pending_token),
            completed_epochs=int(getattr(result, "completed_epochs", len(emitted))),
        )
        return output

    def checkpoint_info(self) -> dict[str, object] | None:
        return self._checkpoint_info.public_dict() if self._checkpoint_info else None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.native.close()
        finally:
            self._checkpoint_weights.clear()
            self._closed = True

    def __enter__(self) -> "PersistentDecoder":
        self._require_open()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


__all__ = [
    "AtomCacheBinding",
    "PersistentDecoder",
    "QuantumRequest",
    "QuantumResult",
    "persistent_checkpoint_bytes",
    "write_atom_fp16_cache",
    "write_atom_shuffled_cache",
]
