# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validation and loading for compiled MK1 persistent checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import MK1Config, SUPPORTED_BACKEND
from .errors import CheckpointError

_MANIFEST_CANDIDATES = (
    "persistent_decoder_manifest.json",
    "manifest.json",
    "config.json",
)
_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
}
_TORCH_DTYPE_NAMES = {
    torch.bool: "BOOL",
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
}
if hasattr(torch, "float8_e4m3fn"):
    _TORCH_DTYPE_NAMES[torch.float8_e4m3fn] = "F8_E4M3"

PERSISTENT_LAYER_COUNT = 36
PERSISTENT_BINDING_COUNT = 471
def persistent_binding_order() -> tuple[str, ...]:
    """Return the public native ABI order for GPT-OSS-120B MK1 weights."""

    names = ["model.embed_tokens.persistent_decoder_weight"]
    for index in range(PERSISTENT_LAYER_COUNT):
        prefix = f"model.layers.{index}"
        names.extend(
            (
                f"{prefix}.input_layernorm.persistent_decoder_weight",
                f"{prefix}.self_attn.qkv_proj.0.persistent_decoder_weight",
                f"{prefix}.self_attn.qkv_proj.0.persistent_decoder_bias",
                f"{prefix}.self_attn.sinks",
                f"{prefix}.self_attn.o_proj.0.persistent_decoder_weight",
                f"{prefix}.self_attn.o_proj.persistent_decoder_bias",
                f"{prefix}.post_attention_layernorm.persistent_decoder_weight",
                f"{prefix}.mlp.router.persistent_decoder_weight",
                f"{prefix}.mlp.router.persistent_decoder_bias",
                f"{prefix}.mlp.experts.gate_up_proj.persistent_decoder_weight",
                f"{prefix}.mlp.experts.gate_up_proj.persistent_decoder_bias",
                f"{prefix}.mlp.experts.down_proj.persistent_decoder_weight",
                f"{prefix}.mlp.experts.down_proj.persistent_decoder_bias",
            )
        )
    names.extend(
        (
            "model.norm.persistent_decoder_weight",
            "lm_head.persistent_decoder_weight",
        )
    )
    result = tuple(names)
    if len(result) != PERSISTENT_BINDING_COUNT:
        raise AssertionError("MK1 persistent binding order is internally inconsistent")
    return result


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    model_family: str | None
    model_revision: str | None
    backend: str
    checkpoint_backend: str
    checkpoint_mode: str | None
    tensor_count: int
    persistent_bytes: int
    shard_files: tuple[Path, ...]
    tensor_records: tuple[dict[str, Any], ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "backend": self.backend,
            "checkpoint_backend": self.checkpoint_backend,
            "checkpoint_mode": self.checkpoint_mode,
            "tensor_count": self.tensor_count,
            "persistent_bytes": self.persistent_bytes,
            "shards": [str(path) for path in self.shard_files],
        }


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot read checkpoint manifest {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CheckpointError(f"checkpoint manifest {path} must contain a JSON object")
    return value, payload


def _first_string(mapping: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str) and value:
            return value
    return None


def _backend_manifest(
    manifest: dict[str, Any], backend: str
) -> tuple[dict[str, Any], str]:
    backends = manifest.get("persistent_decoder_backends")
    if isinstance(backends, dict):
        selected = backends.get(backend)
        if isinstance(selected, dict):
            return selected, backend
        raise CheckpointError(
            f"checkpoint does not declare backend {backend}"
        )
    declared = _first_string(manifest, "backend", "backend_abi")
    if declared is not None and declared != backend:
        raise CheckpointError(
            f"checkpoint backend {declared} does not match requested backend {backend}"
        )
    return manifest, declared or backend


def _persistent_records(
    selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layouts = selected.get("layouts")
    if isinstance(layouts, dict):
        section = layouts.get("persistent")
        if not isinstance(section, dict):
            raise CheckpointError("checkpoint has no persistent tensor layout")
    else:
        section = selected
    records = section.get("tensors", section.get("tensor_catalog"))
    if not isinstance(records, list) or not records:
        raise CheckpointError("persistent checkpoint has no tensor catalog")
    if not all(isinstance(record, dict) for record in records):
        raise CheckpointError("persistent tensor catalog entries must be objects")
    return records, section


def _validated_record(root: Path, record: dict[str, Any]) -> tuple[str, Path, int]:
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise CheckpointError("persistent tensor record has an invalid name")
    shard_name = record.get("shard")
    if not isinstance(shard_name, str) or not shard_name:
        raise CheckpointError(f"persistent tensor {name} has no shard")
    shard = (root / shard_name).resolve()
    if shard.parent != root or not shard.is_file():
        raise CheckpointError(
            f"persistent tensor {name} has missing shard {shard_name}"
        )
    dtype = record.get("dtype")
    shape = record.get("shape")
    byte_size = record.get("byte_size")
    if dtype not in _DTYPE_BYTES:
        raise CheckpointError(
            f"persistent tensor {name} has unsupported dtype {dtype!r}"
        )
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or value < 0 for value in shape
    ):
        raise CheckpointError(f"persistent tensor {name} has an invalid shape")
    expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
    if byte_size != expected_bytes:
        raise CheckpointError(
            f"persistent tensor {name} byte size is {byte_size!r}, expected {expected_bytes}"
        )
    content_hash = record.get("content_sha256")
    if content_hash is not None and (
        not isinstance(content_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", content_hash)
    ):
        raise CheckpointError(f"persistent tensor {name} has an invalid content hash")
    return name, shard, expected_bytes


def validate_checkpoint(path: str | Path, config: MK1Config) -> CheckpointInfo:
    """Validate one local compiled checkpoint without performing network I/O."""

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointError(f"checkpoint path is not a directory: {root}")
    manifest_path = next(
        (root / name for name in _MANIFEST_CANDIDATES if (root / name).is_file()),
        None,
    )
    if manifest_path is None:
        raise CheckpointError(
            f"checkpoint has none of the supported manifests: {', '.join(_MANIFEST_CANDIDATES)}"
        )
    manifest, payload = _load_json(manifest_path)
    selected, checkpoint_backend = _backend_manifest(manifest, config.backend)
    format_version = selected.get("format_version")
    if format_version is not None and format_version not in (2, 3):
        raise CheckpointError(
            f"unsupported persistent checkpoint format {format_version!r}"
        )

    model_family = _first_string(selected, "model_family", "model") or _first_string(
        manifest, "model_family", "model"
    )
    model_revision = _first_string(
        selected, "model_revision", "source_revision", "revision"
    ) or _first_string(manifest, "model_revision", "source_revision", "revision")
    if (
        model_family is not None
        and model_family.lower().replace("_", "-") != config.model_family
    ):
        raise CheckpointError(
            f"checkpoint model family {model_family} does not match {config.model_family}"
        )
    if model_revision is not None and model_revision != config.model_revision:
        raise CheckpointError(
            f"checkpoint model revision {model_revision} does not match {config.model_revision}"
        )

    records, section = _persistent_records(selected)
    expected_order = persistent_binding_order()
    by_name: dict[str, dict[str, Any]] = {}
    shard_files: set[Path] = set()
    persistent_bytes = 0
    for record in records:
        name, shard, byte_size = _validated_record(root, record)
        if name in by_name:
            raise CheckpointError(f"persistent tensor catalog duplicates {name}")
        by_name[name] = record
        shard_files.add(shard)
        persistent_bytes += byte_size

    missing = [name for name in expected_order if name not in by_name]
    unexpected = sorted(set(by_name) - set(expected_order))
    if missing or unexpected:
        raise CheckpointError(
            "persistent tensor catalog differs from the 471-binding ABI: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    declared_count = section.get("binding_count", section.get("tensor_count"))
    if declared_count is not None and int(declared_count) != PERSISTENT_BINDING_COUNT:
        raise CheckpointError(
            f"persistent binding count is {declared_count}, expected {PERSISTENT_BINDING_COUNT}"
        )
    declared_bytes = section.get("byte_size")
    if declared_bytes is not None and int(declared_bytes) != persistent_bytes:
        raise CheckpointError(
            f"persistent byte size is {declared_bytes}, expected {persistent_bytes}"
        )

    return CheckpointInfo(
        root=root,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=hashlib.sha256(payload).hexdigest(),
        model_family=model_family,
        model_revision=model_revision,
        backend=config.backend,
        checkpoint_backend=checkpoint_backend,
        checkpoint_mode=_first_string(selected, "checkpoint_mode"),
        tensor_count=PERSISTENT_BINDING_COUNT,
        persistent_bytes=persistent_bytes,
        shard_files=tuple(sorted(shard_files)),
        tensor_records=tuple(by_name[name] for name in expected_order),
    )


def persistent_checkpoint_bytes(
    path: str | Path, *, backend: str = SUPPORTED_BACKEND
) -> int:
    """Return the exact persistent tensor payload declared by a checkpoint."""

    return validate_checkpoint(path, MK1Config(backend=backend)).persistent_bytes


def load_persistent_weights(
    info: CheckpointInfo, *, device: torch.device | str | int
) -> list[torch.Tensor]:
    """Load, verify, and order all persistent tensors on the target device."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise CheckpointError(
            "loading an MK1 checkpoint requires safetensors"
        ) from error

    by_shard: dict[Path, list[dict[str, Any]]] = {}
    for record in info.tensor_records:
        by_shard.setdefault((info.root / str(record["shard"])).resolve(), []).append(
            record
        )

    loaded: dict[str, torch.Tensor] = {}
    for shard, records in by_shard.items():
        try:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for record in records:
                    name = str(record["name"])
                    if name not in available:
                        raise CheckpointError(
                            f"checkpoint shard {shard.name} is missing {name}"
                        )
                    tensor = handle.get_tensor(name).contiguous()
                    dtype_name = _TORCH_DTYPE_NAMES.get(tensor.dtype)
                    if dtype_name != record["dtype"]:
                        raise CheckpointError(
                            f"persistent tensor {name} dtype is {dtype_name}, expected {record['dtype']}"
                        )
                    if list(tensor.shape) != record["shape"]:
                        raise CheckpointError(
                            f"persistent tensor {name} shape is {list(tensor.shape)}, "
                            f"expected {record['shape']}"
                        )
                    content_hash = record.get("content_sha256")
                    if content_hash is not None:
                        byte_view = tensor.view(torch.uint8).reshape(-1).numpy()
                        actual_hash = hashlib.sha256(byte_view).hexdigest()
                        if actual_hash != content_hash:
                            raise CheckpointError(
                                f"persistent tensor {name} content SHA-256 mismatch"
                            )
                    loaded[name] = tensor.to(device=device).contiguous()
        except CheckpointError:
            raise
        except Exception as error:  # noqa: BLE001
            raise CheckpointError(
                f"cannot load persistent shard {shard}: {error}"
            ) from error
    return [loaded[name] for name in persistent_binding_order()]
