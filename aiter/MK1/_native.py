# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load the packaged MK1 native extension and verify its compiled ABI."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from .errors import NativeBinaryError

PACKAGE_ROOT = Path(__file__).resolve().parent
NATIVE_EXTENSION_NAME = "module_persistent_decoder"
NATIVE_EXTENSION_FILENAME = f"{NATIVE_EXTENSION_NAME}.so"
NATIVE_ABI_VERSION = 1


def _validate_native_extension(extension: ModuleType, binary: Path) -> None:
    actual_abi = getattr(extension, "native_abi_version", None)
    if actual_abi != NATIVE_ABI_VERSION:
        raise NativeBinaryError(
            f"native extension {binary} provides ABI {actual_abi!r}; "
            f"AITER MK1 requires ABI {NATIVE_ABI_VERSION}"
        )
    if not hasattr(extension, "PersistentDecoder"):
        raise NativeBinaryError(f"native extension {binary} has no PersistentDecoder")


def load_native_extension(*, extension_path: str | Path | None = None) -> ModuleType:
    """Load the packaged extension or an explicit development override."""

    configured = extension_path or os.environ.get("AITER_MK1_NATIVE_EXTENSION")
    binary = Path(configured or PACKAGE_ROOT / NATIVE_EXTENSION_FILENAME)
    binary = binary.expanduser().resolve()
    if not binary.is_file():
        raise NativeBinaryError(f"MK1 native extension is missing: {binary}")

    existing = sys.modules.get(NATIVE_EXTENSION_NAME)
    existing_file = getattr(existing, "__file__", None)
    if existing is not None and existing_file is not None:
        if Path(existing_file).resolve() == binary:
            _validate_native_extension(existing, binary)
            return existing

    spec = importlib.util.spec_from_file_location(NATIVE_EXTENSION_NAME, binary)
    if spec is None or spec.loader is None:
        raise NativeBinaryError(f"cannot create an import loader for {binary}")

    extension: ModuleType | None = None
    try:
        extension = importlib.util.module_from_spec(spec)
        sys.modules[NATIVE_EXTENSION_NAME] = extension
        spec.loader.exec_module(extension)
        _validate_native_extension(extension, binary)
    except Exception as error:  # noqa: BLE001
        if existing is not None:
            sys.modules[NATIVE_EXTENSION_NAME] = existing
        elif (
            extension is not None
            and sys.modules.get(NATIVE_EXTENSION_NAME) is extension
        ):
            sys.modules.pop(NATIVE_EXTENSION_NAME, None)
        if isinstance(error, NativeBinaryError):
            raise
        raise NativeBinaryError(
            f"cannot load native extension {binary}: {error}"
        ) from error
    return extension
