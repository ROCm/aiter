# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import logging

from .chip_info import get_asic_revision, get_gfx_runtime

logger = logging.getLogger("aiter")

_A0_ALLOWLIST: frozenset[str] = frozenset()


def _probe_arch_is_gfx1250() -> bool:
    # Arch undeterminable -> don't block; the C++ gate is authoritative for A0.
    try:
        return get_gfx_runtime() == "gfx1250"
    except Exception:  # noqa: BLE001
        return False


def _probe_stepping_ok() -> bool:
    if not _probe_arch_is_gfx1250():
        return True
    try:
        return get_asic_revision() >= 1
    except Exception as e:  # noqa: BLE001
        # Arch is gfx1250 here, so an unknown stepping may be A0: fail closed.
        # Mirrors require_gfx1250_asm_or_throw() in csrc/include/aiter_hip_common.h,
        # which likewise only lets an undeterminable ARCH through.
        logger.warning(
            "gfx1250 asm gate: could not read ASIC revision (%s); "
            "treating device as unsupported (fail-closed).",
            e,
        )
        return False


# Probed once at import, not per call: the gate sits at the top of hot ops, and
# a module-level constant keeps it free under torch.compile (a custom op here
# costs ~2.4us/call and turns the answer into an opaque graph node).
_ASM_SUPPORTED = _probe_stepping_ok()


def is_gfx1250_asm_supported() -> bool:
    """False on gfx1250 A0 (shipped asm is B0+ only); True otherwise.

    Frameworks can call this at startup to select a backend before the hard gate.
    """
    return _ASM_SUPPORTED


def require_gfx1250_asm(op_name: str) -> None:
    """Raise on gfx1250 A0 (shipped asm is B0+ only); no-op otherwise."""
    if _ASM_SUPPORTED or op_name in _A0_ALLOWLIST:
        return
    raise RuntimeError(
        f"{op_name} asm is only supported on gfx1250 B0+ "
        "(current device is gfx1250 A0)."
    )
