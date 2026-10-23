# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import logging

from .chip_info import get_asic_revision, get_gfx_runtime

logger = logging.getLogger("aiter")

_A0_ALLOWLIST: frozenset[str] = frozenset()


def _probe_gfx1250_arch() -> str | None:
    # Arch undeterminable -> don't block; the C++ gate is authoritative for A0.
    try:
        arch = get_gfx_runtime()
        return arch if arch in ("gfx1250", "gfx1250-strict") else None
    except Exception:  # noqa: BLE001
        return None


def _probe_stepping_ok() -> bool:
    arch = _probe_gfx1250_arch()
    if arch is None:
        return True
    # gfx1250-strict is the compiler/runtime identity for MI450 A0.  Never let
    # it reach shipped gfx1250 ASM, even if stepping discovery is unavailable.
    if arch == "gfx1250-strict":
        return False
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
    """False on gfx1250-strict/A0 (shipped asm is B0+ only); True otherwise.

    Frameworks can call this at startup to select a backend before the hard gate.
    """
    return _ASM_SUPPORTED


def require_gfx1250_asm(op_name: str) -> None:
    """Raise on gfx1250-strict/A0; no-op on gfx1250 B0+ and other targets."""
    if _ASM_SUPPORTED or op_name in _A0_ALLOWLIST:
        return
    raise RuntimeError(
        f"{op_name} asm is only supported on gfx1250 B0+ "
        "(current device is gfx1250-strict/A0)."
    )
