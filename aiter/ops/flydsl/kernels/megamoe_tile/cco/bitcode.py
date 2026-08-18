# SPDX-License-Identifier: MIT
"""JIT the AITER-local CCO device bridge for the active ROCm arch/NIC."""

from __future__ import annotations

from pathlib import Path
import tempfile


_cached: str | None = None


def _sdma_enabled() -> bool:
    """Mirror the installed MORI host build flag in the private device TU.

    The private bridge only instantiates GDA, and ``ccoDevComm`` layout is stable
    in either mode.  Matching the host flag nevertheless prevents a stale cache
    entry from silently masking a packaging/configuration mismatch.
    """

    import os

    env = os.environ.get("BUILD_CCO_SDMA")
    if env is not None:
        return env.strip().upper() in {"1", "ON", "TRUE", "YES"}
    try:
        from mori.cco.device._build_flags import BUILD_CCO_SDMA

        return bool(BUILD_CCO_SDMA)
    except (ImportError, AttributeError):
        return False


def get_bitcode_path(cov: int = 6) -> str:
    global _cached
    if _cached is not None:
        return _cached

    from mori.jit.config import detect_build_config, detect_nic_type, get_mori_source_root
    from mori.jit.cache import get_cache_dir
    from mori.jit.core import (
        FileBaton,
        _collect_include_dirs,
        _hipcc_device_bc,
        _strip_lifetime_intrinsics,
    )

    source = Path(__file__).with_name("cco_device_bridge.cpp")
    mori_root = get_mori_source_root()
    if mori_root is None:
        raise RuntimeError("MORI JIT sources are required for the CCO bridge")

    cfg = detect_build_config()
    nic = detect_nic_type()
    sdma_enabled = _sdma_enabled()
    cache = get_cache_dir(
        cfg.arch,
        [source, mori_root / "include" / "mori" / "cco", mori_root / "include" / "mori" / "core"],
        nic,
        cov=cov,
    ) / f"aiter_megamoe_cco_sdma{int(sdma_enabled)}"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / "libaiter_megamoe_cco.bc"
    lock = cache / ".libaiter_megamoe_cco.bc.lock"

    with FileBaton(lock, wait_for=str(output)) as baton:
        if not baton.skipped and not output.is_file():
            with tempfile.TemporaryDirectory() as td:
                raw = Path(td) / "bridge.bc"
                _hipcc_device_bc(
                    cfg,
                    source,
                    _collect_include_dirs(mori_root),
                    raw,
                    cov=cov,
                    extra_defines=[f"-DBUILD_CCO_SDMA={int(sdma_enabled)}"],
                )
                _strip_lifetime_intrinsics(cfg, raw, output)

    _cached = str(output)
    return _cached
