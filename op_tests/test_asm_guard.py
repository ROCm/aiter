# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the gfx1250 B0-only asm dispatch guard.

Hardware-independent: the guard's decision logic is exercised by monkeypatching
the arch/stepping helpers, so the A0 path can be tested without a gfx1250 A0
device (which almost nobody has).
"""

from aiter.jit.utils import asm_guard


def report_current_platform():
    """Print this machine's auto-detected arch + stepping (best-effort, __main__)."""
    try:
        from aiter.jit.utils.chip_info import get_asic_revision, get_gfx_runtime

        arch = get_gfx_runtime()
        rev = get_asic_revision()
        stepping = {0: "A0", 1: "B0", 2: "C0"}.get(rev, f"rev{rev}")
        print(
            f"[auto-detect] platform={arch} asicRevision={rev} ({stepping}) "
            f"gfx1250_asm_supported={asm_guard.is_gfx1250_asm_supported()}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[auto-detect] unavailable ({type(e).__name__}: {e})")


def _set(monkey_arch, monkey_rev):
    """Patch the arch/stepping sources (zero-arg callables) and reset the cache."""
    asm_guard.get_gfx_runtime = monkey_arch
    asm_guard.get_asic_revision = monkey_rev
    asm_guard._is_gfx1250_asm_supported_cached.cache_clear()


def _raise():
    def _f():
        raise RuntimeError("boom")

    return _f


def test_non_gfx1250_is_supported():
    # Non-gfx1250: gate does not apply, stepping must not even be consulted.
    def _rev_should_not_be_called():
        raise AssertionError("stepping consulted on non-gfx1250 arch")

    _set(lambda: "gfx942", _rev_should_not_be_called)
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is True


def test_gfx1250_a0_is_unsupported():
    _set(lambda: "gfx1250", lambda: 0)
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is False


def test_gfx1250_b0_is_supported():
    _set(lambda: "gfx1250", lambda: 1)
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is True


def test_gfx1250_c0_is_supported():
    # Any stepping >= 1 (B0, C0, ...) is supported.
    _set(lambda: "gfx1250", lambda: 2)
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is True


def test_unknown_arch_does_not_overblock():
    # Arch undeterminable (e.g. rocminfo missing): do NOT block every arch.
    # The C++ AITER load-site gate remains authoritative for gfx1250 A0.
    _set(_raise(), lambda: 0)
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is True


def test_gfx1250_unreadable_stepping_fails_closed():
    # gfx1250 + unreadable stepping -> fail closed. Silence the by-design warning.
    import logging

    _set(lambda: "gfx1250", _raise())
    logging.getLogger("aiter").setLevel(logging.ERROR)
    try:
        assert asm_guard._is_gfx1250_asm_supported_cached(0) is False
    finally:
        logging.getLogger("aiter").setLevel(logging.NOTSET)


def test_require_raises_on_a0():
    _set(lambda: "gfx1250", lambda: 0)
    raised = False
    try:
        asm_guard.require_gfx1250_asm("some_asm_op")
    except RuntimeError:
        raised = True
    assert raised, "require_gfx1250_asm must raise on gfx1250 A0"


def test_require_noop_on_b0():
    _set(lambda: "gfx1250", lambda: 1)
    # Must NOT raise on B0.
    asm_guard.require_gfx1250_asm("some_asm_op")


def test_cache_is_per_device():
    # Mixed-stepping node: device 0 = A0, device 1 = B0. Each device must cache
    # independently, and querying device 1 must not clobber device 0's answer
    # (the bug when the cache was maxsize=1 with no device key).
    revs = {0: 0, 1: 1}
    pending = []

    def fake_rev():
        return revs[pending.pop(0)]

    _set(lambda: "gfx1250", fake_rev)

    pending[:] = [0]
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is False  # A0
    pending[:] = [1]
    assert asm_guard._is_gfx1250_asm_supported_cached(1) is True  # B0
    # Device 0 again -> served from cache, still False (not overwritten by dev 1).
    assert asm_guard._is_gfx1250_asm_supported_cached(0) is False


if __name__ == "__main__":
    report_current_platform()
    test_non_gfx1250_is_supported()
    test_gfx1250_a0_is_unsupported()
    test_gfx1250_b0_is_supported()
    test_gfx1250_c0_is_supported()
    test_unknown_arch_does_not_overblock()
    test_gfx1250_unreadable_stepping_fails_closed()
    test_require_raises_on_a0()
    test_require_noop_on_b0()
    test_cache_is_per_device()
    print("ALL_PASS")
