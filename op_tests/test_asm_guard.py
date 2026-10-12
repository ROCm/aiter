# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Checks the auto-detected stepping and the gfx1250 B0-only asm gate.

Everything here reads THIS machine through rocminfo -- nothing is faked -- so
the A0 branch is only exercised when the test runs on a gfx1250 A0 board.
"""

import re
from pathlib import Path

import pytest

from aiter.jit.utils import asm_guard
from aiter.jit.utils.chip_info import (
    _rocminfo_gpu_agents,
    get_asic_revision,
    get_gfx_runtime,
)

_STEPPING_NAME = {0: "A0", 1: "B0", 2: "C0"}


@pytest.fixture(scope="module")
def detected():
    """(arch, asicRevision) of this machine; skip when there is no GPU."""
    try:
        return get_gfx_runtime(), get_asic_revision()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no detectable GPU ({type(e).__name__}: {e})")


def test_rocminfo_reports_the_running_gpu(detected):
    arch, _ = detected
    agents = _rocminfo_gpu_agents()
    assert agents, "rocminfo reported no GPU agent"
    assert arch in [name for name, _ in agents]


def test_stepping_is_the_node_minimum(detected):
    # Node-wide answer: on a mixed-stepping node the lowest stepping wins, so
    # the gate stays closed for the whole node.
    arch, rev = detected
    assert rev == min(r for name, r in _rocminfo_gpu_agents() if name == arch)


def test_stepping_is_plausible(detected):
    _, rev = detected
    assert 0 <= rev <= 15, f"implausible ASIC Revision {rev}"


def test_gate_follows_detected_stepping(detected):
    arch, rev = detected
    expected = arch != "gfx1250" or rev >= 1
    assert asm_guard.is_gfx1250_asm_supported() is expected


def test_require_follows_gate(detected):
    supported = asm_guard.is_gfx1250_asm_supported()
    try:
        asm_guard.require_gfx1250_asm("some_asm_op")
    except RuntimeError:
        assert not supported, "require_gfx1250_asm raised on a supported device"
        return
    assert supported, "require_gfx1250_asm passed on gfx1250 A0"


def test_a0_allowlists_are_both_empty():
    # Two allowlists on purpose: Python keys by op name, C++ by kernel name, so
    # entries cannot be compared 1:1. What IS checkable is that both are empty
    # while all shipped gfx1250 asm is B0-only -- adding to one side alone
    # fails here and forces the other side to be revisited.
    assert asm_guard._A0_ALLOWLIST == frozenset()

    header = Path(__file__).resolve().parents[1] / "csrc/include/aiter_hip_common.h"
    body = re.search(
        r"kA0AllowList\[\]\s*=\s*\{(.*?)\};", header.read_text(), re.DOTALL
    )
    assert body is not None, "kA0AllowList not found in aiter_hip_common.h"
    entries = re.sub(r"//[^\n]*", "", body.group(1))  # drop comments
    assert '"' not in entries, "C++ A0 allowlist gained an entry"


if __name__ == "__main__":
    arch, rev = get_gfx_runtime(), get_asic_revision()
    print(
        f"[auto-detect] platform={arch} asicRevision={rev} "
        f"({_STEPPING_NAME.get(rev, f'rev{rev}')}) "
        f"gfx1250_asm_supported={asm_guard.is_gfx1250_asm_supported()}"
    )
    raise SystemExit(pytest.main([__file__, "-v"]))
