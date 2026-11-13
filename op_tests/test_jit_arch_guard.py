# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the JIT arch-coverage guard offload-target parser."""

import contextlib
import os
import tempfile

from aiter.jit import core
from aiter.jit.core import _so_offload_archs


def _patch_jit_dir(tmpdir):
    """Point the arch helpers at an isolated jit dir."""
    orig = core.get_user_jit_dir
    core.get_user_jit_dir = lambda: tmpdir
    return orig


def _restore_jit_dir(orig):
    core.get_user_jit_dir = orig


@contextlib.contextmanager
def _gpu_archs(value):
    """Set GPU_ARCHS for a block (None unsets it); CI runners export their own."""
    old = os.environ.get("GPU_ARCHS")
    if value is None:
        os.environ.pop("GPU_ARCHS", None)
    else:
        os.environ["GPU_ARCHS"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GPU_ARCHS", None)
        else:
            os.environ["GPU_ARCHS"] = old


def test_parse_offload_archs():
    blob = (
        b"\x00ELF junk hipv4-amdgcn-amd-amdhsa--gfx942 more "
        b"hipv4-amdgcn-amd-amdhsa--gfx950 "
        b"hipv4-amdgcn-amd-amdhsa--gfx1250-strict "
        b"host__amdhsa--gfx1201 tail"
    )
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert _so_offload_archs(path) == {
            "gfx942",
            "gfx950",
            "gfx1201",
            "gfx1250-strict",
        }
    finally:
        os.remove(path)


def test_parse_host_only_is_empty():
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
        f.write(b"pure host extension, no device code")
        path = f.name
    try:
        assert _so_offload_archs(path) == set()
    finally:
        os.remove(path)


def test_missing_file_is_empty():
    assert _so_offload_archs("/no/such/module.so") == set()


def test_parse_offload_archs_keeps_full_target_ids():
    """Target-id features, the strict suffix, and ROCm generic targets.

    The parser stops at the ':' that starts a feature list, but it must keep
    every hyphenated part of the processor name itself.  Truncating at the
    first hyphen reported gfx1250-strict as gfx1250 (the bug this guard
    exists to prevent) and gfx9-4-generic as gfx9, which never matches a
    running arch and so rebuilt the module on every import.
    """
    cases = {
        b"amdhsa--gfx90a:sramecc+:xnack-": "gfx90a",
        b"amdhsa--gfx1250-strict": "gfx1250-strict",
        b"amdhsa--gfx1250-strict:xnack-": "gfx1250-strict",
        b"amdhsa--gfx9-4-generic": "gfx9-4-generic",
    }
    for blob, expected in cases.items():
        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
            f.write(b"\x00ELF junk hipv4-amdgcn-amd-" + blob + b"\x00tail")
            path = f.name
        try:
            assert _so_offload_archs(path) == {expected}, blob
        finally:
            os.remove(path)


def test_validate_strict_build_target():
    old = os.environ.get("GPU_ARCHS")
    try:
        os.environ["GPU_ARCHS"] = "gfx1250-strict"
        assert core.validate_and_update_archs() == ["gfx1250-strict"]
    finally:
        if old is None:
            os.environ.pop("GPU_ARCHS", None)
        else:
            os.environ["GPU_ARCHS"] = old


def test_needs_arch_rebuild():
    # A rebuild (GPU_ARCHS=native) covers the running arch, so each uncovered
    # case below rebuilds; the guard that raises instead is tested after it.
    with _gpu_archs("native"), tempfile.TemporaryDirectory() as tmpdir:
        orig = _patch_jit_dir(tmpdir)
        orig_runtime = core.get_gfx_runtime
        try:
            so_path = os.path.join(tmpdir, "module_dummy.so")
            with open(so_path, "wb") as f:
                f.write(
                    b"\x00ELF junk hipv4-amdgcn-amd-amdhsa--gfx942 more "
                    b"hipv4-amdgcn-amd-amdhsa--gfx950 tail"
                )

            # Running on an arch the .so does NOT cover -> rebuild.
            core.get_gfx_runtime = lambda: "gfx1201"
            assert core._needs_arch_rebuild("module_dummy") is True

            # Running on a covered arch -> no rebuild.
            core.get_gfx_runtime = lambda: "gfx942"
            assert core._needs_arch_rebuild("module_dummy") is False

            # gfx1250 and gfx1250-strict are intentionally incompatible.
            with open(so_path, "wb") as f:
                f.write(b"hipv4-amdgcn-amd-amdhsa--gfx1250-strict")
            core.get_gfx_runtime = lambda: "gfx1250"
            assert core._needs_arch_rebuild("module_dummy") is True

            with open(so_path, "wb") as f:
                f.write(b"hipv4-amdgcn-amd-amdhsa--gfx1250")
            core.get_gfx_runtime = lambda: "gfx1250-strict"
            assert core._needs_arch_rebuild("module_dummy") is True

            # Unknown build archs (no manifest, no .so) -> no rebuild.
            assert core._needs_arch_rebuild("module_unknown") is False
        finally:
            core.get_gfx_runtime = orig_runtime
            _restore_jit_dir(orig)


def test_needs_arch_rebuild_raises_when_a_rebuild_cannot_cover_the_arch():
    """A .so for other archs under a GPU_ARCHS that omits the running arch.

    Loading it only faults at its first kernel launch, and a rebuild would miss
    the arch too, so the guard raises at load and names GPU_ARCHS.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = _patch_jit_dir(tmpdir)
        orig_runtime = core.get_gfx_runtime
        try:
            so_path = os.path.join(tmpdir, "module_dummy.so")
            with open(so_path, "wb") as f:
                f.write(
                    b"\x00ELF junk hipv4-amdgcn-amd-amdhsa--gfx942 more "
                    b"hipv4-amdgcn-amd-amdhsa--gfx950 tail"
                )
            core.get_gfx_runtime = lambda: "gfx1201"
            with _gpu_archs("gfx950"):
                try:
                    core._needs_arch_rebuild("module_dummy")
                except RuntimeError as e:
                    message = str(e)
                else:
                    raise AssertionError("expected a RuntimeError")
            for part in ("module_dummy", so_path, "gfx1201", "GPU_ARCHS='gfx950'"):
                assert part in message, (part, message)
            # A rebuild that does cover the running arch is attempted instead.
            for value in ("gfx950;gfx1201", "native"):
                with _gpu_archs(value):
                    assert core._needs_arch_rebuild("module_dummy") is True, value
            # An unusable GPU_ARCHS is left for the build to report.
            with _gpu_archs("not-an-arch"):
                assert core._needs_arch_rebuild("module_dummy") is True
            # A covered arch needs neither, whatever GPU_ARCHS says.
            core.get_gfx_runtime = lambda: "gfx942"
            with _gpu_archs("gfx950"):
                assert core._needs_arch_rebuild("module_dummy") is False
        finally:
            core.get_gfx_runtime = orig_runtime
            _restore_jit_dir(orig)


if __name__ == "__main__":
    test_parse_offload_archs()
    test_parse_host_only_is_empty()
    test_missing_file_is_empty()
    test_needs_arch_rebuild()
    test_needs_arch_rebuild_raises_when_a_rebuild_cannot_cover_the_arch()
    print("ALL_PASS")
