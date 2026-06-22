# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the JIT arch-coverage guard offload-target parser."""

import os
import tempfile

from aiter.jit.core import _so_offload_archs


def test_parse_offload_archs():
    blob = (
        b"\x00ELF junk hipv4-amdgcn-amd-amdhsa--gfx942 more "
        b"hipv4-amdgcn-amd-amdhsa--gfx950 host__amdhsa--gfx1201 tail"
    )
    with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as f:
        f.write(blob)
        path = f.name
    try:
        assert _so_offload_archs(path) == {"gfx942", "gfx950", "gfx1201"}
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


if __name__ == "__main__":
    test_parse_offload_archs()
    test_parse_host_only_is_empty()
    test_missing_file_is_empty()
    print("ALL_PASS")
