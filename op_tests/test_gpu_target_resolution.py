# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU target resolution and cache identity tests with mocked GPU probes."""

import contextlib
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Ensure the repo-local aiter is imported, not any system/site-packages install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def test_runtime_arch_resolution():
    from aiter.jit.utils import chip_info

    gpu_target_strings = ("gfx950:256;gfx942:304", "gfx942:304;gfx950:256")
    live_expected_gfxs = (
        ("gfx942", "gfx942"),
        ("gfx1201", "gfx950"),
    )
    for gpu_target_string in gpu_target_strings:
        for live_gfx, expected_gfx in live_expected_gfxs:
            with (
                _target_env(AITER_GPU_TARGETS=gpu_target_string),
                _cleared_cache(chip_info.get_gfx_custom_op_core),
                mock.patch.object(chip_info, "_detect_native", return_value=[live_gfx]),
            ):
                resolved_gfx = chip_info.GFX_MAP[chip_info.get_gfx_custom_op_core()]

            assert (
                resolved_gfx == expected_gfx
            ), f"{gpu_target_string}, live {live_gfx}: got {resolved_gfx}"


def test_opus_flags_follow_target_membership():
    from aiter.jit import core

    required_flags = {
        "-mllvm -amdgpu-expert-scheduling-mode",
        "-mllvm -enable-post-misched=1",
    }
    for gpu_target_string in ("gfx1250:256;gfx950:256", "gfx950:256;gfx1250:256"):
        with (
            _target_env(AITER_GPU_TARGETS=gpu_target_string),
            _cleared_cache(core.get_gfx_list),
        ):
            args = core.get_args_of_build("module_deepgemm_opus")

        assert required_flags <= set(args["flags_extra_hip"]), gpu_target_string


def test_gpu_archs_takes_the_live_cu_count():
    from aiter.jit.utils import chip_info

    with (
        _target_env(GPU_ARCHS="gfx950"),
        mock.patch.object(chip_info, "get_gfx_runtime", return_value="gfx950"),
        mock.patch.object(chip_info, "get_cu_num", return_value=128),
    ):
        targets = chip_info.get_build_targets()

    assert targets == [("gfx950", 128)], targets


def test_template_cache_separates_architectures():
    import csrc.cpp_itfs.utils as cpp_utils

    directories = []
    for gfx in ("gfx942", "gfx950"):
        with (
            mock.patch.object(cpp_utils, "GPU_ARCH", gfx),
            _cleared_cache(cpp_utils.get_arch_key),
        ):
            directories.append(cpp_utils.get_template_build_dir("same_specialization"))

    assert directories[0] != directories[1], directories


def test_hsaco_lookup_uses_live_arch():
    import csrc.cpp_itfs.utils as cpp_utils

    with (
        tempfile.TemporaryDirectory() as build_dir,
        mock.patch.object(cpp_utils, "BUILD_DIR", build_dir),
        mock.patch.object(cpp_utils, "GPU_ARCH", "gfx942;gfx950"),
        mock.patch.object(cpp_utils, "get_gfx_runtime", return_value="gfx950"),
    ):
        hsaco_name = cpp_utils.get_default_func_name("kernel", (1,))
        hsaco_dir = Path(build_dir) / "gfx950"
        hsaco_dir.mkdir()
        (hsaco_dir / f"{hsaco_name}.hsaco").write_bytes(b"test")

        assert cpp_utils.check_hsaco("kernel", {"X": 1})


@contextlib.contextmanager
def _target_env(**values):
    with mock.patch.dict(os.environ):
        for name in ("AITER_GPU_TARGETS", "GPU_ARCHS", "CU_NUM"):
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


@contextlib.contextmanager
def _cleared_cache(function):
    function.cache_clear()
    try:
        yield
    finally:
        function.cache_clear()


if __name__ == "__main__":
    test_runtime_arch_resolution()
    test_opus_flags_follow_target_membership()
    test_gpu_archs_takes_the_live_cu_count()
    test_template_cache_separates_architectures()
    test_hsaco_lookup_uses_live_arch()
    print("ALL_PASS")
