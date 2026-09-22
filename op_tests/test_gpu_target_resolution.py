# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for GPU target resolution: the arch and CU count aiter builds and
dispatches for, and the caches keyed on that answer.

The live GPU probes are faked, so these run without a GPU.
"""

import contextlib
import os
import sys
import tempfile
from unittest import mock

# Ensure the repo-local aiter is imported, not any system/site-packages install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


@contextlib.contextmanager
def _restored_env(*names):
    """Clear `names` for the duration, then restore whatever was there."""
    original = {name: os.environ.pop(name, None) for name in names}
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_runtime_arch_resolution():
    from aiter.jit import core
    from aiter.jit.utils import chip_info

    with _restored_env("AITER_GPU_TARGETS", "GPU_ARCHS", "CU_NUM"):
        try:
            os.environ["AITER_GPU_TARGETS"] = "gfx950:256;gfx942:304"
            with mock.patch.object(
                chip_info, "_detect_native", return_value=["gfx942"]
            ):
                chip_info.get_gfx_custom_op_core.cache_clear()
                detected = chip_info.GFX_MAP[chip_info.get_gfx_custom_op_core()]
                assert detected == "gfx942", (
                    f"multi-target runtime dispatch should use the live named "
                    f"arch, got {detected}"
                )

            # Live arch not among the named targets: the result is the
            # order-independent max(named), not whichever entry happens to be last.
            for target_spec in ("gfx950:256;gfx942:304", "gfx942:304;gfx950:256"):
                os.environ["AITER_GPU_TARGETS"] = target_spec
                with mock.patch.object(
                    chip_info, "_detect_native", return_value=["gfx1201"]
                ):
                    chip_info.get_gfx_custom_op_core.cache_clear()
                    detected = chip_info.GFX_MAP[chip_info.get_gfx_custom_op_core()]
                assert detected == "gfx950", (
                    f"un-named live arch should resolve to max(named) for "
                    f"{target_spec}, got {detected}"
                )
            chip_info.get_gfx_custom_op_core.cache_clear()

            opus_flag_sets = []
            for target_spec in ("gfx1250:256;gfx950:256", "gfx950:256;gfx1250:256"):
                os.environ["AITER_GPU_TARGETS"] = target_spec
                core.get_gfx_list.cache_clear()
                opus_flag_sets.append(
                    {
                        flag
                        for flag in core.get_args_of_build("module_deepgemm_opus")[
                            "flags_extra_hip"
                        ]
                        if flag
                    }
                )

            required_flags = {
                "-mllvm -amdgpu-expert-scheduling-mode",
                "-mllvm -enable-post-misched=1",
            }
            assert all(required_flags <= flags for flags in opus_flag_sets), (
                f"gfx1250 OPUS flags should follow target membership independent "
                f"of order, got {opus_flag_sets}"
            )
        finally:
            chip_info.get_gfx_custom_op_core.cache_clear()
            chip_info.get_gfx.cache_clear()
            core.get_gfx_list.cache_clear()


def test_gpu_archs_takes_the_live_cu_count():
    from aiter.jit.utils import chip_info

    # A binned gfx950: naming the arch alone must not resolve to the full SKU.
    with _restored_env("AITER_GPU_TARGETS", "GPU_ARCHS", "CU_NUM"):
        os.environ["GPU_ARCHS"] = "gfx950"
        chip_info.get_gfx_runtime.cache_clear()
        chip_info.get_cu_num.cache_clear()
        try:
            with (
                mock.patch.object(
                    chip_info, "_detect_native", return_value=["gfx950"]
                ),
                mock.patch.object(chip_info, "get_cu_num_custom_op", return_value=128),
            ):
                targets = chip_info.get_build_targets()
        finally:
            chip_info.get_gfx_runtime.cache_clear()
            chip_info.get_cu_num.cache_clear()
    assert targets == [
        ("gfx950", 128)
    ], f"GPU_ARCHS should take the live CU count for a binned part, got {targets}"


def test_cpp_itfs_cache_identity():
    import csrc.cpp_itfs.utils as cpp_utils

    original_gpu_arch = cpp_utils.GPU_ARCH
    original_build_dir = cpp_utils.BUILD_DIR
    try:
        cpp_utils.GPU_ARCH = "gfx942"
        cpp_utils.get_arch_key.cache_clear()
        gfx942_dir = cpp_utils.get_template_build_dir("same_specialization")
        cpp_utils.GPU_ARCH = "gfx950"
        cpp_utils.get_arch_key.cache_clear()
        gfx950_dir = cpp_utils.get_template_build_dir("same_specialization")
        assert gfx942_dir != gfx950_dir, (
            f"template library cache should separate gfx942 and gfx950, got "
            f"{gfx942_dir!r} for both"
        )

        with tempfile.TemporaryDirectory() as build_dir:
            cpp_utils.BUILD_DIR = build_dir
            cpp_utils.GPU_ARCH = "gfx942;gfx950"
            cpp_utils.get_arch_key.cache_clear()
            constexprs = {"X": 1}
            hsaco_name = cpp_utils.get_default_func_name("kernel", (1,))
            actual_dir = os.path.join(build_dir, "gfx950")
            os.makedirs(actual_dir)
            with open(os.path.join(actual_dir, f"{hsaco_name}.hsaco"), "wb") as f:
                f.write(b"test")
            with mock.patch.object(cpp_utils, "get_gfx_runtime", return_value="gfx950"):
                assert cpp_utils.check_hsaco(
                    "kernel", constexprs
                ), "HSACO lookup should use the live arch, not the composite path"
    finally:
        cpp_utils.GPU_ARCH = original_gpu_arch
        cpp_utils.BUILD_DIR = original_build_dir
        cpp_utils.get_arch_key.cache_clear()


if __name__ == "__main__":
    test_runtime_arch_resolution()
    test_gpu_archs_takes_the_live_cu_count()
    test_cpp_itfs_cache_identity()
    print("ALL_PASS")
