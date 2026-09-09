# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for GPU target resolution: the arch and CU count aiter builds and
dispatches for, and the caches keyed on that answer.

The HIP device is faked, so these run without a GPU.
"""

import contextlib
import os
import sys
import tempfile
from types import SimpleNamespace
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


@contextlib.contextmanager
def _fake_hip_host(devices, current=0):
    """Present `devices` as the visible HIP devices; [] means no HIP context."""
    import torch

    props = [
        SimpleNamespace(gcnArchName=arch, multi_processor_count=cu)
        for arch, cu in devices
    ]
    # With no devices is_available() is False, so chip_info short-circuits
    # before it reaches the other two patches.
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=bool(props)),
        mock.patch.object(torch.cuda, "current_device", return_value=current),
        mock.patch.object(torch.cuda, "get_device_properties", lambda i: props[i]),
    ):
        yield


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


def test_active_device_arch_resolution():
    from aiter.jit.utils import chip_info

    def _reset():
        chip_info.get_gfx_runtime.cache_clear()
        chip_info.get_cu_num.cache_clear()
        chip_info.get_gfx.cache_clear()
        chip_info.get_gfx_custom_op_core.cache_clear()

    # A mixed host: rocminfo and HIP both enumerate the gfx942 part first, but
    # the process launches on device 1. Every answer must describe device 1, so
    # reading device 0 fails these with gfx942/304 rather than passing quietly.
    mixed_host = [("gfx942:sramecc+:xnack-", 304), ("gfx950:sramecc+:xnack-", 256)]
    rocminfo_first = mock.patch.object(
        chip_info, "_detect_native_rocminfo", return_value=["gfx942"]
    )
    with _restored_env("AITER_GPU_TARGETS", "GPU_ARCHS", "CU_NUM"):
        try:
            _reset()
            with rocminfo_first, _fake_hip_host(mixed_host, current=1):
                native = chip_info._detect_native()
                assert native == ["gfx950"], (
                    f"_detect_native should return the launching device, not the "
                    f"first agent, got {native}"
                )
                runtime = chip_info.get_gfx_runtime()
                assert (
                    runtime == "gfx950"
                ), f"get_gfx_runtime should resolve the launching device, got {runtime}"
                cu_num = chip_info.get_cu_num()
                assert (
                    cu_num == 256
                ), f"get_cu_num should read the launching device, got {cu_num}"

            # No HIP context (GPU-less build host): rocminfo is still the fallback.
            _reset()
            with rocminfo_first, _fake_hip_host([]):
                runtime = chip_info.get_gfx_runtime()
                assert (
                    runtime == "gfx942"
                ), f"no HIP context should fall back to rocminfo, got {runtime}"

            # That fallback must not be pinned: a caller running before the HIP
            # context exists would otherwise fix gfx942 for the whole process.
            with rocminfo_first, _fake_hip_host(mixed_host, current=1):
                runtime = chip_info.get_gfx_runtime()
                assert runtime == "gfx950", (
                    f"a pre-context rocminfo answer must not be memoised, got "
                    f"{runtime}"
                )
        finally:
            _reset()


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
    test_active_device_arch_resolution()
    test_cpp_itfs_cache_identity()
    print("ALL_PASS")
