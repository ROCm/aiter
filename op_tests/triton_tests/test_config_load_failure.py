# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""A failed config load must not leave ``_get_config`` memoized in a partial state.

These ``_get_config`` helpers cache the parsed JSON on the function object and guard the
load with ``hasattr``. If the attribute is created before the file is read, a failed read
leaves an empty memo behind, ``lru_cache`` does not cache the exception, and every later
call skips the load and raises something unrelated -- hiding which file was missing.

Needs no tuned config and no particular GPU: the configs directory is redirected to an
empty temporary directory, so the load fails for the arch actually under test.
"""

import importlib

import pytest
import torch

# module -> arguments its _get_config takes
CONFIG_LOADERS = {
    "aiter.ops.triton._triton_kernels.attention.mha": (False, torch.float16),
    "aiter.ops.triton._triton_kernels.attention.mha_fused_bwd": (),
    "aiter.ops.triton._triton_kernels.attention.mha_onekernel_bwd": (),
    "aiter.ops.triton._triton_kernels.attention.extend_attention": (128, torch.float16),
    "aiter.ops.triton._triton_kernels.attention.mla_decode_rope": (),
    "aiter.ops.triton._triton_kernels.moe.moe_routing_sigmoid_top1_fused": (
        128,
        16,
        128,
    ),
}


@pytest.mark.parametrize("module_name", CONFIG_LOADERS)
def test_missing_config_raises_the_same_error_every_call(
    module_name, tmp_path, monkeypatch
):
    module = importlib.import_module(module_name)
    args = CONFIG_LOADERS[module_name]

    monkeypatch.setattr(module, "AITER_TRITON_CONFIGS_PATH", str(tmp_path))
    try:
        if hasattr(module._get_config, "_config_dict"):
            del module._get_config._config_dict
        module._get_config.cache_clear()

        with pytest.raises(FileNotFoundError) as first:
            module._get_config(*args)
        with pytest.raises(FileNotFoundError) as second:
            module._get_config(*args)

        assert str(first.value) == str(second.value)
    finally:
        # Never leave a memo built against tmp_path behind for the rest of the session.
        if hasattr(module._get_config, "_config_dict"):
            del module._get_config._config_dict
        module._get_config.cache_clear()
