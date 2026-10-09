# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

pa_decode_gluon_aot = importlib.import_module(
    "csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot"
).pa_decode_gluon_aot
_aot_prebuild = importlib.import_module(
    "csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_prebuild"
)
get_so_files_size_and_count = _aot_prebuild.get_so_files_size_and_count
prebuild_normal_accuracy_cases_aot_so = (
    _aot_prebuild.prebuild_normal_accuracy_cases_aot_so
)
prebuild_normal_performance_cases_aot_so = (
    _aot_prebuild.prebuild_normal_performance_cases_aot_so
)
pa_decode_test = importlib.import_module("op_tests.triton_tests.test_pa_decode_gluon")


def _run_aot(*args, **kwargs) -> None:
    sliding_window = kwargs.pop("sliding_window", 0)
    ps = kwargs.pop("ps", False)
    if sliding_window != 0 or ps:
        raise ValueError(
            "pa_decode_gluon_aot only supports sliding_window=0 and ps=False"
        )
    pa_decode_gluon_aot(*args, **kwargs)


@contextmanager
def _use_aot_backend() -> Iterator[None]:
    jit_backend = pa_decode_test.pa_decode_gluon
    test_name = pa_decode_test.TEST_NAME
    pa_decode_test.pa_decode_gluon = _run_aot
    pa_decode_test.TEST_NAME = "main.normal_accuracy_performance.aot"
    try:
        yield
    finally:
        pa_decode_test.pa_decode_gluon = jit_backend
        pa_decode_test.TEST_NAME = test_name


def test_normal_accuracy_aot() -> None:
    prebuild_normal_accuracy_cases_aot_so()
    get_so_files_size_and_count()
    with _use_aot_backend():
        pa_decode_test.normal_accuracy_test()


@pytest.mark.parametrize("block_size", [64, 1024])
@pytest.mark.parametrize("quant_kv", [False, True], ids=["bf16-kv", "fp8-kv"])
def test_head_padding_aot_cache_reuse(block_size, quant_kv, monkeypatch) -> None:
    """D256 and D192 must not share an AOT padding specialization."""
    monkeypatch.setattr(pa_decode_test, "USE_TORCH_FLASH_REF", False)
    with _use_aot_backend():
        # Both dimensions round to 256. Compile the unmasked variant first,
        # then check padded storage and reuse the unmasked variant again.
        for head_size in (256, 192, 256):
            result = pa_decode_test.run_pa_gluon_test(
                context_length=1025,
                batch_size=2,
                num_heads=(4, 1),
                head_size=head_size,
                block_size=block_size,
                compute_type=torch.bfloat16,
                query_length=4,
                quant_mode="per_tensor",
                context_partition_size=256,
                trans_v=True,
                kv_varlen=False,
                quant_q=False,
                quant_kv=quant_kv,
                use_sinks=False,
                sliding_window=0,
                ps=False,
            )
            assert result["err_gluon"] == 0


def run_normal_performance_aot() -> None:
    prebuild_normal_performance_cases_aot_so()
    get_so_files_size_and_count()
    with _use_aot_backend():
        pa_decode_test.normal_performance_test()


if __name__ == "__main__":
    test_normal_accuracy_aot()
