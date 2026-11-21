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


def test_head_padding_aot_cache_reuse() -> None:
    """D256 and D192 must not share an AOT padding specialization."""
    with _use_aot_backend():
        for block_size in (64, 1024):
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
                    quant_kv=False,
                    use_sinks=False,
                    sliding_window=0,
                    ps=False,
                )
                assert result["err_gluon"] == 0


@pytest.mark.parametrize(
    "key_dim,value_dim,output_dim,trans_v",
    [
        (128, 192, 192, True),
        (192, 128, 192, True),
        (192, 128, 192, False),
        (192, 192, 128, True),
        (192, 128, 128, True),
    ],
)
def test_aot_rejects_mismatched_head_widths(
    monkeypatch, key_dim, value_dim, output_dim, trans_v
) -> None:
    """Reject incompatible physical storage before AOT compilation or launch."""
    module = importlib.import_module("csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot")
    monkeypatch.setattr(module, "get_cdna_version", lambda: 4)

    def unexpected_compile(**kwargs):
        pytest.fail("invalid head widths reached AOT compilation")

    monkeypatch.setattr(module, "compile", unexpected_compile)
    query = torch.empty((8, 4, 192), dtype=torch.bfloat16, device="meta")
    key = query.new_empty((34, 1, key_dim // 8, 64, 8))
    value = query.new_empty(
        (34, 1, 8, value_dim, 8) if trans_v else (34, 1, value_dim, 64)
    )
    output = query.new_empty((8, 4, output_dim))
    lengths = torch.empty(2, dtype=torch.int32, device="meta")
    tables = lengths.new_empty((2, 17))
    stats = query.new_empty((2, 1, 5, 16), dtype=torch.float32)
    scratch = query.new_empty((2, 1, 5, 16, 192))
    with pytest.raises(ValueError, match="physical Q/K, V and output head dimensions"):
        pa_decode_gluon_aot(
            output,
            query,
            key,
            value,
            lengths,
            tables,
            192**-0.5,
            4,
            5,
            256,
            torch.bfloat16,
            None,
            None,
            None,
            stats,
            stats,
            scratch,
        )


def run_normal_performance_aot() -> None:
    prebuild_normal_performance_cases_aot_so()
    get_so_files_size_and_count()
    with _use_aot_backend():
        pa_decode_test.normal_performance_test()


if __name__ == "__main__":
    test_normal_accuracy_aot()
    test_head_padding_aot_cache_reuse()
