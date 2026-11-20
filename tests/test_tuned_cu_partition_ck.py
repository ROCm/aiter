# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK GEMM tuned-row lookups on compute partitions, with the partition mocked.

A partition (MI300X DPX 152 CUs, MI355X DPX 128 CUs) uses its own rows, then
the rows of its full chip. A GPU that is not a partition (MI308X 80 CUs)
uses only its own rows.
"""

import os
import types

import pandas as pd
import pytest
import torch

from aiter.jit.utils import chip_info
from aiter.ops import batched_gemm_op_a8w8, gemm_op_a8w8, gemm_op_common

MI300X_DPX = ("gfx942", (152, 304))
MI300X_CPX = ("gfx942", (38, 304))
MI355X_DPX = ("gfx950", (128, 256))
MI300X = ("gfx942", (304,))
MI308X = ("gfx942", (80,))


def padded_m(m, n, k, gl):
    """getPaddedM (csrc/py_itfs_cu/gemm_common.cu), without the GPU op."""
    pow2 = 1 << (m - 1).bit_length()
    if gl == 1:
        return 8192 if m > 8192 and n > 4096 else pow2
    if m < 16:
        return pow2
    grain = next((g for top, g in ((256, 16), (1024, 32), (4096, 64)) if m <= top), 128)
    return -(-m // grain) * grain


@pytest.fixture(autouse=True)
def host_padded_m(monkeypatch):
    monkeypatch.setattr(gemm_op_common, "get_padded_m", padded_m)


@pytest.fixture
def device(monkeypatch):
    """Make the current GPU a gfx with the given tuned CU counts."""

    def install(gpu, *modules):
        gfx, cu_nums = gpu
        monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda _: cu_nums)
        monkeypatch.setattr(chip_info, "_shared_tuned_cu_nums", lambda: cu_nums)
        for module in modules:
            monkeypatch.setattr(module, "get_gfx", lambda: gfx)
            monkeypatch.setattr(module, "get_cu_num", lambda: cu_nums[0])
        return gfx, cu_nums[0]

    return install


def write_csv(tmp_path, rows, name="tuned.csv"):
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def gemm_rows(*rows):
    return [
        {"gfx": gfx, "cu_num": cu, "M": m, "N": n, "K": k, "kernelId": i}
        | {"splitK": 0, "kernelName": name}
        for i, (gfx, cu, m, n, k, name) in enumerate(rows)
    ]


CK_ROWS = gemm_rows(
    ("gfx942", 304, 128, 4096, 4096, "full_chip"),
    ("gfx942", 152, 128, 1024, 4096, "own_152"),
    ("gfx942", 304, 128, 1024, 4096, "full_chip"),
    ("gfx942", 80, 128, 2048, 4096, "mi308x"),
    ("gfx942", 152, 128, 512, 4096, "own_152_padded"),
    ("gfx942", 304, 100, 512, 4096, "full_chip_exact"),
    ("gfx950", 256, 128, 4096, 4096, "gfx950_full_chip"),
)


@pytest.mark.parametrize(
    "gpu, m, n, expected",
    [
        (MI300X_DPX, 128, 4096, "full_chip"),
        (MI300X_DPX, 100, 4096, "full_chip"),  # padded M on the full chip
        (MI300X_CPX, 128, 4096, "full_chip"),
        (MI300X_DPX, 128, 1024, "own_152"),  # own row wins
        (MI300X_DPX, 100, 512, "full_chip_exact"),  # exact M before padded M
        (MI300X_DPX, 120, 512, "own_152_padded"),
        (MI300X_DPX, 128, 2048, None),  # never a smaller GPU
        (MI300X, 128, 4096, "full_chip"),
        (MI308X, 128, 4096, None),  # not a partition: no larger rows
        (MI355X_DPX, 128, 4096, "gfx950_full_chip"),
    ],
)
def test_ck_gemm_lookup(device, tmp_path, gpu, m, n, expected):
    gfx, cu = device(gpu)
    tuned_file = write_csv(tmp_path, CK_ROWS)
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    config = gemm_op_a8w8.get_CKGEMM_config(m, n, 4096, tuned_file, gfx=gfx, cu_num=cu)
    assert (config and config["kernelName"]) == (expected or None)


def test_ck_gemm_lookup_for_another_cu_count_is_exact(device, tmp_path):
    """An explicit cu_num of another GPU does not use this partition's rows."""
    device(MI300X_DPX)
    tuned_file = write_csv(tmp_path, CK_ROWS)
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    assert (
        gemm_op_a8w8.get_CKGEMM_config(128, 4096, 4096, tuned_file, cu_num=76) is None
    )


@pytest.mark.parametrize("device_index, expected", [(1, "full_chip"), (0, None)])
def test_ck_gemm_lookup_uses_the_partition_of_its_device(
    monkeypatch, tmp_path, device_index, expected
):
    """HIP device 0 is an MI300X in SPX mode, device 1 a partition in DPX mode."""
    devices = {0: (304,), 1: (152, 304)}
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", devices.get)
    tuned_file = write_csv(tmp_path, CK_ROWS)
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    config = gemm_op_a8w8.get_CKGEMM_config(
        128, 4096, 4096, tuned_file, gfx="gfx942", cu_num=152, device=device_index
    )
    assert (config and config["kernelName"]) == expected


@pytest.mark.parametrize("cu, expected", [(152, "own_152"), (304, None)])
def test_table_without_gfx_uses_the_exact_cu_count(device, tmp_path, cu, expected):
    """A row of a table without gfx can be of any gfx, so a partition keeps its own."""
    gfx, cu_num = device(MI300X_DPX)
    rows = [{k: v for k, v in row.items() if k != "gfx"} for row in CK_ROWS]
    tuned_file = write_csv(tmp_path, [r | {"cu_num": cu} for r in rows[1:2]])
    gemm_op_a8w8.get_CKGEMM_config.cache_clear()
    config = gemm_op_a8w8.get_CKGEMM_config(
        128, 1024, 4096, tuned_file, gfx=gfx, cu_num=cu_num
    )
    assert (config and config["kernelName"]) == expected


@pytest.mark.parametrize(
    "gpu, expected", [(MI300X_DPX, "full_chip"), (MI308X, None), (MI300X, "full_chip")]
)
def test_quant_type_lookup(device, tmp_path, gpu, expected):
    device(gpu, gemm_op_a8w8)
    rows = [row | {"q_dtype_w": str(torch.float8_e4m3fnuz)} for row in CK_ROWS]
    tuned_file = write_csv(tmp_path, rows)
    gemm_op_a8w8.get_GEMM_config_with_quant_type.cache_clear()
    config = gemm_op_a8w8.get_GEMM_config_with_quant_type(
        128, 4096, 4096, torch.float8_e4m3fnuz, tuned_file
    )
    assert (config and config["kernelName"]) == expected


@pytest.mark.parametrize("gpu, expected", [(MI300X_DPX, 304), (MI308X, None)])
def test_batched_lookup(device, monkeypatch, gpu, expected):
    device(gpu, batched_gemm_op_a8w8)
    lookup = batched_gemm_op_a8w8.get_CKBatchedGEMM_config
    row = {"kernelName": "a8w8_batched_ck_1x64x64x128", "splitK": 0}
    monkeypatch.setattr(
        lookup, "ck_batched_gemm_dict", {("gfx942", 304, 2, 64, 64, 128): row}, False
    )
    monkeypatch.setattr(lookup, "has_gfx", True, False)
    lookup.cache_clear()
    config = lookup(2, 64, 64, 128)
    assert (config and 304) == expected


def test_cpp_dispatch_gets_the_partitions(device, monkeypatch):
    """Each CK module gets the partitions of the node when aiter loads it."""
    from aiter.jit import core

    calls = []
    module = types.SimpleNamespace(set_tuned_cu_fallback=lambda *a: calls.append(a))
    info = {0: ("gfx942", (152, 304)), 1: ("gfx942", (304,)), 2: ("gfx950", (64, 256))}
    gpus = [types.SimpleNamespace(gfx=gfx) for gfx, _ in info.values()]
    monkeypatch.setattr(chip_info, "_visible_gpus", lambda: gpus)
    monkeypatch.setattr(chip_info, "_tuned_cu_nums_of_device", lambda d: info[d][1])
    core._publish_tuned_cu_fallbacks(module)
    assert calls == [("gfx942", 152, 304), ("gfx950", 64, 256)]
    assert "AITER_TUNED_CU_NUMS" not in os.environ
