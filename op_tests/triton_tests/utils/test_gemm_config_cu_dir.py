# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json

import pytest

from aiter.ops.triton.utils import config_utils, device_info, gemm_config_utils
from aiter.ops.triton.utils._triton import arch_info

NAME = "GEMM-A16W16"


def _bucket(block_m):
    return {"BLOCK_M": block_m, "BLOCK_N": 64, "BLOCK_K": 64}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(config_utils, "AITER_TRITON_CONFIGS_PATH", str(tmp_path))
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")

    def write(rel, table):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table))

    base = "gfx1250/gluon/gemm/gemm_a16w16"
    write(f"{base}/DEFAULT.json", {"any": _bucket(1)})
    write(
        f"{base}/{NAME}-N=128-K=256.json", {"M_LEQ_16": _bucket(2), "any": _bucket(3)}
    )
    write(
        f"{base}/{NAME}-N=512-K=256.json", {"M_LEQ_16": _bucket(4), "any": _bucket(5)}
    )
    write(
        f"gfx1250/96cu/gluon/gemm/gemm_a16w16/{NAME}-N=128-K=256.json",
        {"M_LEQ_16": _bucket(96), "any": _bucket(97)},
    )

    def clear():
        config_utils.load_config_json.cache_clear()
        gemm_config_utils._get_gemm_config_cached.cache_clear()

    clear()
    yield write
    clear()


def _get(monkeypatch, cu_num, M, N, K):
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: cu_num)
    return gemm_config_utils.get_gemm_config(NAME, M, N, K, backend="gluon")


def test_cu_file_wins(tree, monkeypatch):
    assert _get(monkeypatch, 96, 8, 128, 256) == (_bucket(96), True)
    assert _get(monkeypatch, 96, 4096, 128, 256) == (_bucket(97), False)


def test_missing_cu_file_uses_cu_agnostic_file(tree, monkeypatch):
    assert _get(monkeypatch, 96, 8, 512, 256) == (_bucket(4), True)
    assert _get(monkeypatch, 96, 8, 1024, 256) == (_bucket(1), False)


@pytest.mark.parametrize("cu_num", [None, 192])
def test_other_cu_num_ignores_cu_folder(tree, monkeypatch, cu_num):
    assert _get(monkeypatch, cu_num, 8, 128, 256) == (_bucket(2), True)


def test_cu_default(tree, monkeypatch):
    tree("gfx1250/96cu/gluon/gemm/gemm_a16w16/DEFAULT.json", {"any": _bucket(98)})
    assert _get(monkeypatch, 96, 8, 1024, 256) == (_bucket(98), False)
    assert _get(monkeypatch, 192, 8, 1024, 256) == (_bucket(1), False)


def test_resolve_config_dir_cu_num(tree):
    base = config_utils.resolve_config_dir("gemm", NAME, backend="gluon")
    cu = config_utils.resolve_config_dir("gemm", NAME, backend="gluon", cu_num=96)
    assert base.endswith("/gfx1250/gluon/gemm/gemm_a16w16")
    assert cu.endswith("/gfx1250/96cu/gluon/gemm/gemm_a16w16")
    for bad in (0, -96, True, "96", 96.0):
        with pytest.raises(AssertionError):
            config_utils.resolve_config_dir("gemm", NAME, backend="gluon", cu_num=bad)


@pytest.mark.parametrize("value, expected", [(96, 96), (0, None), (True, None)])
def test_get_config_cu_num(monkeypatch, value, expected):
    monkeypatch.setattr(device_info, "get_num_sms", lambda: value)
    config_utils.get_config_cu_num.cache_clear()
    try:
        assert config_utils.get_config_cu_num() == expected
    finally:
        config_utils.get_config_cu_num.cache_clear()


def test_get_config_cu_num_unreadable(monkeypatch):
    def fail():
        raise RuntimeError("no device")

    monkeypatch.setattr(device_info, "get_num_sms", fail)
    config_utils.get_config_cu_num.cache_clear()
    try:
        assert config_utils.get_config_cu_num() is None
    finally:
        config_utils.get_config_cu_num.cache_clear()
