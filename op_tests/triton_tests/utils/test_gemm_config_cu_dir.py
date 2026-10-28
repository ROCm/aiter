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


@pytest.fixture
def tuning(monkeypatch):
    import os
    import sys

    import aiter

    tuning_dir = os.path.join(
        aiter.__path__[0], "ops", "triton", "utils", "_triton", "tuning"
    )
    monkeypatch.syspath_prepend(tuning_dir)
    import kernels
    import space
    import sweep_configs
    import write_best_configs

    yield kernels, space, sweep_configs, write_best_configs
    for name in ("kernels", "space", "sweep_configs", "write_best_configs", "_utils"):
        sys.modules.pop(name, None)


def test_tuning_defaults_cu_first(tree, tuning):
    kernels, space, _, _ = tuning
    tree(
        "gfx1250/96cu/gluon/gemm/gemm_a16w16/DEFAULT.json",
        {"M_BOUNDS": [16, 64], "any": _bucket(98)},
    )
    spec = kernels.get_spec("gemm_a16w16")
    _, buckets, path = space.load_defaults(spec, "gluon", cu_num=96)
    assert path.endswith("/96cu/gluon/gemm/gemm_a16w16/DEFAULT.json")
    assert buckets["any"] == _bucket(98)
    assert space.load_defaults(spec, "gluon")[1]["any"] == _bucket(1)
    path, seed = kernels.seed_table(spec, "gluon", {"N": 1024, "K": 256}, cu_num=96)
    assert path.endswith("/96cu/gluon/gemm/gemm_a16w16/DEFAULT.json")
    assert seed == {"DEFAULT_FALLBACK": True, "M_BOUNDS": [16, 64]}


def test_install_ignores_other_cu_records(tree, tuning, tmp_path):
    kernels, _, _, write_best_configs = tuning
    runs = tmp_path / "runs"
    runs.mkdir()
    shape = {"M": 16, "N": 128, "K": 256}
    records = [
        {"status": "ok", "baseline": True, "config": None, "us": 9.0, "cu_num": 96},
        {"status": "ok", "config": _bucket(96), "us": 5.0, "cu_num": 96},
        {"status": "ok", "config": _bucket(192), "us": 1.0, "cu_num": 192},
        {"status": "ok", "config": _bucket(7), "us": 2.0},
    ]
    with open(
        runs / "sweep-gfx1250-gluon-gemm_a16w16-M=16-N=128-K=256.jsonl", "w"
    ) as f:
        f.writelines(json.dumps(dict(record, shape=shape)) + "\n" for record in records)
    spec = kernels.get_spec("gemm_a16w16")
    shape_nk = {"N": 128, "K": 256}
    load = write_best_configs.load_winners
    assert load(str(runs), "gfx1250", "gluon", spec, shape_nk)[16].record["us"] == 1.0
    winner = load(str(runs), "gfx1250", "gluon", spec, shape_nk, cu_num=96)[16]
    assert (winner.record["us"], winner.baseline["us"]) == (5.0, 9.0)
    assert (
        load(str(runs), "gfx1250", "gluon", spec, shape_nk, cu_num=192)[16].baseline
        is None
    )


@pytest.mark.parametrize("recorded, ok", [(96, True), (None, True), (192, False)])
def test_resume_needs_same_cu_num(tuning, tmp_path, recorded, ok):
    import argparse

    _, _, sweep_configs, _ = tuning
    out = tmp_path / "sweep.jsonl"
    ready = {"status": "ready", "calls": 24, "replays": 25, "cold_mb": 1024}
    if recorded is not None:
        ready["cu_num"] = recorded
    out.write_text(json.dumps(ready) + "\n")
    args = argparse.Namespace(calls=24, replays=25, cold_mb=1024)
    if ok:
        sweep_configs.check_settings(str(out), args, 96)
    else:
        with pytest.raises(SystemExit):
            sweep_configs.check_settings(str(out), args, 96)


def test_resume_checks_every_ready_record(tuning, tmp_path):
    import argparse

    _, _, sweep_configs, _ = tuning
    out = tmp_path / "sweep.jsonl"
    ready = {"status": "ready", "calls": 24, "replays": 25, "cold_mb": 1024}
    out.write_text(json.dumps(ready) + "\n" + json.dumps(dict(ready, cu_num=96)) + "\n")
    args = argparse.Namespace(calls=24, replays=25, cold_mb=1024)
    with pytest.raises(SystemExit):
        sweep_configs.check_settings(str(out), args, 192)


def test_install_keeps_cu_timing_of_retimed_config(tree, tuning, tmp_path):
    kernels, _, _, write_best_configs = tuning
    runs = tmp_path / "runs"
    runs.mkdir()
    shape = {"M": 16, "N": 128, "K": 256}
    records = [
        {"status": "ok", "config": _bucket(96), "us": 5.0, "cu_num": 96},
        {"status": "ok", "config": _bucket(96), "us": 1.0, "cu_num": 192},
    ]
    with open(
        runs / "sweep-gfx1250-gluon-gemm_a16w16-M=16-N=128-K=256.jsonl", "w"
    ) as f:
        f.writelines(json.dumps(dict(record, shape=shape)) + "\n" for record in records)
    spec = kernels.get_spec("gemm_a16w16")
    winners = write_best_configs.load_winners(
        str(runs), "gfx1250", "gluon", spec, {"N": 128, "K": 256}, cu_num=96
    )
    assert winners[16].record["us"] == 5.0


def test_resolve_installed_scope(tree, tuning, monkeypatch):
    kernels, _, _, _ = tuning
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: 96)
    spec = kernels.get_spec("gemm_a16w16")
    shape = {"M": 8, "N": 128, "K": 256}
    assert kernels.resolve_installed(spec, shape, "gluon") == (_bucket(96), True)
    assert kernels.resolve_installed(spec, shape, "gluon", cu_num=None) == (
        _bucket(2),
        True,
    )
    assert kernels.resolve_installed(spec, shape, "gluon", cu_num=96) == (
        _bucket(96),
        True,
    )


def test_family_bounds_cu_first(tree, tuning):
    kernels, _, _, _ = tuning
    tree(
        "gfx1250/96cu/gluon/gemm/gemm_a16w16/DEFAULT.json",
        {"M_BOUNDS": [16, 64], "any": _bucket(98)},
    )
    spec = kernels.get_spec("gemm_a16w16")
    shape = {"N": 1024, "K": 256}
    assert kernels.family_bounds(spec, "gluon", shape, cu_num=96) == (16, 64)
    assert kernels.family_bounds(spec, "gluon", shape) != (16, 64)


def test_fallback_seed_takes_default_m_bounds(tree, tuning, monkeypatch):
    kernels, _, _, _ = tuning
    base = "gfx1250/gluon/gemm/gemm_a16w16"
    tree(f"{base}/DEFAULT.json", {"M_BOUNDS": [16, 64], "any": _bucket(1)})
    tree(
        f"{base}/{NAME}-N=1024-K=256.json",
        {"DEFAULT_FALLBACK": True, "M_LEQ_64": _bucket(6)},
    )
    spec = kernels.get_spec("gemm_a16w16")
    shape = {"N": 1024, "K": 256}
    _, seed = kernels.seed_table(spec, "gluon", shape)
    assert seed["M_BOUNDS"] == [16, 64]
    assert kernels.family_bounds(spec, "gluon", shape) == (16, 64)
    assert _get(monkeypatch, None, 32, 1024, 256) == (_bucket(6), True)
