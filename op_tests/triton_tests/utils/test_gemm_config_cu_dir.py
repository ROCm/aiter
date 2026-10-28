# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The <arch>/<cu_num>cu/ GEMM config trees: the loader reads one tree whole, and the tuning
scripts plan, seed, resume and install per tree. CPU only: the arch and the CU count are patched.
"""

import json
import os
import sys

import pytest

from aiter.ops.triton.utils import config_utils, device_info, gemm_config_utils
from aiter.ops.triton.utils._triton import arch_info

NAME = "GEMM-A16W16"
FAMILY = "gluon/gemm/gemm_a16w16"


def _bucket(block_m):
    return {"BLOCK_M": block_m, "BLOCK_N": 64, "BLOCK_K": 64}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A gfx1250 default tree with two shape files, and a 96cu tree with its own DEFAULT.json
    and one shape file (N=128). N=512 has a shape file in the default tree only."""
    monkeypatch.setattr(config_utils, "AITER_TRITON_CONFIGS_PATH", str(tmp_path))
    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")

    def write(rel, table):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table))

    write(f"gfx1250/{FAMILY}/DEFAULT.json", {"any": _bucket(1)})
    write(
        f"gfx1250/{FAMILY}/{NAME}-N=128-K=256.json",
        {"M_LEQ_16": _bucket(2), "any": _bucket(3)},
    )
    write(
        f"gfx1250/{FAMILY}/{NAME}-N=512-K=256.json",
        {"M_LEQ_16": _bucket(4), "any": _bucket(5)},
    )
    write(
        f"gfx1250/96cu/{FAMILY}/DEFAULT.json",
        {"M_BOUNDS": [16, 64], "any": _bucket(96)},
    )
    write(
        f"gfx1250/96cu/{FAMILY}/{NAME}-N=128-K=256.json",
        {"M_LEQ_16": _bucket(97), "any": _bucket(98)},
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


def test_cu_tree_serves_its_shape_file(tree, monkeypatch):
    assert _get(monkeypatch, 96, 8, 128, 256) == (_bucket(97), True)
    assert _get(monkeypatch, 96, 4096, 128, 256) == (_bucket(98), False)


def test_cu_tree_never_reads_the_default_tree(tree, monkeypatch):
    # N=512 has a shape file in the default tree only: the 96cu DEFAULT.json serves it
    assert _get(monkeypatch, 96, 8, 512, 256) == (_bucket(96), False)


@pytest.mark.parametrize("cu_num", [None, 192])
def test_no_cu_tree_means_the_default_tree(tree, monkeypatch, cu_num):
    assert _get(monkeypatch, cu_num, 8, 128, 256) == (_bucket(2), True)
    assert _get(monkeypatch, cu_num, 8, 512, 256) == (_bucket(4), True)
    assert _get(monkeypatch, cu_num, 8, 1024, 256) == (_bucket(1), False)


def test_cu_folder_without_default_is_not_a_tree(tree, monkeypatch):
    os.remove(
        f"{config_utils.AITER_TRITON_CONFIGS_PATH}/gfx1250/96cu/{FAMILY}/DEFAULT.json"
    )
    config_utils.load_config_json.cache_clear()
    assert _get(monkeypatch, 96, 8, 128, 256) == (_bucket(2), True)


def test_resolve_config_dir_cu_num(tree):
    base = config_utils.resolve_config_dir("gemm", NAME, backend="gluon")
    cu = config_utils.resolve_config_dir("gemm", NAME, backend="gluon", cu_num=96)
    assert base.endswith(f"/gfx1250/{FAMILY}")
    assert cu.endswith(f"/gfx1250/96cu/{FAMILY}")
    for bad in (0, -96, True, "96", 96.0):
        with pytest.raises(AssertionError):
            config_utils.resolve_config_dir("gemm", NAME, backend="gluon", cu_num=bad)


def test_gemm_config_dir(tree):
    default_dir = config_utils.resolve_config_dir("gemm", NAME, backend="gluon")
    assert gemm_config_utils.gemm_config_dir(NAME, "gluon", None) == default_dir
    assert gemm_config_utils.gemm_config_dir(NAME, "gluon", 192) == default_dir
    assert gemm_config_utils.gemm_config_dir(NAME, "gluon", 96).endswith(
        f"/96cu/{FAMILY}"
    )


@pytest.mark.parametrize("value, expected", [(96, 96), (0, None), (True, None)])
def test_get_config_cu_num(monkeypatch, value, expected):
    monkeypatch.setattr(device_info, "get_num_sms", lambda: value)
    monkeypatch.setattr(config_utils, "_CU_NUM", None)
    monkeypatch.setattr(config_utils, "_CU_NUM_READ", False)
    assert config_utils.get_config_cu_num() == expected


def test_get_config_cu_num_reads_once(monkeypatch):
    calls = []

    def read():
        calls.append(1)
        raise RuntimeError("no device")

    monkeypatch.setattr(device_info, "get_num_sms", read)
    monkeypatch.setattr(config_utils, "_CU_NUM", None)
    monkeypatch.setattr(config_utils, "_CU_NUM_READ", False)
    assert config_utils.get_config_cu_num() is None
    assert config_utils.get_config_cu_num() is None
    assert len(calls) == 1  # an unreadable count means the default tree for the process


@pytest.fixture
def tuning(monkeypatch):
    import aiter

    tuning_dir = os.path.join(
        aiter.__path__[0], "ops", "triton", "utils", "_triton", "tuning"
    )
    monkeypatch.syspath_prepend(tuning_dir)
    import _utils
    import kernels
    import space
    import sweep_configs
    import write_best_configs

    yield kernels, space, sweep_configs, write_best_configs, _utils
    for name in ("_utils", "kernels", "space", "sweep_configs", "write_best_configs"):
        sys.modules.pop(name, None)


def _records(path, records, shape):
    with open(path, "w") as f:
        f.writelines(json.dumps(dict(record, shape=shape)) + "\n" for record in records)


def test_config_tree(tree, tuning):
    kernels, space, _, _, _ = tuning
    spec = kernels.get_spec("gemm_a16w16")
    assert space.config_tree(spec, "gluon").endswith(f"/gfx1250/{FAMILY}")
    assert space.config_tree(spec, "gluon", 96).endswith(f"/gfx1250/96cu/{FAMILY}")
    with pytest.raises(
        SystemExit
    ):  # no 192cu tree: --cu-dir refuses, never creates one
        space.config_tree(spec, "gluon", 192)


def test_plan_reads_the_cu_tree(tree, tuning):
    kernels, space, _, _, _ = tuning
    spec = kernels.get_spec("gemm_a16w16")
    _, buckets, path = space.load_defaults(spec, "gluon", 96)
    assert path.endswith(f"/96cu/{FAMILY}/DEFAULT.json")
    assert buckets["any"] == _bucket(96)
    assert space.load_defaults(spec, "gluon")[1]["any"] == _bucket(1)
    shape_nk = {"N": 1024, "K": 256}
    path, seed = kernels.seed_table(spec, "gluon", shape_nk, 96)
    assert path.endswith(f"/96cu/{FAMILY}/DEFAULT.json")
    assert seed == {"DEFAULT_FALLBACK": True, "M_BOUNDS": [16, 64]}
    path, seed = kernels.seed_table(spec, "gluon", {"N": 128, "K": 256}, 96)
    assert path.endswith(f"/96cu/{FAMILY}/{NAME}-N=128-K=256.json")
    assert seed == {"M_LEQ_16": _bucket(97), "any": _bucket(98)}
    assert kernels.family_bounds(spec, "gluon", shape_nk, 96) == (16, 64)
    assert kernels.family_bounds(spec, "gluon", shape_nk) != (16, 64)


def test_resolve_installed_scopes(tree, tuning, monkeypatch):
    kernels, _, _, _, _ = tuning
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: 96)
    spec = kernels.get_spec("gemm_a16w16")
    shape = {"M": 8, "N": 128, "K": 256}
    assert kernels.resolve_installed(spec, shape, "gluon") == (_bucket(97), True)
    assert kernels.resolve_installed(spec, shape, "gluon", cu_num=None) == (
        _bucket(2),
        True,
    )
    assert kernels.resolve_installed(spec, shape, "gluon", cu_num=96) == (
        _bucket(97),
        True,
    )


def test_baseline_is_the_install_scope(tree, tuning, monkeypatch):
    kernels, _, _, _, _ = tuning
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: 96)
    spec = kernels.get_spec("gemm_a16w16")
    keys = ("BLOCK_M", "BLOCK_N", "BLOCK_K")
    shape = {"M": 8, "N": 128, "K": 256}
    # the default tree's config differs from what this 96-CU device reads: launch it explicitly
    assert kernels.installed_baseline(spec, shape, "gluon", keys) == (
        _bucket(2),
        True,
        [],
        True,
    )
    assert kernels.installed_baseline(spec, shape, "gluon", keys, 96) == (
        _bucket(97),
        True,
        [],
        False,
    )
    config, _, dropped, explicit = kernels.installed_baseline(
        spec, shape, "gluon", keys[:2], 96
    )
    assert (config, dropped, explicit) == (
        {"BLOCK_M": 97, "BLOCK_N": 64},
        ["BLOCK_K"],
        True,
    )


def test_read_records_filters_the_cu_count(tuning, tmp_path):
    _, _, _, _, _utils = tuning
    path = tmp_path / "sweep.jsonl"
    shape = {"M": 16, "N": 128, "K": 256}
    _records(
        path,
        [
            {"status": "ready", "cu_num": 96},
            {"status": "ok", "config": _bucket(7), "us": 5.0, "cu_num": 96},
            {"status": "ok", "config": _bucket(8), "us": 1.0, "cu_num": 192},
            {"status": "ok", "config": _bucket(9), "us": 2.0},
        ],
        shape,
    )
    assert len(_utils.read_records(str(path))) == 3
    assert [r["us"] for r in _utils.read_records(str(path), 96).values()] == [5.0]


@pytest.mark.parametrize("recorded, ok", [(96, True), (None, True), (192, False)])
def test_resume_needs_the_same_cu_count(tuning, tmp_path, recorded, ok):
    import argparse

    _, _, sweep_configs, _, _ = tuning
    out = tmp_path / "sweep.jsonl"
    ready = {"status": "ready", "calls": 24, "replays": 25, "cold_mb": 1024}
    if recorded is not None:
        ready["cu_num"] = recorded
    out.write_text(json.dumps(ready) + "\n" + json.dumps(ready) + "\n")
    args = argparse.Namespace(calls=24, replays=25, cold_mb=1024)
    if ok:
        sweep_configs.check_settings(str(out), args, 96)
    else:
        with pytest.raises(SystemExit):
            sweep_configs.check_settings(str(out), args, 96)


def test_install_with_cu_dir_writes_the_cu_tree_only(
    tree, tuning, tmp_path, monkeypatch
):
    kernels, _, _, write_best_configs, _ = tuning
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: 96)
    monkeypatch.setattr(config_utils, "get_config_cu_num", lambda: 96)
    runs = tmp_path / "runs"
    runs.mkdir()
    shape = {"M": 16, "N": 1024, "K": 256}
    _records(
        runs / "sweep-gfx1250-gluon-gemm_a16w16-M=16-N=1024-K=256.jsonl",
        [
            {"status": "ok", "baseline": True, "config": None, "us": 9.0, "cu_num": 96},
            {"status": "ok", "config": _bucket(7), "us": 5.0, "cu_num": 96},
            {"status": "ok", "config": _bucket(8), "us": 1.0, "cu_num": 192},
        ],
        shape,
    )
    spec = kernels.get_spec("gemm_a16w16")
    target = write_best_configs.install(
        spec, "gluon", {"N": 1024, "K": 256}, str(runs), cu_dir=True
    )
    assert target.endswith(f"/gfx1250/96cu/{FAMILY}/{NAME}-N=1024-K=256.json")
    with open(target) as f:
        table = json.load(f)
    # only the 96-CU record wins, the seed and the "any" are the 96cu tree's own
    assert table == {
        "DEFAULT_FALLBACK": True,
        "M_BOUNDS": [16, 64],
        "M_LEQ_16": _bucket(7),
        "any": _bucket(96),
    }
    default_dir = config_utils.resolve_config_dir("gemm", NAME, backend="gluon")
    assert not os.path.exists(f"{default_dir}/{NAME}-N=1024-K=256.json")


def test_install_without_cu_dir_writes_the_default_tree(
    tree, tuning, tmp_path, monkeypatch
):
    kernels, _, _, write_best_configs, _ = tuning
    monkeypatch.setattr(gemm_config_utils, "get_config_cu_num", lambda: 96)
    monkeypatch.setattr(config_utils, "get_config_cu_num", lambda: 96)
    runs = tmp_path / "runs"
    runs.mkdir()
    shape = {"M": 16, "N": 1024, "K": 256}
    _records(
        runs / "sweep-gfx1250-gluon-gemm_a16w16-M=16-N=1024-K=256.jsonl",
        [{"status": "ok", "config": _bucket(7), "us": 5.0, "cu_num": 96}],
        shape,
    )
    spec = kernels.get_spec("gemm_a16w16")
    target = write_best_configs.install(spec, "gluon", {"N": 1024, "K": 256}, str(runs))
    assert target.endswith(f"/gfx1250/{FAMILY}/{NAME}-N=1024-K=256.json")
    with open(target) as f:
        table = json.load(f)
    assert table == {
        "DEFAULT_FALLBACK": True,
        "M_LEQ_16": _bucket(7),
        "any": _bucket(1),
    }
