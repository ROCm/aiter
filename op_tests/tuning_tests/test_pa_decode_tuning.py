# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU contracts for PA's real FlyDSL autotuner and persistent cache."""

import importlib.util
import inspect
import os
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "aiter/ops/flydsl/pa_decode_tuning.py"
MODULE_NAME = "_pa_decode_tuning_cpu_test"


def load_module(monkeypatch):
    # A stable module name preserves the native cache namespace across reloads.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tuning(monkeypatch):
    module = load_module(monkeypatch)
    yield module
    factory = getattr(module, "_get_runtime_tuner", None)
    if factory is not None:
        factory.cache_clear()


@pytest.fixture
def autotune(monkeypatch, tmp_path):
    at = pytest.importorskip("flydsl.autotune")
    monkeypatch.setattr(at, "torch", None)
    monkeypatch.setattr(at, "_device_fingerprint", lambda *a, **k: "cpu-test-device")
    monkeypatch.setattr(at, "_toolchain_fingerprint", lambda: "cpu-test-toolchain")
    monkeypatch.setattr(at, "_env_fingerprint", lambda: ())
    monkeypatch.setenv("FLYDSL_AUTOTUNE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("FLYDSL_AUTOTUNE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("FLYDSL_AUTOTUNE", raising=False)
    return at


@pytest.fixture
def native(autotune, tuning, tmp_path):
    if "validate_hook" not in inspect.signature(autotune.Autotuner).parameters:
        pytest.skip("requires FlyDSL Autotuner.validate_hook (0.3.4.1 or newer)")
    tuning._get_runtime_tuner.cache_clear()
    return types.SimpleNamespace(
        at=autotune, module=tuning, cache_dir=tmp_path / "cache"
    )


def shape(tuning, **changes):
    options = {
        "batch_size": 4,
        "context_length": 4096,
        "num_query_heads": 16,
        "num_kv_heads": 1,
        "head_dim": 128,
        "seed": 0,
    }
    options.update(changes)
    return tuning.make_shape(**options)


class FakeSession:
    """Only execution and timing are fake; selection and cache stay native."""

    rounds = 3

    def __init__(self, native, times=None, *, forbid=False, reject=()):
        self.config_type = native.at.Config
        self.select_impl = native.module._select_config
        self.times = {128: 0.1, 512: 0.2} if times is None else times
        self.forbid = forbid
        self.reject = set(reject)
        self.events = []
        self.last_budget = None

    def record(self, event, *values):
        assert not self.forbid, f"cache hit unexpectedly called session.{event}"
        self.events.append((event, *values))

    def configs(self):
        self.record("configs")
        return [self.config_type(workgroup_budget=b) for b in self.times]

    @contextmanager
    def validate(self, budget):
        self.record("validate", budget)
        if budget in self.reject:
            raise ArithmeticError("synthetic candidate accuracy failure")
        yield
        self.record("validated", budget)

    def launch(self, budget):
        self.record("launch", budget)
        assert budget in self.times
        self.last_budget = budget

    def benchmark(self, fn, *, warmup, rep):
        self.record("benchmark", warmup, rep)
        assert type(warmup) is int and warmup >= 0
        assert type(rep) is int and rep > 0
        for _ in range(warmup + rep):
            fn()
        return self.times[self.last_budget]

    def select(self, results):
        self.record("select")
        return self.select_impl(results)


@contextmanager
def forbid_search(native, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("runtime cache lookup attempted to benchmark a candidate")

    with monkeypatch.context() as patch:
        patch.setattr(native.at.Autotuner, "_bench_one", unexpected)
        yield


def search(native, problem, session=None, **kwargs):
    session = FakeSession(native) if session is None else session
    config = native.module._resolve_config(
        problem, "gfx950", 256, session=session, **kwargs
    )
    assert isinstance(config, native.at.Config)
    return config.kwargs["workgroup_budget"], session


def cache_bytes(directory):
    return {p.name: p.read_bytes() for p in directory.rglob("*.json")}


def test_cli_runs_without_site_packages(tmp_path):
    env = dict(os.environ, FLYDSL_AUTOTUNE_CACHE_DIR=str(tmp_path / "cache"))
    env.pop("PYTHONPATH", None)
    for args in (
        ["--help"],
        [
            "--dry-run",
            "--shape",
            "16,1,128",
            "--architecture",
            "gfx950",
            "--num-cu",
            "256",
            "--budgets",
            "128,512",
        ],
    ):
        result = subprocess.run(
            [sys.executable, "-S", str(SOURCE), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip()
    assert not cache_bytes(tmp_path)


def test_select_config_uses_97_percent_performance_and_returns_measured_pair(
    tuning, autotune
):
    small = autotune.Config(workgroup_budget=128)
    fast = autotune.Config(workgroup_budget=512)
    # 3.05% extra latency is still within 97% of the best performance.
    results = [(fast, 0.1), (small, 0.10305)]
    config, elapsed = tuning._select_config(results)
    assert config is small and elapsed == results[1][1]
    assert tuning._select_config([(small, 0.104), (fast, 0.1)]) == (fast, 0.1)
    config, elapsed = tuning._select_config([(fast, 0.1), (small, 0.1)])
    assert config is small and elapsed == 0.1


def test_runtime_miss_does_not_search_or_poison_later_search(native, monkeypatch):
    problem = shape(native.module)
    with forbid_search(native, monkeypatch):
        assert native.module.get_cached_budget(problem, "gfx950", 256) == 512
    assert not cache_bytes(native.cache_dir)
    budget, session = search(native, problem)
    assert budget == 128
    assert any(event[0] == "benchmark" for event in session.events)


def test_hit_ignores_session_and_nonstatic_input_metadata(native, monkeypatch):
    problem = shape(native.module)
    assert search(native, problem)[0] == 128
    before = cache_bytes(native.cache_dir)
    assert before
    poison = FakeSession(native, times={256: 0.01, 512: 0.2}, forbid=True)
    changed_samples = shape(native.module, seed=11, length_mode="varlen")
    with forbid_search(native, monkeypatch):
        assert search(native, changed_samples, poison)[0] == 128
        assert native.module.get_cached_budget(problem, "gfx950", 256) == 128
    assert poison.events == []
    assert cache_bytes(native.cache_dir) == before


def test_normalized_per_token_scales_reuse_offline_native_cache(native, monkeypatch):
    torch = pytest.importorskip("torch")
    problem = shape(
        native.module,
        batch_size=2,
        context_length=32,
        query_length=2,
        num_kv_heads=2,
        page_size=16,
    )
    inputs = native.module._make_inputs(
        torch, problem, torch.device("cpu"), torch.float8_e4m3fn
    )
    packed = tuple(inputs[name] for name in ("query", "key", "value"))
    raw_scales = (inputs["key_scale"], inputs["value_scale"])
    # pa_decode.normalize_scale removes this trailing singleton before launch.
    normalized_scales = tuple(scale.squeeze(-1) for scale in raw_scales)
    assert all(scale.dim() == 4 for scale in raw_scales)
    assert all(scale.dim() == 3 for scale in normalized_scales)
    synthetic = native.module._synthetic_storage_key(problem)
    offline = native.module.storage_key(*packed, *raw_scales)
    runtime = native.module.storage_key(*packed, *normalized_scales)
    assert synthetic == offline == runtime

    assert search(native, problem)[0] == 128
    before = cache_bytes(native.cache_dir)
    assert before
    native.module._get_runtime_tuner.cache_clear()
    with forbid_search(native, monkeypatch):
        assert (
            native.module.get_cached_budget(problem, "gfx950", 256, storage_key=runtime)
            == 128
        )
    assert cache_bytes(native.cache_dir) == before


def test_fresh_module_and_tuner_read_the_native_disk_cache(native, monkeypatch):
    problem = shape(native.module)
    assert search(native, problem)[0] == 128
    assert cache_bytes(native.cache_dir)
    fresh = load_module(monkeypatch)
    assert fresh is not native.module
    with forbid_search(native, monkeypatch):
        assert fresh.get_cached_budget(problem, "gfx950", 256) == 128
        first = fresh._get_runtime_tuner(str(native.cache_dir))
        assert isinstance(first, native.at.Autotuner)
        fresh._get_runtime_tuner.cache_clear()
        assert fresh.get_cached_budget(problem, "gfx950", 256) == 128
        assert fresh._get_runtime_tuner(str(native.cache_dir)) is not first
    fresh._get_runtime_tuner.cache_clear()


def test_shape_device_source_and_storage_have_separate_keys(native, monkeypatch):
    problem = shape(native.module)
    storage = ("cpu-layout-a",)
    assert search(native, problem, storage_key=storage)[0] == 128
    variants = [
        (shape(native.module, context_length=8192), "gfx950", 256, storage),
        (shape(native.module, per_token=False), "gfx950", 256, storage),
        (shape(native.module, window=257), "gfx950", 256, storage),
        (shape(native.module, dtype="float16"), "gfx950", 256, storage),
        (shape(native.module, trans_v=False), "gfx950", 256, storage),
        (problem, "gfx942", 256, storage),
        (problem, "gfx950", 120, storage),
        (problem, "gfx950", 256, ("cpu-layout-b",)),
    ]
    with forbid_search(native, monkeypatch):
        for changed, arch, cu, layout in variants:
            assert (
                native.module.get_cached_budget(changed, arch, cu, storage_key=layout)
                == 2 * cu
            )
        monkeypatch.setattr(native.module, "_implementation_hash", lambda: "new-source")
        assert (
            native.module.get_cached_budget(problem, "gfx950", 256, storage_key=storage)
            == 512
        )


def test_force_search_is_offline_only(native, monkeypatch):
    problem = shape(native.module)
    assert search(native, problem)[0] == 128
    before = cache_bytes(native.cache_dir)
    monkeypatch.setenv("FLYDSL_AUTOTUNE", "1")
    with forbid_search(native, monkeypatch):
        assert native.module.get_cached_budget(problem, "gfx950", 256) == 512
    assert cache_bytes(native.cache_dir) == before
    forced = FakeSession(native, times={256: 0.1, 512: 0.2})
    assert search(native, problem, forced)[0] == 256
    assert any(event[0] == "benchmark" for event in forced.events)
    monkeypatch.delenv("FLYDSL_AUTOTUNE")
    # Native cache instances do not watch disk updates from other instances.
    native.module._get_runtime_tuner.cache_clear()
    with forbid_search(native, monkeypatch):
        assert native.module.get_cached_budget(problem, "gfx950", 256) == 256


def test_invalid_native_cache_configs_fall_back_safely(native, monkeypatch):
    problem = shape(native.module)
    assert search(native, problem)[0] == 128
    writer = native.module._get_runtime_tuner(str(native.cache_dir))
    assert isinstance(writer, native.at.Autotuner) and len(writer.cache) == 1
    key = next(iter(writer.cache))
    invalid = [
        native.at.Config(workgroup_budget=value) for value in (0, -1, True, 1.5, "128")
    ]
    invalid.extend(
        [
            native.at.Config(workgroup_budget=128, foo=1),
            native.at.Config(workgroup_budget=128, num_warps=8),
            native.at.Config(workgroup_budget=128, waves_per_eu=2),
        ]
    )
    with forbid_search(native, monkeypatch):
        for config in invalid:
            # Use FlyDSL's serializer and disk reader; do not emulate its format.
            writer.cache[key] = config
            writer._save_disk_cache()
            native.module._get_runtime_tuner.cache_clear()
            assert native.module.get_cached_budget(problem, "gfx950", 256) == 512


def test_validation_hook_excludes_a_faster_incorrect_candidate(native, monkeypatch):
    problem = shape(native.module)
    session = FakeSession(native, times={128: 0.01, 256: 0.1, 512: 0.2}, reject={128})
    assert search(native, problem, session)[0] == 256
    assert ("validate", 128) in session.events
    assert ("launch", 128) not in session.events
    assert ("validated", 256) in session.events
    with forbid_search(native, monkeypatch):
        assert native.module.get_cached_budget(problem, "gfx950", 256) == 256
