# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from aiter.aot.flydsl import common
from aiter.aot.flydsl.spec import AotSpec, AotSpecRegistry, register_default_specs


def _install_fake_spec_module(monkeypatch, name: str, jobs, compiler=None) -> str:
    module_name = f"_aiter_test_{name}"
    module = ModuleType(module_name)
    module.get_aot_jobs = lambda: jobs
    module.compile_one_config = compiler or (
        lambda **job: {**job, "compile_time": 0.01}
    )
    monkeypatch.setitem(sys.modules, module_name, module)
    return module_name


def test_default_specs_are_declarative_and_ordered():
    registry = AotSpecRegistry()
    assert register_default_specs(registry) == 6
    assert [spec.name for spec in registry.get_all_specs()] == [
        "moe",
        "mxfp4_moe",
        "gemm",
        "grouped_moe",
        "chunk_gdn_h",
        "mega_moe",
    ]


def test_registry_is_idempotent_but_rejects_name_collisions():
    registry = AotSpecRegistry()
    spec = AotSpec("test", "test.module")
    assert registry.register(spec) is spec
    assert registry.register(spec) is spec

    with pytest.raises(ValueError, match="already registered"):
        registry.register(AotSpec("test", "other.module"))


def test_spec_collects_copies_and_compiles_jobs(monkeypatch):
    source_job = {"kernel_name": "kernel", "value": 3}
    module_name = _install_fake_spec_module(monkeypatch, "spec", [source_job])
    spec = AotSpec("test", module_name)

    jobs = spec.collect_jobs()
    assert jobs == [source_job]
    assert jobs[0] is not source_job
    assert spec.compile(jobs[0]) == {
        "kernel_name": "kernel",
        "value": 3,
        "compile_time": 0.01,
    }


def test_spec_validates_job_and_result_types(monkeypatch):
    module_name = _install_fake_spec_module(monkeypatch, "bad_job", ["not-a-dict"])
    with pytest.raises(TypeError, match="job 0 must be a dict"):
        AotSpec("bad_job", module_name).collect_jobs()

    module_name = _install_fake_spec_module(
        monkeypatch,
        "bad_result",
        [{"kernel_name": "kernel"}],
        compiler=lambda **_job: None,
    )
    with pytest.raises(TypeError, match="compiler must return a dict"):
        AotSpec("bad_result", module_name).compile({"kernel_name": "kernel"})


def test_run_aot_uses_specs_and_restores_cache_env(monkeypatch, tmp_path):
    module_name = _install_fake_spec_module(
        monkeypatch,
        "run",
        [{"kernel_name": "kernel"}],
    )
    spec = AotSpec("test", module_name)
    observed = {}

    def fake_pool(
        worker_specs,
        max_workers,
        kernel_timeout,
        max_retries,
        result_dir,
        *,
        start_method,
    ):
        observed.update(
            worker_specs=worker_specs,
            max_workers=max_workers,
            kernel_timeout=kernel_timeout,
            max_retries=max_retries,
            result_dir=result_dir,
            start_method=start_method,
            aot_import=os.environ.get("AITER_AOT_IMPORT"),
            cache_dir=os.environ.get("FLYDSL_RUNTIME_CACHE_DIR"),
        )
        return [{"kernel_name": "kernel", "compile_time": 0.01}]

    monkeypatch.setattr(common, "_run_file_pool", fake_pool)
    monkeypatch.setenv("AITER_AOT_IMPORT", "original-mode")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", "original-cache")
    monkeypatch.setenv("AITER_FLYDSL_AOT_START_METHOD", "spawn")

    cache_dir = tmp_path / "cache"
    common.run_aot(str(cache_dir), specs=(spec,))

    assert observed["start_method"] == "spawn"
    assert observed["aot_import"] == "1"
    assert observed["cache_dir"] == str(cache_dir)
    assert observed["worker_specs"][0][1] == {
        "spec": spec,
        "job": {"kernel_name": "kernel"},
    }
    assert os.environ["FLYDSL_RUNTIME_CACHE_DIR"] == "original-cache"
    assert os.environ["AITER_AOT_IMPORT"] == "original-mode"
    assert not (cache_dir / ".aot_results").exists()


def test_run_aot_reports_spec_name_on_failure(monkeypatch, tmp_path):
    module_name = _install_fake_spec_module(
        monkeypatch,
        "failure",
        [{"kernel_name": "broken"}],
    )
    spec = AotSpec("family", module_name)
    monkeypatch.setattr(common, "_run_file_pool", lambda *_args, **_kwargs: [None])

    with pytest.raises(AssertionError, match="family broken worker died"):
        common.run_aot(str(tmp_path / "cache"), specs=(spec,))


def test_invalid_start_method_is_rejected(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_AOT_START_METHOD", "not-a-method")
    with pytest.raises(ValueError, match="AITER_FLYDSL_AOT_START_METHOD"):
        common.get_start_method()


def test_spawn_worker_resolves_spec_in_child(tmp_path):
    spec = AotSpec(
        "fake",
        "aiter.aot.flydsl.tests._fake_aot_module",
    )
    job = spec.collect_jobs()[0]
    label = common.JobLabel("fake", "fake_kernel")

    results = common._run_file_pool(
        [(common._compile_aot_job, {"spec": spec, "job": job}, label)],
        max_workers=1,
        kernel_timeout=30,
        max_retries=0,
        result_dir=str(tmp_path),
        start_method="spawn",
    )

    assert results == [{"kernel_name": "fake_kernel", "value": 7, "compile_time": 0.01}]


def test_legacy_parallel_entry_uses_spawn_with_importable_worker(monkeypatch):
    from aiter.aot.flydsl.tests._fake_aot_module import compile_one_config

    monkeypatch.setenv("AITER_FLYDSL_AOT_WORKERS", "1")
    monkeypatch.setenv("AITER_FLYDSL_AOT_START_METHOD", "spawn")
    results = common.run_jobs_parallel(
        compile_one_config,
        [{"kernel_name": "legacy_kernel", "value": 11}],
    )

    assert results == [
        {"kernel_name": "legacy_kernel", "value": 11, "compile_time": 0.01}
    ]


def test_spawn_worker_serializes_deterministic_errors(tmp_path):
    from aiter.aot.flydsl.tests._fake_aot_module import compile_one_config

    label = common.JobLabel("fake", "broken")
    results = common._run_file_pool(
        [(compile_one_config, {"kernel_name": "broken", "value": -1}, label)],
        max_workers=1,
        kernel_timeout=30,
        max_retries=2,
        result_dir=str(tmp_path),
        start_method="spawn",
    )

    assert results == [
        {
            "kernel_name": "broken",
            "compile_time": None,
            "error": "ValueError: negative test value",
        }
    ]
