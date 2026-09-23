# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Offline PA compilation and persistent-cache regressions; no GPU required."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

_KERNELS = Path(__file__).resolve().parents[1] / "aiter/ops/flydsl/kernels"
_CASES = {
    "partitioned": {},
    "mtp4": {"query_length": 4, "query_splits": 1},
    "mtp4_wide": {
        "query_length": 4,
        "query_splits": 1,
        "wide_kv_addressing": True,
    },
    "mtp4_buffer": {
        "query_length": 4,
        "query_splits": 1,
        "wide_kv_addressing": True,
        "kv_buffer_u32": True,
    },
    "planned_dense_split2": {
        "query_group_size": 8,
        "num_seqs": 1,
        "num_kv_heads": 8,
        "num_partitions": 256,
        "query_length": 4,
        "use_work_plan": True,
        "work_capacity": 32,
        "max_context_length": 4096,
    },
    "planned_odd_rows": {
        "query_group_size": 9,
        "block_size": 16,
        "num_kv_heads": 2,
        "num_partitions": 3,
        "query_length": 2,
        "query_splits": 2,
        "use_work_plan": True,
        "work_capacity": 8,
        "sliding_window": 1,
    },
}


def _compile_worker(kernels, output, cases):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    # Import the real kernel and dependencies without aiter's top-level GPU
    # discovery/native-extension imports. These package shells live only here.
    for name in ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels"):
        package = types.ModuleType(name)
        package.__path__ = [str(kernels)]
        sys.modules[name] = package
    from aiter.ops.flydsl.kernels.pa_decode_kernel import compile_pa_decode_tile

    results = {}
    for name in cases:
        options = {
            "head_dim": 128,
            "query_group_size": 16,
            "block_size": 128,
            "num_seqs": 4,
            "num_kv_heads": 1,
            "num_compute_units": 304,
            "num_partitions": 8,
            "query_dtype": "bf16",
            "per_token_kv": True,
            **_CASES[name],
        }
        launch = compile_pa_decode_tile(**options)["launch"]
        pointer_types = (
            fx.BFloat16,
            fx.Float32,
            fx.Float32,
            fx.BFloat16,
            fx.BFloat16,
            fx.Float8E4M3FN,
            fx.Float8E4M3FN,
            fx.Int32,
            fx.Int32,
            fx.Float32,
            fx.Float32,
            fx.Float32,
        )
        heads, page = options["num_kv_heads"], options["block_size"]
        row_stride = heads * options["query_group_size"] * options["head_dim"]
        # COMPILE_ONLY means these null pointers are never dereferenced.
        args = [flyc.from_c_void_p(dtype, 0) for dtype in pointer_types]
        args += [
            128,
            options["num_seqs"],
            heads,
            heads * page,
            page,
            row_stride,
            options["head_dim"],
            row_stride,
            options["head_dim"],
            flyc.from_c_void_p(fx.Int32, 0),
            options.get("work_capacity", 0),
            fx.Stream(None),
        ]
        flyc.compile(launch, *args)
        compiled = launch._last_compiled
        artifact = (
            compiled[1]
            if compiled is not None
            else next(iter(launch._mem_cache.values()))
        )
        results[name] = {
            "key": launch.manager_key,
            "ir_hash": hashlib.sha256(artifact.ir.encode()).hexdigest(),
            "compiled": compiled is not None,
            "cache_hits": launch.cache_info().hits,
        }
    output.write_text(json.dumps(results))


def _run_compile(kernels, cache, output, cases=tuple(_CASES)):
    env = {
        **os.environ,
        "COMPILE_ONLY": "1",
        "ARCH": "gfx950",
        "FLYDSL_GPU_ARCH": "gfx950",
        "ROCR_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "CUDA_VISIBLE_DEVICES": "-1",
        "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        "FLYDSL_RUNTIME_CACHE_DIR": str(cache),
        "FLYDSL_RUNTIME_RUN_ONLY": "0",
        "FLYDSL_COMPILE_LLVM_DIR": "",
        "FLYDSL_DUMP_IR": "0",
        "FLYDSL_DEBUG_ENABLE_DEBUG_INFO": "0",
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(kernels),
            str(output),
            *cases,
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(output.read_text())


@pytest.fixture(scope="module")
def compiled_cases(tmp_path_factory):
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    directory = tmp_path_factory.mktemp("pa_decode_compile")
    cache = directory / "cache"
    cold = _run_compile(_KERNELS, cache, directory / "cold.json")
    warm = _run_compile(_KERNELS, cache, directory / "warm.json")
    return cold, warm, cache


def test_specializations_have_distinct_cache_keys(compiled_cases):
    cold, _, _ = compiled_cases
    assert len({entry["key"] for entry in cold.values()}) == len(_CASES)


@pytest.mark.parametrize("case", _CASES)
def test_offline_compilation_and_cache_reuse(compiled_cases, case):
    cold, warm, _ = compiled_cases
    assert cold[case]["compiled"]
    assert not warm[case]["compiled"]
    assert warm[case]["cache_hits"] > 0
    assert cold[case]["key"] == warm[case]["key"]
    assert cold[case]["ir_hash"] == warm[case]["ir_hash"]


def test_helper_edit_invalidates_cached_binary(compiled_cases, tmp_path):
    cold, _, cache = compiled_cases
    kernels = tmp_path / "kernels"
    shutil.copytree(
        _KERNELS / "pa_decode",
        kernels / "pa_decode",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in (
        "pa_decode_kernel.py",
        "buffer_ops.py",
        "dpp_utils.py",
        "kernels_common.py",
        "tensor_shim.py",
        "utils.py",
    ):
        shutil.copy2(_KERNELS / name, kernels / name)
    relocated = _run_compile(
        kernels, cache, tmp_path / "relocated.json", ("partitioned",)
    )
    assert not relocated["partitioned"]["compiled"]
    assert relocated["partitioned"]["key"] == cold["partitioned"]["key"]

    # Change actual generated arithmetic in a helper, leaving the factory and
    # specialization untouched. The previous binary must not be reused.
    softmax = kernels / "pa_decode/op_softmax.py"
    source = softmax.read_text()
    epsilon = "1e-8 / self.traits.FP8_MAX"
    assert source.count(epsilon) == 1
    softmax.write_text(source.replace(epsilon, "2e-8 / self.traits.FP8_MAX"))
    edited = _run_compile(kernels, cache, tmp_path / "edited.json", ("partitioned",))
    assert edited["partitioned"]["compiled"]
    assert edited["partitioned"]["key"] != cold["partitioned"]["key"]
    assert edited["partitioned"]["ir_hash"] != cold["partitioned"]["ir_hash"]


if __name__ == "__main__":
    _compile_worker(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
