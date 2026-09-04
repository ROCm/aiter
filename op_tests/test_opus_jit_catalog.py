# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import torch


def _missing_tuned_kernel(*_args, **_kwargs):
    raise RuntimeError("Kernel id 312 not found in a16w16 bf16 tune lookup table")


def test_jit_dependency_fingerprint_tracks_contents_and_glob_members(tmp_path):
    from aiter.jit.core import _jit_dependency_fingerprint

    first = tmp_path / "first.csv"
    first.write_text("solidx\n315\n")
    pattern = str(tmp_path / "*.csv")

    initial = _jit_dependency_fingerprint([pattern])
    first.write_text("solidx\n312\n")
    assert _jit_dependency_fingerprint([pattern]) != initial

    after_edit = _jit_dependency_fingerprint([pattern])
    second = tmp_path / "second.csv"
    second.write_text("solidx\n315\n")
    assert _jit_dependency_fingerprint([pattern]) != after_edit

    listed = os.pathsep.join((str(first), str(second)))
    before_listed_edit = _jit_dependency_fingerprint([listed])
    second.write_text("solidx\n316\n")
    assert _jit_dependency_fingerprint([listed]) != before_listed_edit


def test_jit_dependency_invalidation_uses_module_build_lock(tmp_path, monkeypatch):
    import aiter.jit.core as jit_core

    module = "test_dependency_module"
    monkeypatch.setattr(jit_core, "bd_dir", str(tmp_path))
    monkeypatch.setattr(jit_core, "get_user_jit_dir", lambda: str(tmp_path))
    dependency = tmp_path / "dependency.csv"
    dependency.write_text("solidx\n312\n")
    module_path = tmp_path / f"{module}.so"
    module_path.touch()
    locked = False

    def fake_lock(*, lockPath, MainFunc):
        nonlocal locked
        assert lockPath == str(tmp_path / f"lock_{module}")
        locked = True
        try:
            return MainFunc()
        finally:
            locked = False

    def fake_rm_module(name):
        assert locked
        assert name == module
        module_path.unlink(missing_ok=True)

    def fake_clear_build(name):
        assert locked
        assert name == module

    monkeypatch.setattr(jit_core, "mp_lock", fake_lock)
    monkeypatch.setattr(jit_core, "rm_module", fake_rm_module)
    monkeypatch.setattr(jit_core, "clear_build", fake_clear_build)
    monkeypatch.setattr(
        jit_core, "_get_jit_dependency_patterns", lambda name: [str(dependency)]
    )
    jit_core._jit_dependency_checked.pop(module, None)

    fingerprint = jit_core._prepare_jit_dependencies(module)
    stamp = tmp_path / f"{module}.dependencies.sha256"

    assert stamp.read_text() == fingerprint
    assert not module_path.exists()
    jit_core._jit_dependency_checked.pop(module, None)


def test_tuned_gemm_missing_opus_kernel_falls_back_to_torch(monkeypatch):
    from aiter import tuned_gemm

    marker = torch.empty(0, dtype=torch.bfloat16)
    monkeypatch.setattr(tuned_gemm, "_opus_tune", _missing_tuned_kernel)
    monkeypatch.setattr(tuned_gemm, "torch_gemm", lambda *args, **kwargs: marker)

    inp = torch.empty((2, 4), dtype=torch.bfloat16)
    weights = torch.empty((3, 4), dtype=torch.bfloat16)
    result = tuned_gemm.opus_gemm(
        inp,
        weights,
        solidx=312,
        otype=torch.bfloat16,
        config={"splitK": 0},
    )

    assert result is marker


def test_tuned_gemm_missing_opus_kernel_honors_fp32_output(monkeypatch):
    from aiter import tuned_gemm

    monkeypatch.setattr(tuned_gemm, "_opus_tune", _missing_tuned_kernel)

    inp = torch.ones((2, 4), dtype=torch.bfloat16)
    weights = torch.ones((3, 4), dtype=torch.bfloat16)
    bias = torch.ones(3, dtype=torch.float32)
    result = tuned_gemm.opus_gemm(
        inp,
        weights,
        solidx=312,
        bias=bias,
        otype=torch.float32,
        config={"splitK": 0},
    )

    assert result.dtype == torch.float32
    assert torch.equal(result, torch.full((2, 3), 5.0))


def test_shape_driven_opus_missing_kernel_uses_heuristic(monkeypatch):
    import aiter.ops.opus.gemm_op_a16w16 as opus

    monkeypatch.setattr(
        opus._opus_common,
        "lookup_tuned",
        lambda **kwargs: {"solidx": 312, "splitK": 0},
    )
    monkeypatch.setattr(opus, "opus_gemm_a16w16_tune", _missing_tuned_kernel)

    def fake_dispatch(_x, _w, output, *_args):
        output.fill_(7)

    monkeypatch.setattr(opus, "_opus_gemm_bf16_dispatch", fake_dispatch)

    inp = torch.empty((2, 4), dtype=torch.bfloat16)
    weights = torch.empty((3, 4), dtype=torch.bfloat16)
    result = opus.gemm_a16w16_opus(inp, weights)

    assert torch.all(result == 7)
