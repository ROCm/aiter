# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch


def test_jit_dependency_fingerprint_tracks_contents_and_glob_members(tmp_path):
    from aiter.jit.core import _jit_dependency_fingerprint

    first = tmp_path / "first.csv"
    first.write_text("solidx\n315\n")
    pattern = str(tmp_path / "*.csv")

    initial = _jit_dependency_fingerprint([pattern])
    first.write_text("solidx\n312\n")
    assert _jit_dependency_fingerprint([pattern]) != initial

    after_edit = _jit_dependency_fingerprint([pattern])
    (tmp_path / "second.csv").write_text("solidx\n315\n")
    assert _jit_dependency_fingerprint([pattern]) != after_edit


def test_jit_dependency_stamp_is_written_once_per_process(tmp_path, monkeypatch):
    import aiter.jit.core as jit_core

    module = "test_dependency_module"
    monkeypatch.setattr(jit_core, "bd_dir", str(tmp_path))
    jit_core._jit_dependency_recorded.discard(module)

    jit_core._record_jit_dependencies(module, "first")
    stamp = tmp_path / f"{module}.dependencies.sha256"
    stamp.write_text("sentinel")
    jit_core._record_jit_dependencies(module, "second")

    assert stamp.read_text() == "sentinel"
    jit_core._jit_dependency_recorded.discard(module)


def test_tuned_gemm_missing_opus_kernel_falls_back_to_torch(monkeypatch):
    import aiter.tuned_gemm as tuned_gemm

    marker = object()
    monkeypatch.setattr(tuned_gemm, "_opus_kernel_compiled", lambda *_: False)
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


def test_shape_driven_opus_missing_kernel_uses_heuristic(monkeypatch):
    import aiter.ops.opus.gemm_op_a16w16 as opus

    monkeypatch.setattr(
        opus._opus_common,
        "lookup_tuned",
        lambda **kwargs: {"solidx": 312, "splitK": 0},
    )
    monkeypatch.setattr(opus, "is_a16w16_kernel_compiled", lambda *_: False)

    def fake_dispatch(_x, _w, output, *_args):
        output.fill_(7)

    monkeypatch.setattr(opus, "_opus_gemm_bf16_dispatch", fake_dispatch)

    inp = torch.empty((2, 4), dtype=torch.bfloat16)
    weights = torch.empty((3, 4), dtype=torch.bfloat16)
    result = opus.gemm_a16w16_opus(inp, weights)

    assert torch.all(result == 7)
