# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A tuned OPUS row naming a kid that the loaded module lacks must fall back.

``module_deepgemm_opus`` is a subset build, so a row can pass the Python
registry checks and still name a kid that is not compiled in; the C++ launcher
then raises ``kid N is not compiled in the current OPUS module``. The module
embeds its compiled-kid manifest (``opus_compiled_kids()``) and the tuned-row
call sites consult it. The CPU tests stub that set; the GPU test at the end
runs against the module that is really loaded.
"""

import json
import os
import re
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from aiter.ops.opus import compiled_kids

REPO = Path(__file__).resolve().parents[1]
GFX950_CU = 256
# gfx950 split-K kid used by the shape-driven tests of test_opus_a16w16_gemm.py;
# the heuristic floor picks kid 52 for the same (M, N, K).
TUNED_KID, TUNED_SPLIT_K, HEURISTIC_KID = 6, 2, 52
SHAPE = (128, 64, 512)


def _key(kid, family="a16w16", gfx="gfx950", cu=GFX950_CU):
    return f"{family}/{gfx}/{cu}/{kid}"


@pytest.fixture(autouse=True)
def _fresh_caches():
    compiled_kids.compiled_opus_kids.cache_clear()
    compiled_kids._warn_uncompiled_kid.cache_clear()
    yield
    compiled_kids.compiled_opus_kids.cache_clear()
    compiled_kids._warn_uncompiled_kid.cache_clear()


@pytest.fixture
def warnings(monkeypatch):
    messages = []
    monkeypatch.setattr(compiled_kids.logger, "warning", messages.append)
    return messages


def _stub_compiled(monkeypatch, keys):
    value = None if keys is None else frozenset(keys)
    monkeypatch.setattr(compiled_kids, "compiled_opus_kids", lambda: value)


# ---- the decision ---------------------------------------------------------


def test_compiled_kid_is_accepted_silently(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(TUNED_KID)])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", "gfx950", GFX950_CU, TUNED_KID)
    assert warnings == []


def test_uncompiled_kid_is_rejected_with_one_warning_naming_the_kid(
    monkeypatch, warnings
):
    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    for _ in range(3):
        assert not compiled_kids.tuned_kid_is_compiled(
            "a16w16", "gfx950", GFX950_CU, TUNED_KID
        )
    assert len(warnings) == 1
    assert f"OPUS kid {TUNED_KID} " in warnings[0]
    assert "AITER_REBUILD=1" in warnings[0] and "opus_gemm_tune.py" in warnings[0]
    # A different missing kid is a different one-time warning.
    assert not compiled_kids.tuned_kid_is_compiled("a16w16", "gfx950", GFX950_CU, 7)
    assert len(warnings) == 2


@pytest.mark.parametrize(
    ("family", "gfx", "cu"),
    (
        ("a8w8_blockscale_bpreshuffle", "gfx950", GFX950_CU),  # other family
        ("a16w16", "gfx942", GFX950_CU),  # other arch
        ("a16w16", "gfx950", 128),  # other CU count
    ),
)
def test_compiled_set_is_keyed_by_family_arch_and_cu(
    monkeypatch, warnings, family, gfx, cu
):
    _stub_compiled(monkeypatch, [_key(TUNED_KID)])
    assert not compiled_kids.tuned_kid_is_compiled(family, gfx, cu, TUNED_KID)


def test_strict_gfx1250_runtime_uses_the_gfx1250_key(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(20007, gfx="gfx1250")])
    assert compiled_kids.tuned_kid_is_compiled(
        "a16w16", "gfx1250-strict", GFX950_CU, 20007
    )


def test_unknown_manifest_keeps_previous_behaviour(monkeypatch, warnings):
    _stub_compiled(monkeypatch, None)
    assert compiled_kids.tuned_kid_is_compiled("a16w16", "gfx950", GFX950_CU, TUNED_KID)
    assert warnings == []


def test_manifest_is_parsed_from_the_module_export(monkeypatch):
    monkeypatch.setattr(
        compiled_kids, "opus_compiled_kids", lambda: f"{_key(1)};{_key(2)};"
    )
    assert compiled_kids.compiled_opus_kids() == {_key(1), _key(2)}


def test_module_without_the_export_is_unknown_not_an_error(monkeypatch, warnings):
    def legacy():
        raise AttributeError("module has no attribute 'opus_compiled_kids'")

    monkeypatch.setattr(compiled_kids, "opus_compiled_kids", legacy)
    assert compiled_kids.compiled_opus_kids() is None
    assert len(warnings) == 1 and "AITER_REBUILD=1" in warnings[0]


# ---- shape-driven gemm_a16w16_opus ---------------------------------------


def _run_shape_driven(monkeypatch, tuned_row):
    from aiter.ops import opus
    from aiter.ops.opus import gemm_op_a16w16, policy

    calls = []
    monkeypatch.setattr(
        gemm_op_a16w16, "_device_arch_and_cu", lambda _d: ("gfx950", GFX950_CU)
    )
    monkeypatch.setattr(policy, "lookup_a16w16_opus_config", lambda **_k: tuned_row)
    monkeypatch.setattr(
        gemm_op_a16w16,
        "_launch_a16w16_gemm",
        lambda XQ, WQ, Y, **kwargs: calls.append(kwargs) or Y,
    )
    M, N, K = SHAPE
    A = torch.empty((M, K), device="meta", dtype=torch.bfloat16)
    B = torch.empty((N, K), device="meta", dtype=torch.bfloat16)
    opus.gemm_a16w16_opus(A, B)
    return [(c["kid"], c["split_k"]) for c in calls]


def test_shape_driven_dispatch_uses_a_compiled_tuned_kid(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(TUNED_KID), _key(HEURISTIC_KID)])
    row = {"solidx": TUNED_KID, "splitK": TUNED_SPLIT_K}
    assert _run_shape_driven(monkeypatch, row) == [(TUNED_KID, TUNED_SPLIT_K)]
    assert warnings == []


def test_shape_driven_dispatch_falls_back_for_an_uncompiled_tuned_kid(
    monkeypatch, warnings
):
    # The heuristic floor is compiled by construction.
    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    row = {"solidx": TUNED_KID, "splitK": TUNED_SPLIT_K}
    assert _run_shape_driven(monkeypatch, row) == [(HEURISTIC_KID, 0)]
    assert _run_shape_driven(monkeypatch, row) == [(HEURISTIC_KID, 0)]
    assert len(warnings) == 1 and f"OPUS kid {TUNED_KID} " in warnings[0]


def test_explicit_kernel_id_is_not_filtered(monkeypatch, warnings):
    """A caller-chosen kid still reaches the launcher (and its own error)."""
    from aiter.ops import opus
    from aiter.ops.opus import gemm_op_a16w16, policy

    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    calls = []
    monkeypatch.setattr(
        gemm_op_a16w16, "_device_arch_and_cu", lambda _d: ("gfx950", GFX950_CU)
    )
    monkeypatch.setattr(policy, "lookup_a16w16_opus_config", lambda **_k: None)
    monkeypatch.setattr(
        gemm_op_a16w16,
        "_launch_a16w16_gemm",
        lambda XQ, WQ, Y, **kwargs: calls.append(kwargs) or Y,
    )
    M, N, K = SHAPE
    A = torch.empty((M, K), device="meta", dtype=torch.bfloat16)
    B = torch.empty((N, K), device="meta", dtype=torch.bfloat16)
    opus.gemm_a16w16_opus(A, B, kernelId=TUNED_KID, splitK=TUNED_SPLIT_K)
    assert [(c["kid"], c["split_k"]) for c in calls] == [(TUNED_KID, TUNED_SPLIT_K)]
    assert warnings == []


# ---- global tuned GEMM (aiter.tuned_gemm) --------------------------------


def _tuned_config(monkeypatch):
    try:
        import aiter.tuned_gemm as tuned
    except ImportError as exc:  # e.g. no ROCm runtime on this host
        pytest.skip(f"aiter.tuned_gemm is not importable here: {exc}")
    M, N, K = SHAPE
    bf16 = str(torch.bfloat16)
    key = ("gfx950", GFX950_CU, M, N, K, False, bf16, bf16, False, False)
    row = {
        "libtype": "opus",
        "solidx": TUNED_KID,
        "splitK": TUNED_SPLIT_K,
        "kernelName": "",
    }
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: {key: row})
    monkeypatch.setattr(tuned, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: GFX950_CU)
    monkeypatch.setattr(tuned, "get_padded_m", lambda m, _n, _k, _gl: m)
    monkeypatch.setattr(tuned, "_opus_launch", object())
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        return tuned.get_GEMM_A16W16_config(M, N, K, False, bf16, bf16)
    finally:
        tuned.get_GEMM_A16W16_config.cache_clear()


def test_global_tuned_gemm_keeps_a_compiled_opus_row(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(TUNED_KID)])
    config = _tuned_config(monkeypatch)
    assert (config["libtype"], config["solidx"]) == ("opus", TUNED_KID)
    assert warnings == []


def test_global_tuned_gemm_skips_an_uncompiled_opus_row(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    config = _tuned_config(monkeypatch)
    # No other row: the framework default takes over, as for a stale row.
    assert (config["libtype"], config["solidx"]) == ("torch", 0)
    assert len(warnings) == 1 and f"OPUS kid {TUNED_KID} " in warnings[0]


# ---- a8w8 blockscale bpreshuffle (the other subset-built family) ----------


def test_bpreshuffle_rows_use_the_same_gate(monkeypatch, warnings):
    family = "a8w8_blockscale_bpreshuffle"
    _stub_compiled(monkeypatch, [_key(3, family=family)])
    assert compiled_kids.tuned_kid_is_compiled(family, "gfx950", GFX950_CU, 3)
    assert not compiled_kids.tuned_kid_is_compiled(family, "gfx950", GFX950_CU, 4)
    assert len(warnings) == 1 and family in warnings[0]


# ---- codegen: the embedded manifest is the compiled set -------------------


def test_codegen_embeds_the_selected_kids_in_the_module():
    generator = REPO / "csrc/opus_gemm/gen_instances.py"
    registry = runpy.run_path(str(generator.with_name("opus_gemm_common.py")))
    requested = registry["KernelKey"]("a16w16", "gfx950", GFX950_CU, TUNED_KID)
    # Same stand-in for torch that test_jit_cache_transaction.py uses.
    runner = (
        "import os, runpy, sys, types; "
        "sys.argv = sys.argv[1:]; "
        "sys.path.insert(0, os.path.dirname(sys.argv[0])); "
        "sys.modules['torch'] = types.SimpleNamespace("
        "cuda=types.SimpleNamespace(is_available=lambda: False)); "
        "runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    with tempfile.TemporaryDirectory() as tmp:
        sidecar = os.path.join(tmp, "compiled_kids.json")
        Path(sidecar).write_text("[]")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                runner,
                str(generator),
                "--working_path",
                tmp,
                "--targets",
                f"gfx950/{GFX950_CU}",
                "--extra_kernel_keys",
                requested.encode(),
            ],
            env={**os.environ, "GPU_ARCHS": "gfx950"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        header = Path(tmp, "opus_compiled_kids.h").read_text()
        match = re.search(r'#define OPUS_COMPILED_KIDS "([^"]*)"', header)
        assert match, header
        embedded = set(match.group(1).split(";"))
        assert embedded == set(json.loads(Path(sidecar).read_text()))
        assert requested.encode() in embedded
        # Every embedded kid is routable by Python's own registry.
        assert all(key.count("/") == 3 for key in embedded)


# ---- GPU: the module that is really loaded --------------------------------


def _gpu_module_arch():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    from aiter.ops.opus._arch import _device_arch_and_cu

    arch, cu_num = _device_arch_and_cu(torch.device("cuda"))
    if arch not in ("gfx950", "gfx942"):
        pytest.skip(f"needs a gfx942/gfx950 OPUS module, got {arch}")
    return arch, cu_num


def test_gpu_loaded_module_reports_its_compiled_kids_and_falls_back():
    from aiter.ops import opus
    from aiter.ops.opus import policy
    from csrc.opus_gemm.opus_gemm_common import get_registry

    arch, cu_num = _gpu_module_arch()
    compiled = compiled_kids.compiled_opus_kids()
    if compiled is None:
        pytest.skip("loaded module_deepgemm_opus predates opus_compiled_kids()")
    registry = get_registry("a16w16", arch, cu_num)
    assert {
        _key(k, gfx=arch, cu=cu_num) for k in registry
    } - compiled, "this build compiles every registered kid; nothing to fall back from"
    assert any(key.startswith(f"a16w16/{arch}/{cu_num}/") for key in compiled)

    M, N, K = 256, 512, 1024
    from aiter.ops.opus import gemm_op_a16w16
    from aiter.ops.opus.launch_plan import _get_cached_a16w16_launch_plan

    def plannable(kid):
        try:
            return _get_cached_a16w16_launch_plan(
                arch, M, N, K, 1, cu_num, False, torch.bfloat16, torch.bfloat16, kid, 0
            ).resolved_kid
        except ValueError:
            return None

    # One kid this shape accepts that is compiled, one that is not.
    kept = next(
        k
        for k in sorted(registry)
        if _key(k, gfx=arch, cu=cu_num) in compiled and plannable(k) == k
    )
    victim = next(
        k
        for k in sorted(registry)
        if _key(k, gfx=arch, cu=cu_num) not in compiled and plannable(k) == k
    )

    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    ref = A.float() @ B.float().T

    # Without the gate the launcher raises, which is what killed the rank.
    with pytest.raises(RuntimeError, match="is not compiled in the current OPUS"):
        opus.gemm_a16w16_opus(A, B, kernelId=victim)

    launched = []
    real_launch = gemm_op_a16w16._launch_a16w16_backend

    def spy(XQ, WQ, Y, bias, workspace, kid, split_k):
        launched.append(kid)
        return real_launch(XQ, WQ, Y, bias, workspace, kid, split_k)

    original = policy.lookup_a16w16_opus_config
    gemm_op_a16w16._launch_a16w16_backend = spy
    try:
        for tuned_kid in (kept, victim):
            policy.lookup_a16w16_opus_config = lambda _kid=tuned_kid, **_k: {
                "solidx": _kid,
                "splitK": 0,
            }
            Y = opus.gemm_a16w16_opus(A, B)
            torch.cuda.synchronize()
            assert torch.allclose(Y.float(), ref, rtol=3e-2, atol=3e-1)
    finally:
        policy.lookup_a16w16_opus_config = original
        gemm_op_a16w16._launch_a16w16_backend = real_launch

    # A compiled tuned kid is launched as before; an uncompiled one is not.
    assert launched[0] == kept
    assert launched[1] != victim
