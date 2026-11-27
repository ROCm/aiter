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
import threading
import types
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
    compiled_kids._reset_cache()
    yield
    compiled_kids._reset_cache()


@pytest.fixture(autouse=True)
def _physical_device(monkeypatch):
    """The device the C++ dispatcher keys on: a full gfx950 (256 CUs)."""
    _stub_device(monkeypatch, "gfx950", GFX950_CU)


@pytest.fixture
def warnings(monkeypatch):
    messages = []
    monkeypatch.setattr(compiled_kids.logger, "warning", messages.append)
    return messages


def _stub_device(monkeypatch, gfx, cu, live_gfx=None):
    """One physical device. ``live_gfx`` is what rocminfo says (strict or not)."""
    monkeypatch.setattr(compiled_kids, "_device_arch_and_cu", lambda _d: (gfx, cu))
    monkeypatch.setattr(compiled_kids, "get_gfx_runtime", lambda: live_gfx or gfx)


def _stub_devices(monkeypatch, by_index):
    """Several devices: ``{index: (gfx, cu)}``; the current device is cuda:0."""

    def lookup(device):
        device = torch.device(device)
        return by_index[0 if device.index is None else device.index]

    monkeypatch.setattr(compiled_kids, "_device_arch_and_cu", lookup)
    monkeypatch.setattr(compiled_kids, "get_gfx_runtime", lambda: "gfx950")


def _stub_compiled(monkeypatch, keys):
    value = None if keys is None else frozenset(keys)
    monkeypatch.setattr(compiled_kids, "compiled_opus_kids", lambda: value)


# ---- the decision ---------------------------------------------------------


def test_compiled_kid_is_accepted_silently(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(TUNED_KID)])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)
    assert warnings == []


def test_uncompiled_kid_is_rejected_with_one_warning_naming_the_kid(
    monkeypatch, warnings
):
    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    for _ in range(3):
        assert not compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)
    assert len(warnings) == 1
    assert f"OPUS kid {TUNED_KID} " in warnings[0]
    assert "AITER_REBUILD=1" in warnings[0] and "opus_gemm_tune.py" in warnings[0]
    assert "opus_gemm_a8w8_tune.py" not in warnings[0]
    # A different missing kid is a different one-time warning.
    assert not compiled_kids.tuned_kid_is_compiled("a16w16", 7)
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
    _stub_device(monkeypatch, gfx, cu)
    assert not compiled_kids.tuned_kid_is_compiled(family, TUNED_KID)


def test_strict_gfx1250_device_uses_the_strict_namespace(monkeypatch, warnings):
    # torch reports a strict (MI450 A0) device as plain gfx1250; rocminfo knows.
    _stub_device(monkeypatch, "gfx1250", GFX950_CU, live_gfx="gfx1250-strict")
    # A kid listed for regular gfx1250 only has no device code for a strict device.
    _stub_compiled(monkeypatch, [_key(20007, gfx="gfx1250")])
    assert not compiled_kids.tuned_kid_is_compiled("a16w16", 20007)
    assert len(warnings) == 1 and "gfx1250-strict" in warnings[0]
    _stub_compiled(monkeypatch, [_key(20007, gfx="gfx1250-strict")])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", 20007)


def test_regular_gfx1250_device_uses_the_regular_namespace(monkeypatch, warnings):
    _stub_device(monkeypatch, "gfx1250", GFX950_CU, live_gfx="gfx1250")
    _stub_compiled(monkeypatch, [_key(20007, gfx="gfx1250-strict")])
    assert not compiled_kids.tuned_kid_is_compiled("a16w16", 20007)
    _stub_compiled(monkeypatch, [_key(20007, gfx="gfx1250")])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", 20007)


def test_decision_follows_the_tensor_device_not_the_current_device(
    monkeypatch, warnings
):
    """cuda:0 (current) has 256 CUs, the tensor lives on cuda:1 with 128."""
    _stub_devices(monkeypatch, {0: ("gfx950", 256), 1: ("gfx950", 128)})
    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=256)])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)  # current
    assert compiled_kids.tuned_kid_is_compiled(
        "a16w16", TUNED_KID, torch.device("cuda", 0)
    )
    assert not compiled_kids.tuned_kid_is_compiled(
        "a16w16", TUNED_KID, torch.device("cuda", 1)
    )
    assert len(warnings) == 1 and "128 CUs" in warnings[0]


def test_decision_uses_the_physical_cu_count_not_the_cu_num_override(
    monkeypatch, warnings
):
    """``CU_NUM=128`` on a 256-CU device: C++ still looks up the 256-CU table."""
    monkeypatch.setenv("CU_NUM", "128")
    # Compiled for the 128-CU table only: the 256-CU launcher cannot run it.
    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=128)])
    assert not compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)
    assert len(warnings) == 1 and "256 CUs" in warnings[0]
    # Compiled for the physical 256-CU table: runs, whatever CU_NUM says.
    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=256)])
    assert compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)


def test_unknown_manifest_keeps_previous_behaviour(monkeypatch, warnings):
    _stub_compiled(monkeypatch, None)

    def no_device(_device):
        raise AssertionError("an unknown manifest must not probe the device")

    monkeypatch.setattr(compiled_kids, "_device_arch_and_cu", no_device)
    assert compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID)
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


# ---- the once-only warnings are thread-safe -------------------------------


def _run_threads(target, count=8):
    barrier = threading.Barrier(count)

    def worker():
        barrier.wait()
        target()

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_uncompiled_kid_warns_once_across_threads(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(HEURISTIC_KID)])
    results = []
    _run_threads(
        lambda: results.append(compiled_kids.tuned_kid_is_compiled("a16w16", TUNED_KID))
    )
    assert results == [False] * 8
    assert len(warnings) == 1


def test_legacy_module_warns_once_across_threads(monkeypatch, warnings):
    calls = []

    def legacy():
        calls.append(1)
        threading.Event().wait(0.05)  # let the other threads pile up
        raise AttributeError("module has no attribute 'opus_compiled_kids'")

    monkeypatch.setattr(compiled_kids, "opus_compiled_kids", legacy)
    _run_threads(compiled_kids.compiled_opus_kids)
    assert len(calls) == 1
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


def _tuned_configs(monkeypatch, row_cu=GFX950_CU, devices=(None,)):
    """``get_GEMM_A16W16_config`` once per device, through one shared cache."""
    try:
        import aiter.tuned_gemm as tuned
    except ImportError as exc:  # e.g. no ROCm runtime on this host
        pytest.skip(f"aiter.tuned_gemm is not importable here: {exc}")
    M, N, K = SHAPE
    bf16 = str(torch.bfloat16)
    key = ("gfx950", row_cu, M, N, K, False, bf16, bf16, False, False)
    row = {
        "libtype": "opus",
        "solidx": TUNED_KID,
        "splitK": TUNED_SPLIT_K,
        "kernelName": "",
    }
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: {key: row})
    monkeypatch.setattr(tuned, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(tuned, "get_cu_num", lambda: row_cu)
    monkeypatch.setattr(tuned, "get_padded_m", lambda m, _n, _k, _gl: m)
    monkeypatch.setattr(tuned, "_opus_launch", object())
    tuned.get_GEMM_A16W16_config.cache_clear()
    try:
        return [
            tuned.get_GEMM_A16W16_config(M, N, K, False, bf16, bf16, device=device)
            for device in devices
        ]
    finally:
        tuned.get_GEMM_A16W16_config.cache_clear()


def _tuned_config(monkeypatch, row_cu=GFX950_CU):
    return _tuned_configs(monkeypatch, row_cu)[0]


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


def test_global_tuned_gemm_checks_the_physical_cu_under_a_cu_num_override(
    monkeypatch, warnings
):
    # The tuned-row lookup runs as a 128-CU part (CU_NUM=128) on 256 physical CUs.
    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=128)])
    config = _tuned_config(monkeypatch, row_cu=128)
    assert (config["libtype"], config["solidx"]) == ("torch", 0)
    assert len(warnings) == 1 and "256 CUs" in warnings[0]

    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=256)])
    config = _tuned_config(monkeypatch, row_cu=128)
    assert (config["libtype"], config["solidx"]) == ("opus", TUNED_KID)


def test_global_tuned_gemm_gates_on_the_tensor_device_and_caches_per_device(
    monkeypatch, warnings
):
    # Same (gfx, cu_num) arguments for both calls; only the device differs.
    _stub_devices(monkeypatch, {0: ("gfx950", 256), 1: ("gfx950", 128)})
    _stub_compiled(monkeypatch, [_key(TUNED_KID, cu=256)])
    cuda0, cuda1 = torch.device("cuda", 0), torch.device("cuda", 1)
    configs = _tuned_configs(monkeypatch, devices=(cuda0, cuda1, cuda0, cuda1))
    assert [(c["libtype"], c["solidx"]) for c in configs] == [
        ("opus", TUNED_KID),
        ("torch", 0),
        ("opus", TUNED_KID),
        ("torch", 0),
    ]
    assert len(warnings) == 1 and "128 CUs" in warnings[0]


def test_gemm_a16w16_passes_the_input_device_to_the_config_lookup(monkeypatch):
    try:
        import aiter.tuned_gemm as tuned
    except ImportError as exc:  # e.g. no ROCm runtime on this host
        pytest.skip(f"aiter.tuned_gemm is not importable here: {exc}")
    from aiter.ops.opus import _arch

    seen = []

    class Stop(Exception):
        pass

    def lookup(*_args, **kwargs):
        seen.append(kwargs)
        raise Stop

    monkeypatch.setattr(_arch, "_device_arch_and_cu", lambda _d: ("gfx950", 128))
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config", lookup)
    M, N, K = SHAPE
    A = torch.empty((M, K), device="meta", dtype=torch.bfloat16)
    B = torch.empty((N, K), device="meta", dtype=torch.bfloat16)
    with pytest.raises(Stop):
        tuned._gemm_a16w16(A, B)
    assert seen[0]["device"] == A.device
    assert (seen[0]["gfx"], seen[0]["cu_num"]) == ("gfx950", 128)


# ---- a8w8 blockscale bpreshuffle (the other subset-built family) ----------

BP_FAMILY = "a8w8_blockscale_bpreshuffle"
BP_KID = 3


def test_bpreshuffle_rows_use_the_same_gate(monkeypatch, warnings):
    _stub_compiled(monkeypatch, [_key(BP_KID, family=BP_FAMILY)])
    assert compiled_kids.tuned_kid_is_compiled(BP_FAMILY, BP_KID)
    assert not compiled_kids.tuned_kid_is_compiled(BP_FAMILY, BP_KID + 1)
    assert len(warnings) == 1 and BP_FAMILY in warnings[0]
    assert (
        "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
        in warnings[0]
    )
    assert "opus_gemm_a8w8_tune.py" not in warnings[0]


def _fp8_meta(*shape):
    return torch.empty(shape, device="meta", dtype=torch.float8_e4m3fn)


def _gemm_a8w8():
    import aiter.ops.gemm_op_a8w8 as gemm_a8w8

    return gemm_a8w8


def _run_mxscale_site(monkeypatch, compiled):
    """``_gemm_mxscale_bpreshuffle`` with a tuned OPUS row; returns the backend."""
    import aiter.ops.opus as opus_pkg
    from csrc.opus_gemm import opus_gemm_common

    gemm_a8w8 = _gemm_a8w8()
    _stub_compiled(monkeypatch, [_key(BP_KID, family=BP_FAMILY)] if compiled else [])
    calls = []
    monkeypatch.setattr(gemm_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_a8w8, "get_cu_num", lambda: GFX950_CU)
    monkeypatch.setattr(
        gemm_a8w8,
        "get_mxscale_bpreshuffle_config",
        lambda *_a: {"libtype": "opus", "kernelId": BP_KID, "kernelName": ""},
    )
    monkeypatch.setattr(
        opus_gemm_common, "a8w8_mxscale_bpreshuffle_supports_shape", lambda *_a: True
    )
    monkeypatch.setattr(
        opus_pkg,
        "opus_gemm",
        lambda *_a, **kw: calls.append(("opus", kw["kid"])),
    )
    bmm = types.ModuleType("aiter.ops.flydsl.batched_gemm_a8w8")
    bmm.run_bmm_a8w8_mxfp8 = lambda *_a, **_kw: calls.append(("bmm", None))
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.batched_gemm_a8w8", bmm)
    m, n, k = 128, 256, 512
    Y = torch.empty((m, n), device="meta", dtype=torch.bfloat16)
    gemm_a8w8._gemm_mxscale_bpreshuffle(
        _fp8_meta(m, k),
        _fp8_meta(n, k),
        torch.empty((m, k // 128), device="meta", dtype=torch.uint8),
        torch.empty((n // 128, k // 128), device="meta", dtype=torch.uint8),
        Y,
    )
    return calls


def test_mxscale_bpreshuffle_site_launches_a_compiled_opus_row(monkeypatch, warnings):
    assert _run_mxscale_site(monkeypatch, compiled=True) == [("opus", BP_KID)]
    assert warnings == []


def test_mxscale_bpreshuffle_site_falls_back_to_bmm_for_an_uncompiled_row(
    monkeypatch, warnings
):
    assert _run_mxscale_site(monkeypatch, compiled=False) == [("bmm", None)]
    assert len(warnings) == 1 and f"OPUS kid {BP_KID} " in warnings[0]


def _run_blockscale_site(monkeypatch, compiled):
    """The fp32-scale blockscale bpreshuffle path with a tuned OPUS row."""
    import aiter.ops.opus as opus_pkg
    from aiter.ops.opus import _arch

    gemm_a8w8 = _gemm_a8w8()
    _stub_compiled(monkeypatch, [_key(BP_KID, family=BP_FAMILY)] if compiled else [])
    calls = []
    monkeypatch.setattr(gemm_a8w8, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(_arch, "_device_arch_and_cu", lambda _d: ("gfx950", GFX950_CU))
    monkeypatch.setattr(
        gemm_a8w8,
        "get_CKGEMM_config",
        lambda *_a, **_kw: {"libtype": "opus", "kernelId": BP_KID, "kernelName": ""},
    )
    monkeypatch.setattr(
        opus_pkg, "opus_gemm", lambda *_a, **kw: calls.append(("opus", kw["kid"]))
    )
    monkeypatch.setattr(
        gemm_a8w8,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        lambda *_a, **_kw: calls.append(("ck", None)),
    )
    m, n, k = 128, 256, 512
    gemm_a8w8._gemm_a8w8_blockscale_bpreshuffle_native(
        _fp8_meta(m, k),
        _fp8_meta(n, k),
        torch.empty((m, k // 128), device="meta", dtype=torch.float32),
        torch.empty((n // 128, k // 128), device="meta", dtype=torch.float32),
        torch.bfloat16,
        None,
    )
    return calls


def test_blockscale_bpreshuffle_site_launches_a_compiled_opus_row(
    monkeypatch, warnings
):
    assert _run_blockscale_site(monkeypatch, compiled=True) == [("opus", BP_KID)]
    assert warnings == []


def test_blockscale_bpreshuffle_site_falls_back_to_ck_for_an_uncompiled_row(
    monkeypatch, warnings
):
    assert _run_blockscale_site(monkeypatch, compiled=False) == [("ck", None)]
    assert len(warnings) == 1 and f"OPUS kid {BP_KID} " in warnings[0]


# ---- codegen: the embedded manifest is the compiled set -------------------


def _run_codegen(targets, gpu_archs, extra_keys=()):
    """Run gen_instances.py (torch stubbed); return (embedded kids, sidecar kids)."""
    generator = REPO / "csrc/opus_gemm/gen_instances.py"
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
        cmd = [sys.executable, "-c", runner, str(generator), "--working_path", tmp]
        cmd += ["--targets", *targets]
        if extra_keys:
            cmd += ["--extra_kernel_keys", *extra_keys]
        result = subprocess.run(
            cmd,
            env={**os.environ, "GPU_ARCHS": gpu_archs},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        header = Path(tmp, "opus_compiled_kids.h").read_text()
        match = re.search(r'#define OPUS_COMPILED_KIDS "([^"]*)"', header)
        assert match, header
        embedded = set(match.group(1).split(";"))
        return embedded, set(json.loads(Path(sidecar).read_text()))


def test_codegen_embeds_the_selected_kids_in_the_module():
    registry = runpy.run_path(str(REPO / "csrc/opus_gemm/opus_gemm_common.py"))
    requested = registry["KernelKey"]("a16w16", "gfx950", GFX950_CU, TUNED_KID)
    embedded, sidecar = _run_codegen(
        [f"gfx950/{GFX950_CU}"], "gfx950", [requested.encode()]
    )
    assert embedded == sidecar
    assert requested.encode() in embedded
    # Every embedded kid is routable by Python's own registry.
    assert all(key.count("/") == 3 for key in embedded)


@pytest.mark.parametrize(
    ("gpu_archs", "targets", "regular_has_device_code"),
    (
        ("gfx1250", ["gfx1250/256"], True),
        ("gfx1250-strict", ["gfx1250-strict/256"], False),
        # Mixed offload list: both device passes exist.
        ("gfx1250;gfx1250-strict", ["gfx1250/256"], True),
        ("gfx1250;gfx1250-strict", ["gfx1250/256", "gfx1250-strict/256"], True),
        # The registry --targets must not override what the compiler is given.
        ("gfx1250-strict", ["gfx1250/256"], False),
    ),
)
def test_codegen_manifest_lists_gfx1250_kids_per_namespace(
    gpu_archs, targets, regular_has_device_code
):
    """The gfx1250 device TUs sit behind ``__gfx1250__``: only a plain gfx1250
    offload target gives non-CO kids device code, and a strict device (MI450 A0)
    never runs them. CO kids (prebuilt images) are launchable on both."""
    embedded, sidecar = _run_codegen(targets, gpu_archs)
    common = runpy.run_path(str(REPO / "csrc/opus_gemm/opus_gemm_common.py"))
    co_kids = {str(kid) for kid in common["gfx1250_4wave_co_kernels_list"]}

    def namespace(keys, gfx):
        return {k for k in keys if k.startswith(f"a16w16/{gfx}/")}

    def kid(key):
        return key.rsplit("/", 1)[1]

    routed = namespace(sidecar, "gfx1250")
    non_co = {k for k in routed if kid(k) not in co_kids}
    assert non_co, "this build routes no non-CO a16w16 gfx1250 kid"
    regular = namespace(embedded, "gfx1250")
    strict = namespace(embedded, "gfx1250-strict")
    # The strict namespace never lists a kid that has no device code for it.
    assert {kid(k) for k in strict} == {kid(k) for k in routed} - {
        kid(k) for k in non_co
    }
    assert regular == (routed if regular_has_device_code else routed - non_co)
    # Everything else is the sidecar, untouched.
    other = {k for k in embedded if "/gfx1250" not in k}
    assert other == {k for k in sidecar if "/gfx1250/" not in k}


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
