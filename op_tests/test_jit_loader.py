# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import shutil
import subprocess
import sys
import types

import pytest

from aiter.jit import core

pytestmark = pytest.mark.skipif(
    getattr(os, "RTLD_DEEPBIND", 0) == 0, reason="RTLD_DEEPBIND is a glibc extension"
)

# Stand-in for TileLang's libhip_stub.so, built with plain gcc so this does not
# depend on TileLang being installed. It reproduces the real defect: the
# exported symbol carries the R0600 name but resolves the unversioned one, so
# the legacy implementation writes a differently laid out struct into an R0600
# destination. Forwarding to the real runtime, rather than scribbling a fixed
# pattern, keeps the write inside the caller's buffer -- so an unprotected
# AITER fails the way it does in production, with hipErrorInvalidConfiguration,
# instead of by SIGSEGV.
_STUB_C = """
#include <dlfcn.h>
#include <stddef.h>
typedef int (*legacy_fn)(void *, int);
int hipGetDevicePropertiesR0600(void *prop, int device) {
    static legacy_fn legacy = NULL;
    if (legacy == NULL) {
        const char *names[] = {
            "libamdhip64.so", "libamdhip64.so.7",
            "libamdhip64.so.6", "libamdhip64.so.5"
        };
        void *h = NULL;
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
            h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
            if (h != NULL) break;
        }
        if (h == NULL) return 1;
        legacy = (legacy_fn)dlsym(h, "hipGetDeviceProperties");
        if (legacy == NULL) return 1;
    }
    return legacy(prop, device);
}
"""

_INTERPOSE_SCRIPT = """
import ctypes, math, sys, torch
ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)   # before AITER loads anything

import aiter
n, e, k = 16384, 128, 8
gate = torch.randn(n, e, dtype=torch.float32, device="cuda")
w = torch.empty(n, k, dtype=torch.float32, device="cuda")
i = torch.empty(n, k, dtype=torch.int32, device="cuda")
t = torch.empty(n, k, dtype=torch.int32, device="cuda")

aiter.topk_softmax(w, i, t, gate, False)            # pybind extension path
torch.cuda.synchronize()
want = torch.topk(torch.softmax(gate, -1), k, dim=-1)[0]
assert torch.allclose(w, want, atol=1e-5), "topk_softmax corrupted by interposed HIP"

from aiter.ops.moe_op import topk_softmax_asm       # standalone ctypes path
w_asm = torch.empty_like(w)
i_asm = torch.empty_like(i)
t_asm = torch.empty_like(t)
topk_softmax_asm(w_asm, i_asm, t_asm, gate, False)
torch.cuda.synchronize()
assert torch.allclose(w_asm, want, atol=1e-5), "ctypes topk corrupted by interposed HIP"

from aiter.fused_moe import moe_sorting             # sizes its grid from device props
moe_sorting(i, w, e, 1024, torch.bfloat16)
torch.cuda.synchronize()

from aiter.ops.mha import fmha_v3_varlen_fwd        # Kimi-K3 failure path
q = torch.randn((128, 8, 128), device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
fmha_v3_varlen_fwd(
    q, q, q, cu, cu, 128, 128, 0, 0.0, 1.0 / math.sqrt(128), 0.0,
    False, False, -1, -1, False, False, 1,
)
torch.cuda.synchronize()
print("OK")
"""


def test_deep_import_leaves_process_dlopen_flags_alone():
    """A process-wide flag would be inherited by unrelated concurrent dlopens."""
    handles = dict(core._deep_handles)
    before = sys.getdlopenflags()
    core._deep_import("json")  # pure Python: nothing to pre-open
    assert sys.getdlopenflags() == before
    assert core._deep_handles == handles


def test_deep_import_predlopens_extension(monkeypatch):
    origin = "/tmp/module_deepbind_test.so"
    handle = object()
    module = object()
    seen = {}

    monkeypatch.setattr(core, "_RTLD_DEEPBIND", os.RTLD_DEEPBIND)
    monkeypatch.setattr(
        core.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(origin=origin),
    )

    def fake_cdll(path, mode):
        seen["path"] = path
        seen["mode"] = mode
        return handle

    monkeypatch.setattr(core.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(core.importlib, "import_module", lambda name: module)
    core._deep_handles.pop(origin, None)
    try:
        assert core._deep_import("module_deepbind_test") is module
        assert seen == {"path": origin, "mode": os.RTLD_NOW | os.RTLD_DEEPBIND}
        assert core._deep_handles[origin] is handle
    finally:
        core._deep_handles.pop(origin, None)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", os.RTLD_LAZY | os.RTLD_DEEPBIND), ("1", os.RTLD_LAZY)],
)
def test_generated_python_loader_mode(monkeypatch, value, expected):
    """The standalone generated-library loader follows the same opt-out."""
    from csrc.cpp_itfs import utils

    symbol = object()
    seen = {}

    def fake_cdll(path, mode):
        seen.update(path=path, mode=mode)
        return types.SimpleNamespace(test_symbol=symbol)

    monkeypatch.setenv("AITER_DISABLE_DEEPBIND", value)
    monkeypatch.setattr(utils, "BUILD_DIR", "/tmp/aiter-loader-test")
    monkeypatch.setattr(utils.ctypes, "CDLL", fake_cdll)
    utils.run_lib.cache_clear()
    try:
        assert utils.run_lib("test_symbol", "test_folder") is symbol
        assert seen == {
            "path": "/tmp/aiter-loader-test/test_folder/lib.so",
            "mode": expected,
        }
    finally:
        utils.run_lib.cache_clear()


def test_core_loader_optout_is_applied_at_import():
    """The documented process-start opt-out must affect the actual mode flag."""
    script = "from aiter.jit import core; print(core._RTLD_DEEPBIND)"
    for value, expected in (("0", os.RTLD_DEEPBIND), ("1", 0)):
        env = os.environ.copy()
        env["AITER_DISABLE_DEEPBIND"] = value
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert int(proc.stdout.strip().splitlines()[-1]) == expected


def test_aiter_survives_a_global_hip_interposer(tmp_path):
    """AITER must stay correct with an interposer loaded first.

    Runs in a subprocess: loaded objects and their global ordering cannot be
    reset in-process, so the failing order has to be built from scratch.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        pytest.skip("no C compiler for the interposer")

    src = tmp_path / "interposer.c"
    src.write_text(_STUB_C)
    stub = tmp_path / "libinterposer.so"
    build = subprocess.run(
        [cc, "-shared", "-fPIC", "-o", str(stub), str(src), "-ldl"],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        pytest.skip(f"could not build interposer: {build.stderr}")

    script = tmp_path / "interpose.py"
    script.write_text(_INTERPOSE_SCRIPT)
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (repo_root, env.get("PYTHONPATH")) if path
    )
    proc = subprocess.run(
        [sys.executable, str(script), str(stub)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout

    # Prove that the interposer is effective, so the protected run above
    # cannot pass merely because the test setup failed to reproduce the bug.
    unprotected_env = env.copy()
    unprotected_env["AITER_DISABLE_DEEPBIND"] = "1"
    unprotected = subprocess.run(
        [sys.executable, str(script), str(stub)],
        env=unprotected_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unprotected.returncode != 0, "interposer did not reproduce the bug"
    failure = f"{unprotected.stdout}\n{unprotected.stderr}".lower()
    assert "invalid argument" in failure or "invalid configuration" in failure


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
