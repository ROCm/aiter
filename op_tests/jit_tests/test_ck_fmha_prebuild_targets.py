# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""No-GPU tests for CK fmha build-target selection.

These exercise the real build path -- aiter/jit/optCompilerConfig.json's
eval()'d blob_gen_cmd via core.get_args_of_build -- rather than mocking the
mapping helpers, so they cover the integration this PR actually changes.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

_BUILD_TARGETS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "aiter",
    "jit",
    "utils",
    "build_targets.py",
)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_EXTRACT_TARGETS = f"""
import os, re, sys
sys.path.insert(0, {os.path.join(_REPO_ROOT, "aiter", "jit")!r})
import core
cmds = core.get_args_of_build(os.environ["AITER_TEST_MODULE"])["blob_gen_cmd"]
cmd = cmds[0] if isinstance(cmds, list) else cmds
m = re.search(r"--targets (\\S+)", cmd)
print(m.group(1) if m else "NONE")
"""
_EXTRACT_SETUP_CK_EXCLUDES = f"""
import builtins, importlib.util, json, os, sys

repo = {_REPO_ROOT!r}
sys.path.insert(0, os.path.join(repo, "aiter", "jit", "utils"))
sys.path.insert(0, os.path.join(repo, "aiter"))
sys.path.insert(0, repo)

import setuptools

orig_setup = setuptools.setup
setuptools.setup = lambda *args, **kwargs: None
orig_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch":
        raise ImportError("blocked torch import for setup.py test")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
try:
    spec = importlib.util.spec_from_file_location(
        "_setup_under_test", os.path.join(repo, "setup.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    builtins.__import__ = orig_import
    setuptools.setup = orig_setup

from jit.utils.build_targets import CK_FMHA_MODULES, ck_fmha_enabled

module.CK_FMHA_MODULES = CK_FMHA_MODULES
module.ck_fmha_enabled = ck_fmha_enabled
exclude = module.get_exclude_ops()
print(json.dumps(sorted(name for name in exclude if name in CK_FMHA_MODULES)))
"""


def _load_build_targets():
    """Import build_targets.py directly; it is intentionally torch-free."""
    spec = importlib.util.spec_from_file_location("_bt_under_test", _BUILD_TARGETS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "archs, expected",
    [
        (["gfx942"], ["gfx9"]),
        (["gfx942", "gfx950"], ["gfx9", "gfx950"]),
        # gfx1250 is dropped: aiter uses the ASM/FlyDSL/Triton path there.
        (["gfx942", "gfx950", "gfx1250"], ["gfx9", "gfx950"]),
        (["gfx1250"], []),
        # RDNA rows must survive the gfx1250 exclusion.
        (["gfx1100"], ["gfx11"]),
        (["gfx1151"], ["gfx115"]),
        (["gfx1201"], ["gfx12"]),
    ],
)
def test_map_gpu_archs_drops_gfx1250(archs, expected):
    bt = _load_build_targets()
    assert bt.map_gpu_archs_to_ck_fmha_targets(archs) == expected


@pytest.mark.parametrize(
    "archs, expected",
    [
        (["gfx942", "gfx950"], True),
        (["gfx942", "gfx950", "gfx1250"], True),
        (["gfx1250"], False),
        # Unknown/CPU lists keep CK's default behaviour, they are not an
        # explicit opt-out.
        (["cpu"], True),
        ([], True),
    ],
)
def test_ck_fmha_enabled(archs, expected):
    bt = _load_build_targets()
    assert bt.ck_fmha_enabled_for(archs) is expected


def test_ck_fmha_modules_listed():
    """The exclusion list must name every module that runs 01_fmha/generate.py."""
    bt = _load_build_targets()
    assert set(bt.CK_FMHA_MODULES) == {
        "module_mha_fwd",
        "module_mha_varlen_fwd",
        "module_mha_batch_prefill",
        "module_mha_bwd",
        "module_mha_varlen_bwd",
        "libmha_fwd",
        "libmha_bwd",
    }


def test_ck_fmha_modules_match_config():
    """Guard against optCompilerConfig.json growing a new fmha module."""
    import json

    bt = _load_build_targets()
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "aiter",
        "jit",
        "optCompilerConfig.json",
    )
    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    from_config = {
        name for name, cfg in config.items() if "01_fmha/generate.py" in json.dumps(cfg)
    }
    assert from_config == set(bt.CK_FMHA_MODULES)


def test_setup_skips_varlen_injection_on_gfx1250_only_builds():
    bt = _load_build_targets()
    assert bt.ck_fmha_enabled_for(["gfx1250"]) is False

    setup_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "setup.py",
    )
    with open(setup_path, "r", encoding="utf-8") as fh:
        setup_source = fh.read()

    assert (
        "if PREBUILD_KERNELS == 1 and ENABLE_CK and ck_fmha_enabled():" in setup_source
    )


def _run_subprocess(script, gpu_archs, **extra_env):
    env = dict(os.environ, GPU_ARCHS=gpu_archs, **extra_env)
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, f"GPU_ARCHS={gpu_archs!r} failed:\n{out.stderr}"
    return out.stdout.strip().splitlines()[-1]


def _targets_for(module_name, gpu_archs):
    """Real blob_gen_cmd --targets for a GPU_ARCHS value. No mocking."""
    return _run_subprocess(_EXTRACT_TARGETS, gpu_archs, AITER_TEST_MODULE=module_name)


def _ck_fmha_excludes_for(gpu_archs):
    return json.loads(
        _run_subprocess(
            _EXTRACT_SETUP_CK_EXCLUDES,
            gpu_archs,
            PREBUILD_KERNELS="1",
            AITER_USE_SYSTEM_TRITON="1",
        )
    )


@pytest.mark.parametrize(
    "gpu_archs, expected",
    [
        ("gfx942", "gfx9"),
        # The release default must stay byte-identical to CK's own default.
        ("gfx942;gfx950", "gfx9,gfx950"),
        # gfx1250 is filtered out; one GPU_ARCHS list, one wheel.
        ("gfx942;gfx950;gfx1250", "gfx9,gfx950"),
        ("gfx1100", "gfx11"),
        ("gfx1201", "gfx12"),
    ],
)
def test_prebuild_blob_gen_cmd_targets(gpu_archs, expected):
    assert _targets_for("module_mha_fwd", gpu_archs) == expected


@pytest.mark.parametrize(
    "gpu_archs, expected",
    [
        ("gfx942;gfx950", "gfx9,gfx950"),
        ("gfx942;gfx1100", "gfx9"),
        ("gfx1100", "gfx11"),
    ],
)
def test_prebuild_blob_gen_cmd_targets_for_batch_prefill(gpu_archs, expected):
    assert _targets_for("module_mha_batch_prefill", gpu_archs) == expected


def test_setup_get_exclude_ops_excludes_ck_fmha_modules_on_gfx1250_only_builds():
    bt = _load_build_targets()
    expected = sorted(bt.CK_FMHA_MODULES)
    assert _ck_fmha_excludes_for("gfx1250") == expected


def test_setup_get_exclude_ops_keeps_ck_fmha_modules_on_supported_arches():
    assert _ck_fmha_excludes_for("gfx942;gfx950") == []
