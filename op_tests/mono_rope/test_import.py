# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MONO entry points and kernel sources import with a visible GPU and fresh cache."""

import os
from pathlib import Path
import subprocess
import sys


_REPO = Path(__file__).resolve().parents[2]


def test_mono_rope_import_with_visible_gpu(tmp_path):
    env = os.environ.copy()
    for name in (
        "GPU_ARCHS",
        "AITER_GPU_ARCHS",
        "AITER_RUNTIME_GPU_ARCH",
        "AITER_AOT_IMPORT",
        "AITER_AOT_REPLAY",
        "AITER_TRITON_ONLY",
    ):
        env.pop(name, None)
    env["AITER_JIT_DIR"] = str(tmp_path / "jit")
    env["PYTHONPATH"] = str(_REPO) + os.pathsep + env.get("PYTHONPATH", "")
    code = '''
import sys
from pathlib import Path
import torch
assert torch.cuda.is_available(), "This MONO import test requires a visible GPU"
import aiter
assert Path(aiter.__file__).resolve() == Path("aiter/__init__.py").resolve()
assert not aiter.AITER_TRITON_ONLY and not aiter.AITER_AOT_IMPORT
assert callable(aiter.rmsnorm2d_fwd_with_add)
assert callable(aiter.inverse_rope_group_quant)
from aiter.jit import core
import os
native_core = core.get_module("module_aiter_core")
assert Path(native_core.__file__).parent == Path(os.environ["AITER_JIT_DIR"])
assert aiter.ActivationType is native_core.ActivationType
native_inverse_rope = aiter.inverse_rope_group_quant
import aiter.ops.mono_standard as mono
assert callable(mono.rope_quant_window)
assert callable(mono.inverse_rope_group_quant)
assert aiter.inverse_rope_group_quant is native_inverse_rope
assert mono.inverse_rope_group_quant is not native_inverse_rope
assert "aiter.ops.mono_standard.loader" not in sys.modules
from aiter.ops.mono_standard import atom_helpers
import triton.language as tl
assert atom_helpers.FP8_DTYPE is torch.float8_e4m3fn
assert atom_helpers.FP8_MAX.value == 448.0
assert atom_helpers.FP8_TL_DTYPE.value == tl.float8e4nv
from aiter.ops.mono_standard.sources import rope_window_mono
assert callable(rope_window_mono.rope_quant_window)
from aiter.ops.mono_standard.loader import load_a05, checked_source
assert load_a05().__file__.endswith("/rope_window_mono.py")
assert checked_source("inverse_rope_group_quant_mono.cu").is_file()
assert not any(name.startswith("atom.") for name in sys.modules)
'''
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
