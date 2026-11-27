"""A kernel's IR must not hold the directory aiter is in: its cache identity
is derived from the IR, and a kernel prebuilt in a build tree must be found
once aiter is installed elsewhere."""

import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_asm_mark_names_its_caller_without_the_directory(monkeypatch):
    from aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8 import common

    emitted = []
    monkeypatch.setattr(common.fx.rocdl, "sched_barrier", lambda *_: None)
    monkeypatch.setattr(
        common.llvm, "inline_asm", lambda *args, **_: emitted.append(args[2])
    )
    monkeypatch.setattr(common.ir.Type, "parse", lambda *_: None)

    common.asm_mark("here")

    (asm,) = emitted
    assert asm.startswith("s_nop 1; here ")
    assert os.path.basename(__file__) in asm
    assert os.sep not in asm
    assert str(_REPO_ROOT) not in asm
