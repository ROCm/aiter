# SPDX-License-Identifier: MIT
"""Load the v0/v1 kernel module: direct file load by default.

``import aiter`` pulls in ``aiter/jit``, which builds ``module_aiter_core`` and
touches the GPU at import time (that build is what hung a08-1).  The QSA kernel
needs neither, so the file load is the default here; set
``QSA_IMPORT_PACKAGE=1`` to exercise the package path.

On a machine without ``flydsl`` (e.g. the Windows CPU box) ``allow_stub=True``
installs a minimal ``flydsl`` stub so the module's *host-side* code — the op
wrapper and its guard rails — can still be imported and unit-tested.  Nothing
that builds or launches a kernel can run through the stub; it exists only so the
CPU test suite keeps covering the wrapper on machines with no GPU stack.
"""

import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_PATH = os.path.abspath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "aiter",
        "ops",
        "flydsl",
        "kernels",
        "mqa_logits",
        "qsa_paged_mqa_logits_page4_gfx1250.py",
    )
)


class _StubScalar:
    """Stand-in for ``fx.Int32`` / ``fx.Float32`` / ... (hashable *and* callable)."""

    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return _StubScalar(self.name)

    def __repr__(self):
        return f"<stub {self.name}>"


class _StubVector(_StubScalar):
    @staticmethod
    def from_elements(*args, **kwargs):
        return _StubVector("Vector")


def _make_stub():
    expr = types.ModuleType("flydsl.expr")
    for name in (
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint32",
        "Float16",
        "Float32",
        "BFloat16",
        "Boolean",
        "Pointer",
    ):
        setattr(expr, name, _StubScalar(name))
    expr.Vector = _StubVector("Vector")
    expr.range_constexpr = range
    expr.Tensor = _StubScalar("Tensor")

    rocdl = types.ModuleType("flydsl.expr.rocdl")
    for name in (
        "make_buffer_ptr",
        "make_buffer_tensor",
        "get_buffer_rsrc",
        "BufferCopy",
        "BufferCopy32b",
        "BufferCopy64b",
        "BufferCopy128b",
    ):
        setattr(rocdl, name, _StubScalar(name))
    expr.rocdl = rocdl

    compiler = types.ModuleType("flydsl.compiler")
    for name in ("kernel", "jit", "compile"):
        setattr(compiler, name, lambda *a, **k: (lambda f: f))
    compiler.protocol = types.ModuleType("flydsl.compiler.protocol")
    compiler.protocol.dsl_size_of = lambda *a, **k: 0

    flydsl = types.ModuleType("flydsl")
    flydsl.expr = expr
    flydsl.compiler = compiler
    flydsl.__version__ = "stub"
    return {
        "flydsl": flydsl,
        "flydsl.expr": expr,
        "flydsl.expr.rocdl": rocdl,
        "flydsl.compiler": compiler,
        "flydsl.compiler.protocol": compiler.protocol,
    }


def load_kernel_module(allow_stub=False):
    """Return ``(module, how)``; ``how`` says which import path was used."""
    if os.environ.get("QSA_IMPORT_PACKAGE") == "1":
        from aiter.ops.flydsl.kernels.mqa_logits import (
            qsa_paged_mqa_logits_page4_gfx1250 as K,
        )

        return K, "package:aiter.ops.flydsl.kernels.mqa_logits"

    def _load(suffix):
        spec = importlib.util.spec_from_file_location(
            "qsa_page4_v0" + suffix, KERNEL_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    try:
        return _load(""), f"file:{KERNEL_PATH}"
    except ModuleNotFoundError as exc:
        if not allow_stub or "flydsl" not in str(exc):
            raise
        installed = _make_stub()
        sys.modules.update(installed)
        try:
            mod = _load("_stub")
        finally:
            for name in installed:
                sys.modules.pop(name, None)
        return mod, (
            f"file:{KERNEL_PATH} + stub flydsl "
            "(host-side wrapper only; no kernel can be built)"
        )
