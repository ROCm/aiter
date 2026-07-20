# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single source of truth for MX-format scale rounding mode and dtype enums.

Two parallel views of the same C++ enums (``csrc/include/mx_quant_utils.h``)
are exported:

* :class:`MxScaleRoundModeInt` / :class:`MxDtypeInt` -- plain Python
  classes whose attributes are bare ``int`` constants. Importing them is
  free (no JIT, no extra deps), so they are safe to use from build-time
  code paths such as FlyDSL AOT during wheel ``PREBUILD_KERNELS`` --
  ``module_aiter_core`` has not been built yet at that point.
* :class:`MxScaleRoundMode` / :class:`MxDtype` -- the pybind11 enum
  classes exported from ``csrc/include/rocm_ops.hpp::AITER_CORE_PYBIND``.
  These are loaded **lazily** on first attribute access (``__getattr__``);
  that access triggers ``module_aiter_core`` JIT build/import, so they
  are **not** safe to access during the wheel pre-build flow.

Both views are guaranteed to share the same int values: the lazy loader
asserts ``int(MxScaleRoundMode.RoundUp) == MxScaleRoundModeInt.RoundUp``
etc. on first access. To add or change a mode/dtype, edit
``mx_quant_utils.h`` and the int mirror class below in lockstep.

Cross-stack naming aligns with PyTorch torchao ``ScaleCalculationMode``,
NV Triton, DSv4, FlashInfer, and AMD Quark ``RoundMode``.
"""

from ..jit.core import compile_ops


# --------------------------------------------------------------------------
# Plain-int mirror classes (JIT-free; safe at PREBUILD_KERNELS / AOT time).
# --------------------------------------------------------------------------
class MxScaleRoundModeInt:
    """Bare-int mirror of C++ ``MxScaleRoundMode`` (mx_quant_utils.h).

    Values must stay 1:1 with the C++ enum and the pybind ``MxScaleRoundMode``
    class; the lazy loader below verifies this on first import of the
    pybind enum and raises :class:`AssertionError` on drift.
    """

    RoundDown = 0
    RoundUp = 1
    Even = 2
    Ceil = 3


class MxDtypeInt:
    """Bare-int mirror of C++ ``MxDtype`` (mx_quant_utils.h)."""

    FP4_E2M1 = 0
    FP8_E4M3 = 1
    FP8_E4M3_FNUZ = 2


# --------------------------------------------------------------------------
# Project-wide default round mode (Python single source of truth).
# Must match ``kDefaultMxScaleRoundMode`` in ``mx_quant_utils.h``; the
# drift check in ``__getattr__`` below and in ``test_quant_mxfp4.py``
# verifies this at runtime.
# --------------------------------------------------------------------------
MX_DEFAULT_ROUND_MODE = MxScaleRoundModeInt.RoundUp


# --------------------------------------------------------------------------
# Lazy pybind11 enum loading.
# --------------------------------------------------------------------------
# ``_MxScaleRoundMode`` / ``_MxDtype`` are listed in
# ``aiter/jit/utils/torch_guard.py::NONE_WRAPPED_OP`` so the ``@compile_ops``
# decorator skips the ``torch.library.infer_schema`` step. ``@compile_ops``
# itself only registers the function for lazy JIT; it does **not** build
# ``module_aiter_core`` at decoration time. The build only fires when the
# wrapped function is *called*, which we defer to :func:`__getattr__`.
@compile_ops("module_aiter_core", "MxScaleRoundMode")
def _MxScaleRoundMode(dummy): ...


@compile_ops("module_aiter_core", "MxDtype")
def _MxDtype(dummy): ...


_PYBIND_FACTORIES = {
    "MxScaleRoundMode": (_MxScaleRoundMode, MxScaleRoundModeInt),
    "MxDtype": (_MxDtype, MxDtypeInt),
}


def __getattr__(name):
    """PEP 562 lazy loader for the pybind11 enum classes.

    The first access of ``MxScaleRoundMode`` / ``MxDtype`` builds (or
    imports, if already built) ``module_aiter_core`` and returns the
    pybind11 enum class. Subsequent accesses skip this hook -- the result
    is cached in module ``globals()``.

    Pulling the enum class off the binding gives every importer the *same*
    class object as the HIP kernels see, so the C++ ``enum class`` and
    the Python attribute are literally the same type
    (verifiable via ``is``-comparison).
    """
    pair = _PYBIND_FACTORIES.get(name)
    if pair is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # On first load of MxScaleRoundMode, also verify that the C++
    # kDefaultMxScaleRoundMode matches our Python MX_DEFAULT_ROUND_MODE.
    if name == "MxScaleRoundMode":
        from ..jit.core import get_module

        cpp_default = getattr(
            get_module("module_aiter_core"), "kDefaultMxScaleRoundMode", None
        )
        if cpp_default is not None and cpp_default != MX_DEFAULT_ROUND_MODE:
            raise AssertionError(
                f"MX_DEFAULT_ROUND_MODE={MX_DEFAULT_ROUND_MODE} but C++ "
                f"kDefaultMxScaleRoundMode={cpp_default} "
                f"(drift; update both in lockstep)"
            )
    factory, int_mirror = pair
    # Use the first known int value from the mirror class (not a hard-coded 0)
    # so this survives future enum reorderings where 0 may not be valid.
    first_val = next(
        v
        for k, v in vars(int_mirror).items()
        if not k.startswith("_") and isinstance(v, int)
    )
    cls = type(factory(first_val))
    # Guard against C++/Python enum drift: every named member of the int
    # mirror must round-trip through ``int(cls.<NAME>)``.
    for attr_name, expected in vars(int_mirror).items():
        if attr_name.startswith("_") or not isinstance(expected, int):
            continue
        actual = int(getattr(cls, attr_name))
        if actual != expected:
            raise AssertionError(
                f"{name}.{attr_name} = {actual} but "
                f"{int_mirror.__name__}.{attr_name} = {expected} "
                f"(C++/Python enum drift; update both in lockstep)"
            )
    globals()[name] = cls
    return cls
