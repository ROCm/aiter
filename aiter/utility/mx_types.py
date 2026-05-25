# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single source of truth for MX-format scale rounding mode and dtype enums.

The C++ definitions in ``csrc/include/mx_quant_utils.h`` are exported to
Python via pybind11 in ``csrc/include/rocm_ops.hpp::AITER_CORE_PYBIND``;
this module imports them through aiter's JIT loader so any caller (Python
ops, CPU torch ref, FlyDSL IR builders) sees the **same** enum class as
the HIP kernels do. To add or change a mode/dtype, edit the C++ enum and
rebuild ``module_aiter_core``; nothing on the Python side hard-codes the
values.

Mirrors the existing pattern used by :mod:`aiter.ops.enum` for
``ActivationType`` and ``QuantType``: ``@compile_ops`` is the JIT loader
that ensures ``module_aiter_core.so`` exists (compiling on demand if not),
then we extract the pybind11 enum class via ``type(_factory(0))``.

Importers:
  - ``aiter.utility.fp4_utils``           (CPU torch reference quantizers)
  - ``aiter.ops.quant``                   (Python user-facing quant ops)
  - ``aiter.ops.flydsl.kernels.quant_utils`` (FlyDSL IR builders)

Cross-stack naming aligns with PyTorch torchao ``ScaleCalculationMode``,
NV Triton, DSv4, FlashInfer, and AMD Quark ``RoundMode``.
"""

from ..jit.core import compile_ops


# ``_MxScaleRoundMode`` and ``_MxDtype`` are listed in
# ``aiter/jit/utils/torch_guard.py::NONE_WRAPPED_OP`` so the
# ``@compile_ops`` decorator skips the ``torch.library.infer_schema`` step
# (which would otherwise reject the untyped ``dummy`` argument). The
# wrapped functions still trigger module_aiter_core to build/import on
# first call; we only need a trivial call to extract the enum class.
@compile_ops("module_aiter_core", "MxScaleRoundMode")
def _MxScaleRoundMode(dummy): ...


@compile_ops("module_aiter_core", "MxDtype")
def _MxDtype(dummy): ...


# Pulling the enum classes off the binding gives every importer the *same*
# class object (verifiable via ``is``-comparison), so the C++
# ``enum class MxScaleRoundMode`` and Python ``MxScaleRoundMode`` are not
# just numerically equal -- they are literally the same type.
MxScaleRoundMode = type(_MxScaleRoundMode(0))
MxDtype = type(_MxDtype(0))
