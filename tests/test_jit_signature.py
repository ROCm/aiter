# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A compiled op's public signature must be its own.

``torch_compile_guard`` returns a ``(*args, **kwargs)`` closure, so without
``__wrapped__`` every ``@compile_ops`` function reports that as its signature.
Anything reading the public surface sees no parameter names and no defaults --
including the kernel provider interface, which captures the signature as the op
contract and cannot apply defaults it cannot see.

Needs a built AITER, so it is not part of the CPU-only gate.
"""

from __future__ import annotations

import inspect

from conftest import requires_built_aiter

requires_built_aiter()


def test_a_compiled_op_reports_its_own_parameters():
    from aiter.ops.rmsnorm import rmsnorm

    parameters = inspect.signature(rmsnorm).parameters

    assert list(parameters) == ["out", "input", "weight", "epsilon", "gemma_norm"]
    assert parameters["gemma_norm"].default is False


def test_a_compiled_op_can_be_routed_without_the_generic_binder():
    """The contract is only usable if a binder can be generated against it."""

    from aiter.ops.rmsnorm import rmsnorm
    from aiter.plugin import _build_canonicalizer, _canonicalizer_source

    contract = inspect.signature(rmsnorm)
    assert _canonicalizer_source(contract) is not None, (
        "a (*args, **kwargs) signature falls back to Signature.bind, which is "
        "sixty times more expensive and cannot apply the op's defaults"
    )

    canonicalize = _build_canonicalizer(contract)
    args, kwargs = canonicalize("out", "input", "weight", 1e-6)

    assert args == ("out", "input", "weight", 1e-6, False), "the default is applied"
    assert kwargs == {}
