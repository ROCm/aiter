# SPDX-License-Identifier: MIT
"""Host-side MXFP4 import regressions, including CPython 3.10.9 inspect.

No GPU launch is required. The AST check protects older supported interpreters
even when CI runs on a newer interpreter with a fixed inspect implementation.
"""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

GLUON = "aiter.ops.triton._gluon_kernels.gfx1250.quant.fused_mxfp4_quant"
TRITON = "aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant"
KERNELS = [
    pytest.param(GLUON, "_gluon_fused_rms_mxfp4_quant_kernel", 0, id="gluon-rms"),
    pytest.param(
        GLUON, "_gluon_fused_reduce_rms_mxfp4_quant_kernel", 2, id="gluon-reduce"
    ),
    pytest.param(TRITON, "_fused_rms_mxfp4_quant_kernel", 1, id="triton-rms"),
    pytest.param(
        TRITON,
        "_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel",
        3,
        id="triton-act-num-iter",
    ),
    pytest.param(TRITON, "_fused_reduce_rms_mxfp4_quant_kernel", 2, id="triton-reduce"),
]

# Literal truth tables, independent of the implementation formula.
# Expected groups: Gluon RMS, Triton RMS, reduce RMS, iterative activation.
# Each group's order: EVEN_M_N, EVEN_M_N2, EVEN_M_N3 (where applicable).
CASES = [
    pytest.param(
        (32, 48, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (True,)),
        id="all-aligned",
    ),
    pytest.param(
        (8, 48, 64, 96, 3),
        ((True,), (False, False), (False, False, False), (False,)),
        id="only-rows-per-cta-aligned",
    ),
    pytest.param(
        (16, 48, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (False,)),
        id="block-m1-unaligned",
    ),
    pytest.param(
        (33, 48, 64, 96, 3),
        ((False,), (False, False), (False, False, False), (False,)),
        id="m-unaligned",
    ),
    pytest.param(
        (32, 40, 64, 96, 1),
        ((False,), (False, True), (False, True, True), (True,)),
        id="n1-eight-aligned-iter1",
    ),
    pytest.param(
        (32, 40, 64, 96, 3),
        ((False,), (False, True), (False, True, True), (False,)),
        id="n1-eight-aligned-iter3",
    ),
    pytest.param(
        (32, 16, 64, 96, 3),
        ((True,), (True, True), (True, True, True), (False,)),
        id="n1-sixteen-not-24-aligned",
    ),
    pytest.param(
        (32, 48, 65, 96, 3),
        ((True,), (True, False), (True, False, True), (True,)),
        id="n2-unaligned",
    ),
    pytest.param(
        (32, 48, 64, 97, 3),
        ((True,), (True, True), (True, True, False), (True,)),
        id="n3-unaligned",
    ),
    pytest.param(
        (0, 0, 0, 0, 3),
        ((True,), (True, True), (True, True, True), (True,)),
        id="zero-predicate-only",
    ),
]


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
def test_jit_import_and_complete_source(module_name, kernel_name, kind):
    kernel = getattr(importlib.import_module(module_name), kernel_name)
    # Heuristics -> JITFunction -> original Python function.
    assert "def " + kernel_name in inspect.getsource(kernel.fn.fn)


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
def test_decorators_remain_safe_for_legacy_inspect(module_name, kernel_name, kind):
    # No AITER import: detects reintroduced lambdas on Python 3.10.12/3.12 too.
    path = (
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(*module_name.split("."))
        .with_suffix(".py")
    )
    tree = ast.parse(path.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name
    )
    assert not any(
        isinstance(node, ast.Lambda)
        for decorator in fn.decorator_list
        for node in ast.walk(decorator)
    )


@pytest.mark.parametrize("module_name,kernel_name,kind", KERNELS)
@pytest.mark.parametrize("dimensions,expected", CASES)
def test_alignment_truth_table(module_name, kernel_name, kind, dimensions, expected):
    from aiter.ops.triton.utils.mxfp4_heuristics import even_m_n

    kernel = getattr(importlib.import_module(module_name), kernel_name)
    args = dict(zip(("M", "N1", "N2", "N3", "NUM_ITER"), dimensions))
    args.update(
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_M1=32,
        ROWS_PER_CTA=8,
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_N1=8,
        BLOCK_SIZE_N2=32,
        BLOCK_SIZE_N3=48,
    )
    names = ("EVEN_M_N", "EVEN_M_N2", "EVEN_M_N3")[: len(expected[kind])]
    assert set(kernel.values) == set(names)
    for name, value in zip(names, expected[kind]):
        predicate = kernel.values[name]
        assert predicate.func is even_m_n
        assert predicate(args) is value, name
