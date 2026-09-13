# SPDX-License-Identifier: MIT
"""No GPU import: test exact production metadata bound and per-call wiring."""

import ast
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "aiter/ops/triton/attention/mha.py"


class TensorMetadata:
    def __init__(self, shape, strides, offset=0):
        self.shape = shape
        self.ndim = len(shape)
        self._strides = strides
        self._offset = offset

    def numel(self):
        return math.prod(self.shape)

    def stride(self):
        return self._strides

    def storage_offset(self):
        return self._offset


def bound():
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_varlen_int32_addressable"
    )
    ns = {}
    # Execute only the checked-in metadata helper, avoiding GPU initialization.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)  # noqa: S102
    return ns[node.name]


def fits(tensors, length=65536, batch=1, heads=16):
    return bound()(
        tensors, length, length, batch, heads, {"BLOCK_M": 128, "BLOCK_N": 64}
    )


def test_h3_geometry_and_lse():
    qkv = TensorMetadata((65536, 16, 128), (2048, 128, 1))
    lse = TensorMetadata((65536, 16), (16, 1))
    assert fits([qkv, qkv, qkv, qkv, lse])


@pytest.mark.parametrize(
    "tensor",
    [
        TensorMetadata((65536, 16, 128), (32768, 128, 1)),
        TensorMetadata((1, 16, 128), (2048, 128, 1), offset=2**31 - 1),
        TensorMetadata((1, 16, 128), (2**31, 128, 1)),
        TensorMetadata((1, 16, 128), (2048, -128, 1)),
        TensorMetadata((0, 16, 128), (2048, 128, 1)),
        TensorMetadata((1, 1, 16, 128), (2048, 2048, 128, 1)),
        TensorMetadata((2**31, 16, 128), (0, 128, 1)),
    ],
)
def test_unsafe_or_unsupported_layout(tensor):
    assert not fits([tensor])


def test_output_and_lse_participate_in_bound():
    safe = TensorMetadata((65536, 16, 128), (2048, 128, 1))
    unsafe = TensorMetadata((65536, 16), (2**30, 1))
    assert not fits([safe, safe, safe, safe, unsafe])


def test_masked_coordinate_overflow_is_not_numel_check():
    tensor = TensorMetadata((524200, 16, 128), (2048, 128, 1))
    assert tensor.numel() < 2**31
    assert not fits([tensor], length=524200)


def test_grid_id_overflow():
    tensor = TensorMetadata((1, 1, 128), (128, 128, 1))
    assert not fits([tensor], length=1, batch=2**30, heads=8)


def test_bound_has_no_thread_shared_decision():
    good = TensorMetadata((65536, 16, 128), (2048, 128, 1))
    bad = TensorMetadata((65536, 16, 128), (32768, 128, 1))
    with ThreadPoolExecutor(4) as pool:
        assert (
            list(pool.map(lambda t: fits([t]), [good, bad] * 32)) == [True, False] * 32
        )
    functions = [
        n
        for n in ast.walk(ast.parse(SOURCE.read_text()))
        if isinstance(n, ast.FunctionDef)
        and n.name
        in (
            "_flash_attn_forward",
            "flash_attn_varlen_func",
            "_varlen_int32_addressable",
        )
    ]
    assert not any(isinstance(n, ast.Global) for f in functions for n in ast.walk(f))


def test_autograd_argument_and_gradient_tuple_match():
    cls = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "_FlashAttnVarlenFunc"
    )
    fwd = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )
    bwd = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "backward"
    )
    ret = bwd.body[-1]
    assert isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple)
    assert len(ret.value.elts) == len(fwd.args.args) - 1
