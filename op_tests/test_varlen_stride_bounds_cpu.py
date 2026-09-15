# SPDX-License-Identifier: MIT
"""No GPU import: test exact production metadata bound and per-call wiring."""

import ast
import json
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
    exec(  # noqa: S102
        compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns
    )
    return ns[node.name]


def production_tiles():
    root = SOURCE.parent.parent / "configs"
    cases = []
    for path in sorted(root.glob("*/triton/attention/mha/*.json")):
        for variant, config in json.loads(path.read_text())["fwd"].items():
            cases.append(pytest.param(config, id=f"{path.relative_to(root)}:{variant}"))
    assert cases, "Production MHA configurations were not found"
    return cases


@pytest.fixture(params=production_tiles())
def production_config(request):
    return request.param


@pytest.fixture
def fits(production_config):
    def check(tensors, length=65536, batch=1, heads=16):
        return bound()(tensors, length, length, batch, heads, production_config)

    return check


def test_h3_geometry_and_lse(fits):
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
def test_unsafe_or_unsupported_layout(tensor, fits):
    assert not fits([tensor])


def test_output_and_lse_participate_in_bound(fits):
    safe = TensorMetadata((65536, 16, 128), (2048, 128, 1))
    unsafe = TensorMetadata((65536, 16), (2**30, 1))
    assert not fits([safe, safe, safe, safe, unsafe])


def test_masked_coordinate_overflow_is_not_numel_check(fits, production_config):
    tile = max(production_config["BLOCK_M"], production_config["BLOCK_N"])
    length = (2**31 // 2048) // 2 - tile + 1
    tensor = TensorMetadata((length, 16, 128), (2048, 128, 1))
    assert tensor.numel() < 2**31
    assert not fits([tensor], length=length)


def test_grid_id_overflow(fits):
    tensor = TensorMetadata((1, 1, 128), (128, 128, 1))
    assert not fits([tensor], length=1, batch=2**30, heads=8)


@pytest.mark.parametrize(
    "qlen,klen,window",
    [(1, 2**31 - 1, 0), (2**31 - 1, 1, 0), (10**9, 1, 1500000000)],
)
def test_sequence_coordinates_independent_of_strides(
    qlen, klen, window, production_config
):
    tensor = TensorMetadata((1, 1, 8), (0, 8, 1))
    assert not bound()([tensor], qlen, klen, 1, 1, production_config, window)


def test_head_tile_minimum_16_includes_masked_lanes(fits):
    tensor = TensorMetadata((1, 2, 8), (0, 2**31 - 8, 1))
    assert not fits([tensor], length=1, heads=2)


def test_coordinate_guard_boundary(production_config):
    tensor = TensorMetadata((1, 1, 8), (0, 8, 1))
    kwargs = {
        "tensors": [tensor],
        "max_seqlen_q": 1,
        "max_seqlen_k": 1,
        "batch": 1,
        "heads": 1,
        "config": production_config,
    }
    tile = max(production_config["BLOCK_M"], production_config["BLOCK_N"])
    window = 2**31 - 1 - 2 - 2 * tile
    assert bound()(**kwargs, sliding_window=window)
    assert not bound()(**kwargs, sliding_window=window + 1)


def test_bound_has_no_thread_shared_decision(fits):
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
