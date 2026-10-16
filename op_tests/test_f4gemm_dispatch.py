# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 F4GEMM ASM/Gluon bit-match and CPU launch-geometry checks.

Run the GPU comparisons with:
    python -m pytest -v -s op_tests/test_f4gemm_dispatch.py -k asm_gluon_bitmatch

For arbitrary shapes, seeds, and JSON diagnostics, use test_f4gemm.py's
--check-gluon CLI. Comparisons use the same logical data and scales, shuffled
separately for each backend, and require identical BF16 storage bits.
"""

import ctypes
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from op_tests._f4gemm_test_utils import (
    get_dispatch_lib,
)
from op_tests._f4gemm_test_utils import (
    query_launch_config as _launch_config,
)

_NPS_KERNEL = "_ZN5aiter41f4gemm_bf16_mxfp4_BpreShuffle_192x256_4x4E"
_PS_KERNEL = "_ZN5aiter44f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_psE"


@pytest.fixture(scope="module", autouse=True)
def default_dispatch_environment():
    # Overrides are cached on first use; test them in fresh subprocesses below.
    with pytest.MonkeyPatch.context() as patch:
        patch.delenv("AITER_F4GEMM_TUNE", raising=False)
        yield


@pytest.fixture(scope="module")
def f4gemm_module():
    pytest.importorskip("torch")
    from op_tests import test_f4gemm

    return test_f4gemm


@pytest.fixture(scope="module")
def f4gemm_ut(request):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("ASM/Gluon bit-match requires a gfx1250 GPU")
    arch = getattr(
        torch.cuda.get_device_properties(torch.cuda.current_device()), "gcnArchName", ""
    )
    if arch.split(":")[0] != "gfx1250":
        pytest.skip("ASM/Gluon bit-match requires gfx1250")
    pytest.importorskip("triton")
    from aiter.jit.utils.asm_guard import is_gfx1250_asm_supported

    if not is_gfx1250_asm_supported():
        pytest.skip("F4GEMM ASM requires gfx1250 B0 or newer")

    # Importing the CLI module has no default-device side effect. Scope the
    # CUDA default to this fixture, even when pytest collected that file first.
    with torch.device("cuda"):
        yield request.getfixturevalue("f4gemm_module")


@pytest.mark.parametrize(
    "M,N,K,apre",
    [
        pytest.param(1536, 24576, 1536, 0, id="1536x24576x1536-apre0"),
        pytest.param(1536, 24576, 1536, 1, id="1536x24576x1536-apre1"),
        pytest.param(1536, 7168, 16384, 0, id="1536x7168x16384-apre0"),
        pytest.param(1536, 7168, 16384, 1, id="1536x7168x16384-apre1"),
        # Exercise Gluon's row-major vs preshuffled A-scale boundary.
        pytest.param(1, 256, 256, 0, id="m1-apre0"),
        pytest.param(17, 256, 256, 0, id="m17-apre0"),
        pytest.param(32, 256, 256, 0, id="m32-apre0"),
        pytest.param(32, 256, 256, 1, id="m32-apre1"),
    ],
)
@pytest.mark.parametrize(
    "data_init,scale_init",
    [
        pytest.param("constant", "constant", id="constant"),
        pytest.param("uniform", "auto", id="uniform"),
    ],
)
def test_asm_gluon_bitmatch(f4gemm_ut, M, N, K, apre, data_init, scale_init):
    ut = f4gemm_ut
    inp, _ = ut._prep_mxfp4(
        M, N, K, apre, data_init, scale_init, ut.make_generator(0), check_gluon=True
    )
    out = ut.aiter.gemm_mxfp4_asm(
        inp["A"], inp["B"], inp["sA"], inp["sB"], a_preshuffle=bool(apre)
    )
    gluon_out = ut.run_gluon_mxfp4(*inp["gluon_args"])
    result = ut._compare_gluon_output(out, gluon_out)
    assert result["gluon mismatches"] == 0, result


@pytest.mark.parametrize(
    "asm_bits,gluon_bits,expected_mismatches",
    [
        pytest.param([0x7FC1], [0x7FC1], 0, id="identical-nan-payload"),
        pytest.param([0x0000], [0x8000], 1, id="signed-zero"),
        pytest.param([0x7FC1], [0x7FC2], 1, id="different-nan-payload"),
        pytest.param([0x3F80], [0x3F81], 1, id="one-bf16-ulp"),
    ],
)
def test_bitmatch_storage_bits(
    f4gemm_module, asm_bits, gluon_bits, expected_mismatches
):
    ut = f4gemm_module
    torch = ut.torch
    asm_out = torch.tensor([asm_bits], dtype=torch.uint16, device="cpu").view(
        torch.bfloat16
    )
    gluon_out = torch.tensor([gluon_bits], dtype=torch.uint16, device="cpu").view(
        torch.bfloat16
    )
    result = ut._compare_gluon_output(asm_out, gluon_out)
    assert result["gluon mismatches"] == expected_mismatches
    assert result["gluon bitmatch"] == ("failed" if expected_mismatches else "pass")


@pytest.mark.parametrize("K", [256, 1536, 1792])
@pytest.mark.parametrize(
    "data_init,scale_init", [("constant", "constant"), ("uniform", "auto")]
)
@pytest.mark.parametrize("seed", [0, 1])
def test_nps_asm_gluon_bitmatch(f4gemm_ut, K, data_init, scale_init, seed):
    ut = f4gemm_ut
    M, N = 768, 1024
    inp, _ = ut._prep_mxfp4(
        M, N, K, 0, data_init, scale_init, ut.make_generator(seed), check_gluon=True
    )
    y = ut.torch.empty((M, N), dtype=ut.dtypes.bf16, device=inp["A"].device)
    args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
    out = ut.aiter.gemm_mxfp4_asm(
        *args, a_preshuffle=False, kernelName=_NPS_KERNEL, y=y
    )
    assert out is y
    persistent = ut.aiter.gemm_mxfp4_asm(
        *args, a_preshuffle=False, kernelName=_PS_KERNEL
    )
    ut.torch.testing.assert_close(
        out.view(ut.torch.int16), persistent.view(ut.torch.int16), rtol=0, atol=0
    )
    result = ut._compare_gluon_output(out, ut.run_gluon_mxfp4(*inp["gluon_args"]))
    assert result["gluon mismatches"] == 0, result


@pytest.mark.parametrize("apre", [0, 1])
@pytest.mark.parametrize("outtype", ["bf16", "fp8"])
@pytest.mark.parametrize("intype", ["mxfp4", "nvfp4"])
def test_preallocated_output(f4gemm_ut, apre, outtype, intype):
    ut = f4gemm_ut
    M, N, K = 32, 128, 256
    prep = ut._prep_mxfp4 if intype == "mxfp4" else ut._prep_nvfp4
    gemm = getattr(ut.aiter, f"gemm_{intype}_asm")
    inp, _ = prep(M, N, K, apre, "uniform", "auto", ut.make_generator(0))
    args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
    if intype == "nvfp4":
        args += (inp["gA"], inp["gB"])
    dtype = ut._OUT_DTYPE[outtype]
    expected = gemm(*args, dtype=dtype, a_preshuffle=bool(apre))
    y = ut.torch.empty((M, N), dtype=dtype, device=inp["A"].device)
    y.view(ut.torch.uint8).fill_(0xFF)
    result = gemm(*args, dtype=dtype, a_preshuffle=bool(apre), y=y)
    if outtype == "fp8":
        out, scale = result
        expected, expected_scale = expected
        ut.torch.testing.assert_close(
            ut.unpack_mxfp8_out_scale(scale, M, N).view(ut.torch.uint8),
            ut.unpack_mxfp8_out_scale(expected_scale, M, N).view(ut.torch.uint8),
            rtol=0,
            atol=0,
        )
    else:
        out = result
    assert out is y
    ut.torch.testing.assert_close(
        out.view(ut.torch.uint8), expected.view(ut.torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device", "strides"])
@pytest.mark.parametrize("intype", ["mxfp4", "nvfp4"])
def test_reject_invalid_output(f4gemm_ut, invalid, intype):
    ut = f4gemm_ut
    M, N, K = 32, 128, 256
    prep = ut._prep_mxfp4 if intype == "mxfp4" else ut._prep_nvfp4
    gemm = getattr(ut.aiter, f"gemm_{intype}_asm")
    inp, _ = prep(M, N, K, 0, "constant", "constant", ut.make_generator(0))
    args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
    if intype == "nvfp4":
        args += (inp["gA"], inp["gB"])
    shape = (M, N + 1) if invalid == "shape" else (M, N)
    dtype = ut.torch.float32 if invalid == "dtype" else ut.dtypes.bf16
    device = "cpu" if invalid == "device" else inp["A"].device
    y = ut.torch.empty(shape, dtype=dtype, device=device)
    if invalid == "strides":
        y = y.t().contiguous().t()
    with pytest.raises(ValueError, match="y must be contiguous"):
        gemm(*args, a_preshuffle=False, y=y)


@pytest.mark.parametrize("intype", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("outtype", ["bf16", "fp8"])
@pytest.mark.parametrize("apre", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
def test_empty_batch(f4gemm_module, monkeypatch, intype, outtype, apre, preallocated):
    ut = f4gemm_module
    torch = ut.torch
    from aiter.ops import gemm_op_a4w4 as ops

    def unexpected_launch(*args, **kwargs):
        pytest.fail("an empty batch must not launch an ASM kernel")

    monkeypatch.setattr(ops, "require_gfx1250_asm", lambda name: None)
    monkeypatch.setattr(ops, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(ops, "_mxfp4_gemm_asm", unexpected_launch)
    monkeypatch.setattr(ops, "_nvfp4_gemm_asm", unexpected_launch)
    N, K = 128, 256
    A = torch.empty((0, K // 2), dtype=torch.uint8, device="cpu")
    B = torch.empty((N, K // 2), dtype=torch.uint8, device="cpu")
    block = 32 if intype == "mxfp4" else 16
    sA = torch.empty((0, K // block), dtype=torch.uint8, device="cpu")
    sB = torch.empty((N, K // block), dtype=torch.uint8, device="cpu")
    dtype = ut._OUT_DTYPE[outtype]
    y = torch.empty((0, N), dtype=dtype, device="cpu") if preallocated else None
    args = (A, B, sA, sB)
    scales = (1.0, 1.0) if intype == "nvfp4" else ()
    result = getattr(ops, f"gemm_{intype}_asm")(
        *args, *scales, dtype=dtype, a_preshuffle=apre, y=y
    )
    out = result[0] if outtype == "fp8" else result
    assert out.shape == (0, N) and out.dtype == dtype and out.device == A.device
    if preallocated:
        assert out is y
    if outtype == "fp8":
        assert result[1].shape == (0, 1) and result[1].dtype == ut.dtypes.fp8_e8m0

    # Also cover the public dispatch and preservation of empty batch dimensions.
    kwargs = {"apreshuffle": apre}
    if intype == "nvfp4":
        kwargs["global_A_scale"] = torch.tensor(1.0, device="cpu")
    for leading in [(0,), (2, 0)]:
        args = (A.reshape(*leading, K // 2), B, sA, sB)
        if outtype == "fp8":
            out, scale = ops.gemm_a4w4o8(*args, **kwargs)
            assert scale.shape == (0, 1)
        else:
            out = ops.gemm_a4w4(*args, **kwargs)
        assert out.shape == (*leading, N) and out.dtype == dtype


@pytest.mark.parametrize("knl_name", [None, "unregistered_kernel"])
@pytest.mark.parametrize(
    "message",
    [
        "cannot get heuristic kernel",
        "kernel not in cfg_f4gemm",
        "no kernel registered for f4gemm",
        "selected kernel not in f4gemm manifest",
    ],
)
def test_missing_config_is_reported(f4gemm_module, monkeypatch, knl_name, message):
    ut = f4gemm_module

    def missing_config(*args, **kwargs):
        raise RuntimeError(message)

    def unexpected_prep(*args, **kwargs):
        pytest.fail("unsupported configs must be rejected before input preparation")

    monkeypatch.setattr(ut, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(ut, "_kernel_config", missing_config)
    monkeypatch.setattr(ut, "_prep_mxfp4", unexpected_prep)
    row = ut.test_gemm("mxfp4", 32, 128, 256, 0, knl_name=knl_name, check_gluon=True)
    assert row["asm result"] == f"not support ({message})"
    assert row["knl_name"] == (knl_name or "auto")
    assert row["tile"] == "n/a"
    assert row["gluon bitmatch"] == "not checked"


def test_config_errors_propagate(f4gemm_module, monkeypatch):
    def broken_config(*args, **kwargs):
        raise RuntimeError("unexpected dispatch failure")

    monkeypatch.setattr(f4gemm_module, "_kernel_config", broken_config)
    with pytest.raises(RuntimeError, match="unexpected dispatch failure"):
        f4gemm_module.test_gemm("mxfp4", 32, 128, 256, 0)


@pytest.mark.parametrize(
    "shape,cluster",
    [
        ((32, 256, 256), (2, 1)),
        ((768, 256, 256), (1, 2)),
        ((768, 1024, 256), (4, 1)),
        ((512, 512, 256), (2, 2)),
        ((1024, 256, 256), (1, 4)),
        ((512, 1024, 256), (4, 2)),
        ((1024, 512, 256), (2, 4)),
        ((1024, 1024, 256), (4, 4)),
    ],
)
@pytest.mark.parametrize("intype", ["mxfp4", "nvfp4"])
@pytest.mark.parametrize("outtype", ["bf16", "fp8"])
@pytest.mark.parametrize("apre", [0, 1])
def test_persistent_cluster_outputs(f4gemm_ut, shape, cluster, intype, outtype, apre):
    ut = f4gemm_ut
    pre = "ABpreShuffle" if apre else "BpreShuffle"
    base = f"f4gemm_{outtype}_{intype}_{pre}_256x256_4x4_ps"
    kernel = f"_ZN5aiter{len(base)}{base}E"
    row = ut.test_gemm(
        intype,
        *shape,
        apre,
        outtype,
        mode="func",
        knl_name=kernel,
        num_iters=3,
        num_warmup=0,
        num_rotate=1,
        check_gluon=intype == "mxfp4" and outtype == "bf16",
    )
    assert row["ps_mode"] == 1 and row["cluster"] == f"{cluster[0]}x{cluster[1]}"
    assert row["asm result"] == "pass", row
    if intype == "mxfp4" and outtype == "bf16":
        assert row["gluon bitmatch"] == "pass", row


def test_gemm_defaults_use_preshuffled_a(f4gemm_ut):
    ut = f4gemm_ut
    M, N, K = 768, 1024, 256
    cfg = ut._kernel_config(M, N, K, "mxfp4", "bf16", 1)
    assert (cfg["tile_m"], cfg["tile_n"], cfg["ps_mode"]) == (256, 256, 1)
    inp, _ = ut._prep_mxfp4(
        M, N, K, 1, "uniform", "auto", ut.make_generator(0), check_gluon=True
    )
    out = ut.aiter.gemm_mxfp4_asm(inp["A"], inp["B"], inp["sA"], inp["sB"])
    result = ut._compare_gluon_output(out, ut.run_gluon_mxfp4(*inp["gluon_args"]))
    assert result["gluon mismatches"] == 0, result


@pytest.fixture(scope="module")
def dispatch_lib(geometry_lib):
    lib = geometry_lib
    try:
        _launch_config(lib, 768, 1024, 256, kernel=_PS_KERNEL)
    except RuntimeError as e:
        if "no kernel registered for f4gemm" in str(e):
            pytest.skip("the dispatch module was built without gfx1250 F4GEMM kernels")
        raise
    return lib


@pytest.mark.parametrize(
    "shape,intype,outtype,apre,expected",
    [
        ((768, 1024, 256), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((768, 1024, 1536), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((768, 1024, 1792), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((1536, 24576, 1536), "mxfp4", "bf16", 0, (4, 4, 24, 2, 192, 256, 0)),
        ((1536, 7168, 16384), "mxfp4", "bf16", 0, (4, 4, 7, 2, 192, 256, 0)),
        ((3072, 8192, 8192), "mxfp4", "bf16", 0, (4, 4, 8, 4, 192, 256, 0)),
        ((2304, 16384, 16384), "mxfp4", "bf16", 0, (4, 4, 16, 3, 192, 256, 0)),
        ((768, 50176, 256), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((12288, 16384, 16384), "mxfp4", "bf16", 0, (4, 4, 4, 4, 256, 256, 1)),
        ((767, 1024, 256), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1008, 256), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 288), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "mxfp4", "bf16", 1, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "mxfp4", "fp8", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "nvfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
    ],
)
def test_launch_config(dispatch_lib, shape, intype, outtype, apre, expected):
    assert _launch_config(dispatch_lib, *shape, intype, outtype, apre) == expected


def test_query_cache_is_scoped_by_architecture(dispatch_lib):
    expected = _launch_config(dispatch_lib, 768, 1024, 256)
    with pytest.raises(RuntimeError, match="cannot get heuristic kernel"):
        _launch_config(dispatch_lib, 768, 1024, 256, arch="gfx950")
    assert _launch_config(dispatch_lib, 768, 1024, 256) == expected


def test_explicit_nps_bypasses_heuristic_size_bound(dispatch_lib):
    assert _launch_config(dispatch_lib, 12288, 16384, 16384, kernel=_NPS_KERNEL) == (
        4,
        4,
        16,
        16,
        192,
        256,
        0,
    )


def test_persistent_alignment_error(dispatch_lib):
    with pytest.raises(
        RuntimeError, match="persistent kernels require N%16=0"
    ) as error:
        _launch_config(dispatch_lib, 768, 1000, 256, kernel=_PS_KERNEL)
    assert "NPS" not in str(error.value)


@pytest.mark.parametrize(
    "shape", [(767, 1024, 256), (768, 1008, 256), (768, 1024, 288)]
)
def test_reject_partial_nps_cluster(dispatch_lib, shape):
    with pytest.raises(RuntimeError, match="does not support M/N/K"):
        _launch_config(dispatch_lib, *shape, kernel=_NPS_KERNEL)


@pytest.mark.parametrize(
    "intype,outtype,apre",
    [("nvfp4", "bf16", 0), ("mxfp4", "fp8", 0), ("mxfp4", "bf16", 1)],
)
def test_reject_nps_format_mismatch(dispatch_lib, intype, outtype, apre):
    with pytest.raises(RuntimeError, match="mismatches requested"):
        _launch_config(dispatch_lib, 768, 1024, 256, intype, outtype, apre, _NPS_KERNEL)


@pytest.mark.parametrize(
    "tune,expected,error",
    [
        (None, (4, 1, 16, 4, 256, 256, 1), ""),
        ("4,4,4", (4, 4, 4, 4, 256, 256, 1), ""),
        ("3,2,4", None, "cluster axes"),
        ("4,4", None, "must be cluster_x,cluster_y,grid_y"),
    ],
)
def test_persistent_override_preserves_nps(dispatch_lib, tune, expected, error):
    source = r"""
import json
import os
import sys
import pytest
from op_tests._f4gemm_test_utils import load_dispatch_lib, query_launch_config as _launch_config
from op_tests.test_f4gemm_dispatch import _PS_KERNEL

lib = load_dispatch_lib(sys.argv[1])
assert _launch_config(lib, 768, 1024, 256) == (4, 4, 1, 1, 192, 256, 0)
if sys.argv[2]:
    with pytest.raises(RuntimeError, match=sys.argv[2]):
        _launch_config(lib, 768, 1024, 256, kernel=_PS_KERNEL)
else:
    expected = tuple(json.loads(sys.argv[3]))
    assert _launch_config(lib, 768, 1024, 256, kernel=_PS_KERNEL) == expected
    os.environ["AITER_F4GEMM_TUNE"] = "invalid-after-first-use"
    assert _launch_config(lib, 768, 1024, 256, kernel=_PS_KERNEL) == expected
"""
    env = dict(os.environ)
    if tune is None:
        env.pop("AITER_F4GEMM_TUNE", None)
    else:
        env["AITER_F4GEMM_TUNE"] = tune
    proc = subprocess.run(
        [sys.executable, "-c", source, dispatch_lib._name, error, json.dumps(expected)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def geometry_lib():
    try:
        return get_dispatch_lib()
    except FileNotFoundError as error:
        pytest.skip(str(error))


def _geometry(lib, M, N):
    values = (ctypes.c_int * 4)()
    lib.f4gemm_test_geometry(M, N, values)
    return tuple(values)


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1536, 24576), (4, 2, 8, 4)),
        ((1536, 7168), (4, 2, 8, 4)),
        ((16384, 16384), (4, 4, 4, 4)),
        ((256, 2048), (4, 1, 64, 1)),
        ((1, 16), (2, 1, 128, 1)),
    ],
)
def test_representative_geometry(geometry_lib, shape, expected):
    assert _geometry(geometry_lib, *shape) == expected


@pytest.mark.parametrize(
    "geometry",
    [
        (1, 1, 4),
        (16, 1, 4),
        (1, 16, 4),
        (8, 4, 4),
        (3, 2, 4),
        (0, 4, 4),
        (-2, 4, 4),
        (4, 4, 0),
        (4, 4, 3),
        (4, 4, 32),
    ],
)
def test_reject_invalid_geometry(geometry_lib, geometry):
    assert not geometry_lib.f4gemm_test_valid_geometry(*geometry)


def test_persistent_walk_covers_every_tile_once(geometry_lib):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 28, 31, 32, 63, 64, 65, 96]
    for mt, nt in itertools.product(sizes, repeat=2):
        # Include partial 256x256 tiles while retaining the N%16 contract.
        M, N = mt * 256 - 255, nt * 256 - 240
        cx, cy, gx, gy = _geometry(geometry_lib, M, N)
        assert cx * cy * gx * gy == 256
        assert geometry_lib.f4gemm_test_valid_geometry(cx, cy, gy)
        rows, cols = (mt + cy - 1) // cy, (nt + cx - 1) // cx
        total = rows * cols
        interior_cols = cols // gx * gx
        interior = interior_cols * rows
        visited = set()
        count = 0
        # Reproduce ttmp9/ttmp7 flattening, column-band decode, and the
        # persistent stride. Compare coverage with the Cartesian tile domain.
        for launch_y, launch_x in itertools.product(range(gy), range(gx)):
            flat = launch_y * gx + launch_x
            while flat < total:
                if flat < interior:
                    band = (flat // gx) // rows
                    y = (flat // gx) % rows
                    x = band * gx + flat % gx
                else:
                    tail = flat - interior
                    y = tail // (cols - interior_cols)
                    x = interior_cols + tail % (cols - interior_cols)
                for wy, wx in itertools.product(range(cy), range(cx)):
                    tile = (y * cy + wy, x * cx + wx)
                    if tile[0] < mt and tile[1] < nt:
                        visited.add(tile)
                        count += 1
                flat += gx * gy
        expected = set(itertools.product(range(mt), range(nt)))
        assert visited == expected and count == len(expected), (mt, nt, cx, cy, gx, gy)
        old_slots = ((mt + 3) // 4) * ((nt + 3) // 4) * 16
        assert total * cx * cy <= old_slots, (mt, nt, cx, cy)
