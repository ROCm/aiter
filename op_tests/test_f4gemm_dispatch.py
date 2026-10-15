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
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def f4gemm_ut():
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

    # The CLI module selects CUDA as its default device; restore the caller's
    # default when this module's tests finish.
    previous_device = torch.get_default_device()
    from op_tests import test_f4gemm

    try:
        yield test_f4gemm
    finally:
        torch.set_default_device(previous_device)


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
def test_bitmatch_storage_bits(f4gemm_ut, asm_bits, gluon_bits, expected_mismatches):
    ut = f4gemm_ut
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


@pytest.fixture(scope="module")
def dispatch_lib():
    jit_dir = Path(
        os.environ.get(
            "AITER_JIT_DIR", Path(__file__).resolve().parents[1] / "aiter/jit"
        )
    )
    path = jit_dir / "module_f4gemm_asm_nps_dispatch.so"
    if not path.is_file():
        pytest.skip(
            "build module_f4gemm_asm_nps_dispatch to check the dispatch metadata"
        )
    lib = ctypes.CDLL(str(path))
    lib.f4gemm_launch_config.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    lib.f4gemm_launch_config.restype = ctypes.c_int
    lib.aiter_get_last_error.argtypes = []
    lib.aiter_get_last_error.restype = ctypes.c_char_p
    return lib


def _launch_config(lib, M, N, K, intype="mxfp4", outtype="bf16", apre=0, kernel=None):
    values = []
    for field in range(7):
        value = lib.f4gemm_launch_config(
            M,
            N,
            K,
            intype.encode(),
            outtype.encode(),
            apre,
            kernel.encode() if kernel else None,
            field,
            None,
        )
        if value < 0:
            raise RuntimeError(lib.aiter_get_last_error().decode())
        values.append(value)
    return tuple(values)


_NPS_KERNEL = "_ZN5aiter41f4gemm_bf16_mxfp4_BpreShuffle_192x256_4x4E"
_PS_KERNEL = "_ZN5aiter44f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_psE"


@pytest.mark.parametrize(
    "shape,intype,outtype,apre,expected",
    [
        ((768, 1024, 256), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((768, 1024, 1536), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((768, 1024, 1792), "mxfp4", "bf16", 0, (4, 4, 1, 1, 192, 256, 0)),
        ((1536, 24576, 1536), "mxfp4", "bf16", 0, (4, 4, 24, 2, 192, 256, 0)),
        ((1536, 7168, 16384), "mxfp4", "bf16", 0, (4, 4, 7, 2, 192, 256, 0)),
        ((767, 1024, 256), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1008, 256), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 288), "mxfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "mxfp4", "bf16", 1, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "mxfp4", "fp8", 0, (4, 1, 16, 4, 256, 256, 1)),
        ((768, 1024, 256), "nvfp4", "bf16", 0, (4, 1, 16, 4, 256, 256, 1)),
    ],
)
def test_launch_config(
    dispatch_lib, monkeypatch, shape, intype, outtype, apre, expected
):
    monkeypatch.delenv("AITER_F4GEMM_TUNE", raising=False)
    assert _launch_config(dispatch_lib, *shape, intype, outtype, apre) == expected


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
    "shape", [(0, 1024, 256), (768, -1, 256), (768, 1024, 0), (2**31, 1024, 256)]
)
def test_reject_invalid_dimensions(dispatch_lib, shape):
    with pytest.raises(RuntimeError, match="positive int32 dimensions"):
        _launch_config(dispatch_lib, *shape)


def test_persistent_override_preserves_nps(dispatch_lib, monkeypatch):
    monkeypatch.setenv("AITER_F4GEMM_TUNE", "4,4,4")
    assert _launch_config(dispatch_lib, 768, 1024, 256) == (4, 4, 1, 1, 192, 256, 0)
    assert _launch_config(dispatch_lib, 768, 1024, 256, kernel=_PS_KERNEL) == (
        4,
        4,
        4,
        4,
        256,
        256,
        1,
    )
    monkeypatch.setenv("AITER_F4GEMM_TUNE", "3,2,4")
    with pytest.raises(RuntimeError, match="cluster axes"):
        _launch_config(dispatch_lib, 768, 1024, 256, kernel=_PS_KERNEL)


@pytest.fixture(scope="module")
def geometry_lib(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to exercise the dispatch header")
    output = tmp_path_factory.mktemp("f4gemm_geometry") / "geometry.so"
    include = Path(__file__).resolve().parents[1] / "csrc" / "include"
    source = r"""
        #include "asm_f4gemm_dispatch.hpp"
        extern "C" void select_geometry(int m, int n, int* out) {
            const auto g = aiter::f4gemm::select_launch_geometry(m, n);
            out[0] = g.cluster_x; out[1] = g.cluster_y;
            out[2] = g.grid_x; out[3] = g.grid_y;
        }
        extern "C" bool valid_geometry(int x, int y, int grid_y) {
            try { aiter::f4gemm::make_launch_geometry(x, y, grid_y); return true; }
            catch(const std::invalid_argument&) { return false; }
        }
    """
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-O2",
            f"-I{include}",
            "-x",
            "c++",
            "-",
            "-o",
            str(output),
        ],
        input=source,
        text=True,
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(output))
    lib.select_geometry.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.select_geometry.restype = None
    lib.valid_geometry.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.valid_geometry.restype = ctypes.c_bool
    return lib


def _geometry(lib, M, N):
    values = (ctypes.c_int * 4)()
    lib.select_geometry(M, N, values)
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
    assert not geometry_lib.valid_geometry(*geometry)


def test_persistent_walk_covers_every_tile_once(geometry_lib):
    sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 28, 31, 32, 63, 64, 65, 96]
    for mt, nt in itertools.product(sizes, repeat=2):
        # Include partial 256x256 tiles while retaining the N%16 contract.
        M, N = mt * 256 - 255, nt * 256 - 240
        cx, cy, gx, gy = _geometry(geometry_lib, M, N)
        assert cx * cy * gx * gy == 256
        assert geometry_lib.valid_geometry(cx, cy, gy)
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
