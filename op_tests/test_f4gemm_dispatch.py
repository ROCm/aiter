# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 F4GEMM ASM/Gluon bit-match checks.

Run the GPU comparisons with:
    python -m pytest -v -s op_tests/test_f4gemm_dispatch.py -k asm_gluon_bitmatch

For arbitrary shapes, seeds, and JSON diagnostics, use test_f4gemm.py's
--check-gluon CLI. Comparisons use the same logical data and scales, shuffled
separately for each backend, and require identical BF16 storage bits.
"""

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
