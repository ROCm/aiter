# SPDX-License-Identifier: MIT
"""aiter.ops.flydsl.kernels.mega_moe.shared_expert (MXFP8-weight tile bodies of the shared expert, default off): nothing on a served
path imports it, and its weight / scale addressing is the stored formats' (CPU only; no kernel is compiled).
"""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("flydsl")


def test_served_modules_do_not_import_shared_expert():
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import aiter.ops.flydsl.kernels.mega_moe.mega_moe_v2, aiter.ops.flydsl.kernels.mega_moe.mega_moe_small_m\n"
        "import aiter.ops.flydsl.kernels.mega_moe.gemm1, aiter.ops.flydsl.kernels.mega_moe.gemm2\n"
        "import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config\n"
        "assert 'aiter.ops.flydsl.kernels.mega_moe.shared_expert' not in sys.modules\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[
        -2000:
    ]


def _shared_expert():
    pytest.importorskip("aiter.ops.flydsl.kernels.silu_mul_quant_epilogue")
    from aiter.ops.flydsl.kernels.mega_moe import shared_expert

    return shared_expert


def test_kernel_names_carry_every_compile_knob():
    se = _shared_expert()
    names = {
        se.gemm1_name(5120, 2304, 10.0, ksplit, cm)
        for ksplit in (1, 2)
        for cm in (0, 3)
    } | {se.gemm2_name(2304, 5120, cm) for cm in (0, 3)}
    assert len(names) == 6
    assert all(name.startswith("megamoe_shared_gemm") for name in names)
    # A routed kernel's name can never collide with one of these.
    assert not any(name.startswith("megamoe_small_stage1") for name in names)


@pytest.mark.parametrize("n, k", [(4608, 5120), (5120, 2304)])
def test_weight_group_is_the_preshuffled_chunk(n, k):
    se = _shared_expert()
    shuffle = pytest.importorskip("aiter.ops.shuffle")
    gen = torch.Generator().manual_seed(n + k)
    w = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=gen)
    stored = shuffle.shuffle_weight(w, layout=(16, 16)).contiguous().view(-1, 16)
    for _ in range(400):
        row = int(torch.randint(0, n // 16, (1,), generator=gen)) * 16
        k128 = int(torch.randint(0, k // 128, (1,), generator=gen))
        half, k_lane, row_lane = (
            int(torch.randint(0, hi, (1,), generator=gen)) for hi in (2, 4, 16)
        )
        col = 128 * k128 + 64 * half + 16 * k_lane
        group = se.f8_weight_group(row, k128, half, k_lane, row_lane, k)
        assert torch.equal(stored[group], w[row + row_lane, col : col + 16]), (
            row,
            k128,
            half,
            k_lane,
            row_lane,
        )


def test_gate_up_rows_pair_one_activation_column_block():
    se = _shared_expert()
    half = 2304
    for row_base in (
        0,
        64,
        128 * 2 * 5 + 64 * 3,
    ):  # wave w of n-tile t: 64 w + 256 t (interleaved gate / up columns)
        column = row_base // 2
        assert [se.gate_up_row(row_base, ni, half) for ni in range(4)] == [
            column,
            half + column,
            column + 16,
            half + column + 16,
        ]
        assert [se.gate_up_row(row_base, ni, 0) for ni in range(4)] == [
            row_base + 16 * ni for ni in range(4)
        ]


@pytest.mark.parametrize("rows, k", [(4608, 5120), (5120, 2304)])
def test_scale_dword_holds_the_four_k_group_scales(rows, k):
    se = _shared_expert()
    gen = torch.Generator().manual_seed(rows)
    scale = torch.randint(
        0, 256, (rows // 32, k // 32), dtype=torch.uint8, generator=gen
    )
    dwords = scale.contiguous().view(-1, 4)
    for _ in range(400):
        row32 = int(torch.randint(0, rows // 32, (1,), generator=gen))
        kstep = int(torch.randint(0, k // 256, (1,), generator=gen))
        ksub = int(torch.randint(0, 2, (1,), generator=gen))
        first = (256 * kstep + 128 * ksub) // 32
        assert torch.equal(
            dwords[se.scale32_dword(row32, kstep, ksub, k)],
            scale[row32, first : first + 4],
        )
