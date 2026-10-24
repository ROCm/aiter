# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-tensor HIP quantization must stay inside the logical row tail.

Regression test for https://github.com/ROCm/aiter/issues/6647. Widths that are
not a multiple of four used to read input storage past the logical tensor into
the dynamic scale, and write up to three bytes past the logical output.

Each case runs twice on the same logical input: once followed by zeros and once
by a large poison value in the input storage. The output is written into a
guarded buffer. Neither the guard bytes nor the scale/output may depend on what
follows the logical tensor.
"""

import pytest
import torch

from aiter.ops.quant import dynamic_per_tensor_quant, static_per_tensor_quant
from aiter.utility import dtypes

GUARD = 32
GUARD_BYTE = 0xA5
POISON = 10000.0
# Multiples of four keep the unchanged vector path and act as controls.
WIDTHS = [1, 2, 3, 4, 5, 7, 17, 33, 4097, 4099, 4100]
ROWS = [1, 3]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


def _logical_input(rows, width, in_dtype, out_dtype):
    """Values exactly representable in the input and output types, so the
    reference scale is the output maximum and the quantized values are exact."""
    if out_dtype == dtypes.i8:
        q_max, pattern = 127.0, [-96.0, 80.0, -64.0, 48.0, -32.0, 24.0, -16.0, 8.0]
    else:
        q_max = float(torch.finfo(out_dtype).max)
        pattern = [-192.0, 160.0, -112.0, 80.0, -56.0, 36.0, -24.0, 14.0]
    count = rows * width
    values = torch.tensor([pattern[(i * 5 + 3) % 8] for i in range(count)])
    values[0] = -q_max if count > 1 else q_max
    return values.to(in_dtype), q_max


def _run(logical_cpu, suffix_value, out_dtype, rows, width, static_scale):
    count = rows * width
    suffix = torch.full((16,), suffix_value, dtype=logical_cpu.dtype)
    storage = torch.cat((logical_cpu, suffix)).cuda()
    x = storage[:count].view(rows, width)
    storage_before = storage.clone()

    out_storage = torch.full(
        (GUARD + count + GUARD,), GUARD_BYTE, dtype=torch.uint8, device="cuda"
    )
    out = out_storage[GUARD : GUARD + count].view(out_dtype).view(rows, width)
    if static_scale is None:
        scale = torch.empty(1, dtype=torch.float32, device="cuda")
        dynamic_per_tensor_quant(out, x, scale)
    else:
        scale = static_scale.clone()
        static_per_tensor_quant(out, x, scale)
    torch.cuda.synchronize()

    problems = []
    if not torch.equal(storage, storage_before):
        problems.append("input storage was modified")
    if not torch.all(out_storage[:GUARD] == GUARD_BYTE):
        problems.append("wrote before the output")
    overrun = int((out_storage[GUARD + count :] != GUARD_BYTE).sum())
    if overrun:
        problems.append(f"wrote {overrun} byte(s) past the logical output")
    return out.view(torch.uint8).clone(), scale.clone(), x, problems


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", ["dynamic", "static"])
@pytest.mark.parametrize("out_dtype", [dtypes.fp8, dtypes.i8], ids=["fp8", "i8"])
@pytest.mark.parametrize("in_dtype", [dtypes.bf16, dtypes.fp16], ids=["bf16", "fp16"])
def test_per_tensor_quant_tail(in_dtype, out_dtype, mode, rows, width):
    logical, q_max = _logical_input(rows, width, in_dtype, out_dtype)
    static_scale = (
        torch.ones(1, dtype=torch.float32, device="cuda") if mode == "static" else None
    )

    out_zero, scale_zero, x, problems_zero = _run(
        logical, 0.0, out_dtype, rows, width, static_scale
    )
    out_poison, scale_poison, _, problems_poison = _run(
        logical, POISON, out_dtype, rows, width, static_scale
    )
    problems = [f"zero suffix: {p}" for p in problems_zero]
    problems += [f"poison suffix: {p}" for p in problems_poison]
    if not torch.equal(scale_zero, scale_poison):
        problems.append(
            "scale depends on input past the tensor: "
            f"{scale_zero.item()} vs {scale_poison.item()}"
        )
    if not torch.equal(out_zero, out_poison):
        problems.append("output depends on input past the tensor")
    assert not problems, "; ".join(problems)

    ref_scale = x.float().abs().max().view(1) / q_max
    torch.testing.assert_close(scale_zero, ref_scale, rtol=1e-6, atol=0)
    ref = (x.float() / scale_zero).to(out_dtype).float()
    out = out_zero.view(out_dtype).view(rows, width).float()
    torch.testing.assert_close(out, ref, rtol=0, atol=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
