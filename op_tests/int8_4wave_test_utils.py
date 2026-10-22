# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight

DEFAULT_INT8_4WAVE_SHAPES = (
    (9450, 5120, 5120),
    (9450, 13824, 5120),
    (9450, 5120, 13824),
)
REPRESENTATIVE_INT8_4WAVE_M_VALUES = (512, 9450, 9472, 18900, 37800, 75600)


def make_int8_4wave_case(shape, seed):
    m, n, k = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = (
        torch.rand((m, 1), dtype=torch.float32, device="cuda", generator=generator)
        * 1e-3
        + 1e-4
    )
    w_scale = (
        torch.rand((1, n), dtype=torch.float32, device="cuda", generator=generator)
        * 1e-3
        + 1e-4
    )
    return {
        "x": x,
        "weight": weight,
        "weight_shuffled": shuffle_weight(weight, layout=(16, 16)).contiguous(),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def int8_4wave_reference(case):
    accum = F.linear(case["x"].float(), case["weight"].float())
    return (accum * (case["x_scale"] * case["w_scale"])).to(torch.bfloat16)
