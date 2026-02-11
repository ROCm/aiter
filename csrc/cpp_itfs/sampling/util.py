# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.


import torch

def get_seed_and_offset(
    increment,
    generator = None,
    device = "cuda",
):
    # Update the generator state so that subsequent calls don't reuse the same
    # random numbers
    if generator is None:
        generator = torch.cuda.default_generators[torch.device(device).index]
    state = generator.get_state()
    seed, offset = state.view(torch.int64)
    offset += (increment + 3) // 4 * 4
    generator.set_state(
        torch.tensor(
            [seed, offset], dtype=torch.int64, device=torch.device("cpu")
        ).view(torch.uint8)
    )
    return int(seed), int(offset)
