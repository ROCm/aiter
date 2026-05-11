# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch


def _get_compiled(fn):
    return torch.compile(
        fn, backend="inductor", fullgraph=True, options={"max_autotune": True}
    )
