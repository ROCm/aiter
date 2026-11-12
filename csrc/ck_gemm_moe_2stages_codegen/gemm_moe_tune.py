# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.tuning.moe.fused_moe import FmoeTuner, main
from aiter.tuning.moe.grouped import GroupedFmoeTuner
from aiter.tuning.moe.mxfp4 import Mxfp4FlydslTuner

torch.int4 = getattr(torch, "int4", torch.uint32)

__all__ = ["FmoeTuner", "GroupedFmoeTuner", "Mxfp4FlydslTuner", "main"]


if __name__ == "__main__":
    main()
