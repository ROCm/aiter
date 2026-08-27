# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.utils.types import e4m3_dtype


def get_dtype_str(dtype: torch.dtype):
    if dtype == torch.uint8:
        return "nvfp4"
    if dtype == e4m3_dtype:
        return "fp8"
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return "bf16"
    raise ValueError(f"No unified attention config tag for dtype: {dtype}")


def _dtype_keys(q_tag: str, kv_tag: str) -> str:
    return (f"{q_tag}_{kv_tag}", f"{q_tag}_any", f"any_{kv_tag}", "any")


def get_attention_config() -> tuple[dict, bool]:
    pass
