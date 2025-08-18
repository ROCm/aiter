# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from typing import Optional

from csrc.cpp_itfs.sampling.top_k_renorm_probs import (
    top_k_renorm_probs as top_k_renorm_probs_core,
)
from csrc.cpp_itfs.sampling.top_p_sampling_from_probs import (
    top_p_sampling_from_probs as top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.sampling.top_k_top_p_sampling_from_probs import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs_core,
)
from csrc.cpp_itfs.torch_utils import direct_register_custom_op


def top_k_renorm_probs(
    probs: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
) -> torch.Tensor:
    return top_k_renorm_probs_core(
        probs,
        maybe_top_k_arr,
        top_k_val,
    )


direct_register_custom_op(
    "top_k_renorm_probs",
    top_k_renorm_probs,
    [],
)


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    return top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_p_sampling_from_probs",
    top_p_sampling_from_probs,
    [],
)


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    indices: torch.Tensor,
    maybe_top_k_arr: Optional[torch.Tensor],
    top_k_val: int,
    maybe_top_p_arr: Optional[torch.Tensor],
    top_p_val: float,
    deterministic: bool = False,
) -> torch.Tensor:
    return top_k_top_p_sampling_from_probs_core(
        probs,
        indices,
        maybe_top_k_arr,
        top_k_val,
        maybe_top_p_arr,
        top_p_val,
        deterministic,
    )


direct_register_custom_op(
    "top_k_top_p_sampling_from_probs",
    top_k_top_p_sampling_from_probs,
    [],
)
