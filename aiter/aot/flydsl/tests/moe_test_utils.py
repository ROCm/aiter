# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared CPU-only metadata fixtures for MoE compile-request tests."""

from aiter.ops.flydsl.compile_request import RocmTarget

TARGET = RocmTarget("gfx950", 256)


def stage1_metadata(**overrides):
    values = {
        "model_dim": 7168,
        "inter_dim": 2048,
        "experts": 256,
        "topk": 8,
        "tile_m": 32,
        "tile_n": 128,
        "tile_k": 256,
        "doweight_stage1": False,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "act": "silu",
        "persist_m": 1,
        "use_async_copy": False,
        "k_batch": 1,
        "waves_per_eu": 3,
        "b_nt": 2,
        "gate_mode": "separated",
        "model_dim_pad": 0,
        "inter_dim_pad": 0,
        "enable_bias": False,
        "a_scale_one": False,
        "xcd_swizzle": 0,
        "k_wave": 1,
        "v2_output_layout": False,
    }
    values.update(overrides)
    return values


def stage2_metadata(**overrides):
    values = {
        "model_dim": 7168,
        "inter_dim": 2048,
        "experts": 256,
        "topk": 8,
        "tile_m": 32,
        "tile_n": 128,
        "tile_k": 256,
        "doweight_stage2": True,
        "a_dtype": "fp8",
        "b_dtype": "fp4",
        "out_dtype": "bf16",
        "sort_block_m": 32,
        "waves_per_eu": None,
        "use_async_copy": False,
        "use_global_a": False,
        "cu_num_mul": 1,
        "b_nt": 2,
        "model_dim_pad": 0,
        "inter_dim_pad": 0,
        "xcd_swizzle": 0,
        "enable_bias": False,
    }
    values.update(overrides)
    return values


def stage2_runtime_metadata(**overrides):
    values = {
        "mode": "atomic",
        "accumulate": True,
        "return_per_slot": False,
        "persist": None,
        "token_num": 128,
        "routing_block_count": 288,
        "dtype_str": "bf16",
        "use_mask": False,
        "topk_ids_available": False,
        "num_experts": 0,
        "fp8_intermediate": False,
        "out_dtype_str": "bf16",
        "use_weight": False,
        "scale_blk": None,
        "pitch_align": None,
    }
    values.update(overrides)
    return values


def request_argument_names(request):
    return tuple(argument.name for argument in request.signature.arguments)
