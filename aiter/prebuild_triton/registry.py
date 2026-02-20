# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kernel Specification Registry for Triton precompilation.

Each KernelSpec defines a @triton.jit kernel function and a method to
enumerate all (constexpr, signature, compile_options) variants that
should be precompiled.

The registry is organized by operator category (attention, normalization,
activation, quantization, fusions, MoE routing). Phase 1 covers
attention + normalization + activation + quantization + fusions.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .shapes import (
    ALL_MODELS,
    COMMON_HIDDEN_SIZES,
)


def _unwrap_kernel(fn):
    """Unwrap @triton.heuristics / @triton.autotune to get the inner @triton.jit function.
    ASTSource requires a JITFunction, not a Heuristics/Autotuner wrapper.
    Stop unwrapping once we reach a JITFunction (which has arg_names).
    Handles both triton 3.5 (Heuristics.fn) and 3.6+ (Heuristics.fn) chains."""
    from triton.runtime import JITFunction

    while not isinstance(fn, JITFunction):
        if hasattr(fn, "fn"):
            fn = fn.fn
        else:
            break
    return fn


@dataclass
class KernelVariant:
    """A single compilation variant for a kernel."""

    constexprs: Dict[str, Any]
    signature: Dict[str, str]
    num_warps: int = 4
    num_stages: int = 1
    waves_per_eu: int = 0


@dataclass
class KernelSpec:
    """Specification for a Triton kernel to precompile."""

    name: str
    kernel_fn: Any  # the @triton.jit function
    get_variants: Callable[[], List[KernelVariant]]
    description: str = ""


def _next_power_of_2(n):
    """Return the next power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


# ---------------------------------------------------------------------------
# Variant generators for each kernel category
# ---------------------------------------------------------------------------


def _get_rmsnorm_variants():
    """Generate variants for _rms_norm_kernel."""
    variants = []
    # Typical hidden sizes determine BLOCK_SIZE
    for n_cols in COMMON_HIDDEN_SIZES:
        block_size = min(65536 // 2, _next_power_of_2(n_cols))  # bf16 = 2 bytes
        for use_blocked in [False, True]:
            num_prgms = 304 if use_blocked else 1  # MI300X has 304 CUs
            variants.append(
                KernelVariant(
                    constexprs={
                        "BLOCK_SIZE": block_size,
                        "USE_BLOCKED": use_blocked,
                        "NUM_PRGMS": num_prgms,
                    },
                    signature={
                        "input_ptr": "*bf16",
                        "output_ptr": "*bf16",
                        "g_ptr": "*bf16",
                        "rsigma_ptr": "*fp32",
                        "input_row_stride": "i64",
                        "output_row_stride": "i64",
                        "n_rows": "i32",
                        "n_cols": "i32",
                        "epsilon": "fp32",
                    },
                    num_warps=4 if block_size >= 2048 else 2,
                )
            )
    return variants


def _get_fused_add_rmsnorm_variants():
    """Generate variants for _fused_add_rmsnorm_kernel."""
    variants = []
    for n_cols in COMMON_HIDDEN_SIZES:
        block_size = min(65536 // 2, _next_power_of_2(n_cols))
        for use_blocked in [False, True]:
            num_prgms = 304 if use_blocked else 1
            variants.append(
                KernelVariant(
                    constexprs={
                        "BLOCK_SIZE": block_size,
                        "USE_BLOCKED": use_blocked,
                        "NUM_PRGMS": num_prgms,
                    },
                    signature={
                        "input_ptr": "*bf16",
                        "output_ptr": "*bf16",
                        "res_in_ptr": "*bf16",
                        "res_out_ptr": "*bf16",
                        "g_ptr": "*bf16",
                        "rsigma_ptr": "*fp32",
                        "input_row_stride": "i64",
                        "output_row_stride": "i64",
                        "n_rows": "i32",
                        "n_cols": "i32",
                        "epsilon": "fp32",
                    },
                    num_warps=4 if block_size >= 2048 else 2,
                )
            )
    return variants


def _get_quant_rmsnorm_variants():
    """Generate variants for _quant_rms_norm_kernel (RMSNorm + FP8 quant)."""
    variants = []
    for n_cols in COMMON_HIDDEN_SIZES:
        block_size = min(65536 // 2, _next_power_of_2(n_cols))
        for is_smooth in [False, True]:
            for clamp_max in [False, True]:
                variants.append(
                    KernelVariant(
                        constexprs={
                            "DTYPE_MAX": 448.0,  # FP8 E4M3 max
                            "IS_SMOOTH": is_smooth,
                            "CLAMP_MAX": clamp_max,
                            "CLAMP_OUT": True,
                            "DUMP_INTERMEDIATE": False,
                            "BLOCK_SIZE": block_size,
                        },
                        signature={
                            "input_ptr": "*bf16",
                            "output_ptr": "*fp8e4nv",
                            "x_scale_ptr": "*fp32",
                            "y_scale_ptr": "*fp32",
                            "g_ptr": "*bf16",
                            "aux_ptr": "*fp32",
                            "input_row_stride": "i64",
                            "output_row_stride": "i64",
                            "aux_row_stride": "i64",
                            "n_rows": "i32",
                            "n_cols": "i32",
                            "epsilon": "fp32",
                            "scale_ub_ptr": "*fp32",
                            "out_intermediate_ptr": "*bf16",
                        },
                        num_warps=4 if block_size >= 2048 else 2,
                    )
                )
    return variants


def _get_fused_rms_fp8_quant_variants():
    """Generate variants for _fused_rms_fp8_per_tensor_static_quant_kernel."""
    variants = []
    for n_cols in COMMON_HIDDEN_SIZES:
        block_size_n = _next_power_of_2(n_cols)
        if block_size_n <= 1024:
            num_warps = 1
        elif block_size_n <= 2048:
            num_warps = 4
        elif block_size_n <= 4096:
            num_warps = 8
        else:
            num_warps = 16
        for have_second in [False, True]:
            for first_res in [False, True]:
                for first_out in [False, True]:
                    for rmsnorm_convert in [False, True]:
                        variants.append(
                            KernelVariant(
                                constexprs={
                                    "BLOCK_SIZE_N": block_size_n,
                                    "DTYPE_MAX": 448.0,
                                    "DTYPE_MIN": -448.0,
                                    "HAVE_SECOND_INPUT": have_second,
                                    "FIRST_INPUT_RES": first_res,
                                    "FIRST_INPUT_OUT": first_out,
                                    "RMSNORM_CONVERT_TO_INP1_TYPE": rmsnorm_convert,
                                },
                                signature={
                                    "inp1_ptr": "*bf16",
                                    "weight1_ptr": "*bf16",
                                    "inp2_ptr": "*bf16",
                                    "weight2_ptr": "*bf16",
                                    "res1_ptr": "*bf16",
                                    "out1_fp8_ptr": "*fp8e4nv",
                                    "out2_ptr": "*bf16",
                                    "out_res1_ptr": "*bf16",
                                    "out1_ptr": "*bf16",
                                    "scale_ptr": "*fp32",
                                    "eps1": "fp32",
                                    "eps2": "fp32",
                                    "n_rows": "i32",
                                    "inp1_n_cols": "i32",
                                    "inp2_n_cols": "i32",
                                    "inp1_row_stride": "i64",
                                    "inp2_row_stride": "i64",
                                    "inp1_col_stride": "i64",
                                    "inp2_col_stride": "i64",
                                    "res1_row_stride": "i64",
                                    "res1_col_stride": "i64",
                                    "out1_fp8_row_stride": "i64",
                                    "out1_fp8_col_stride": "i64",
                                    "out2_row_stride": "i64",
                                    "out2_col_stride": "i64",
                                    "out_res1_row_stride": "i64",
                                    "out_res1_col_stride": "i64",
                                    "out1_row_stride": "i64",
                                    "out1_col_stride": "i64",
                                },
                                num_warps=num_warps,
                            )
                        )
    return variants


def _get_act_mxfp4_quant_variants():
    """Generate variants for _act_mul_and_dynamic_mxfp4_quant_kernel."""
    variants = []
    activations = ["silu", "silu_exp2", "gelu", "gelu_tanh"]
    block_configs = [
        # (BLOCK_SIZE_M, BLOCK_SIZE_N, NUM_WARPS)
        (8, 128, 2),
        (8, 256, 4),
        (16, 128, 2),
        (16, 256, 4),
    ]
    for activation in activations:
        for block_m, block_n, num_warps in block_configs:
            for shuffle in [False, True]:
                for even_m_n in [False, True]:
                    variants.append(
                        KernelVariant(
                            constexprs={
                                "BLOCK_SIZE_M": block_m,
                                "BLOCK_SIZE_N": block_n,
                                "NUM_ITER": 1,
                                "NUM_STAGES": 1,
                                "MXFP4_QUANT_BLOCK_SIZE": 32,
                                "EVEN_M_N": even_m_n,
                                "SCALING_MODE": 0,
                                "ACTIVATION": activation,
                                "scaleN": block_n // 32,
                                "scaleM_pad": 1,
                                "scaleN_pad": block_n // 32,
                                "SHUFFLE": shuffle,
                            },
                            signature={
                                "x_ptr": "*bf16",
                                "x_fp4_ptr": "*i8",
                                "bs_ptr": "*i8",
                                "stride_x_m_in": "i32",
                                "stride_x_n_in": "i32",
                                "stride_x_fp4_m_in": "i32",
                                "stride_x_fp4_n_in": "i32",
                                "stride_bs_m_in": "i32",
                                "stride_bs_n_in": "i32",
                                "M": "i32",
                                "N": "i32",
                            },
                            num_warps=num_warps,
                        )
                    )
    return variants


def _get_act_fp8_group_quant_variants():
    """Generate variants for _act_mul_and_dynamic_fp8_group_quant_kernel."""
    variants = []
    activations = ["silu", "silu_exp2", "gelu"]
    for activation in activations:
        for block_n in [128, 256]:
            for quant_block in [128]:
                for even_n in [False, True]:
                    variants.append(
                        KernelVariant(
                            constexprs={
                                "ACTIVATION": activation,
                                "scaleN": block_n // quant_block,
                                "BLOCK_SIZE_N": block_n,
                                "QUANT_BLOCK_SIZE": quant_block,
                                "DTYPE_MAX": 448.0,
                                "DTYPE_MIN": -448.0,
                                "EVEN_N": even_n,
                            },
                            signature={
                                "x_ptr": "*bf16",
                                "x_fp8_ptr": "*fp8e4nv",
                                "x_bs_ptr": "*fp32",
                                "stride_x_m_in": "i32",
                                "stride_x_n_in": "i32",
                                "stride_x_fp8_m_in": "i32",
                                "stride_x_fp8_n_in": "i32",
                                "stride_bs_m_in": "i32",
                                "stride_bs_n_in": "i32",
                                "N": "i32",
                            },
                            num_warps=2,
                        )
                    )
    return variants


def _get_unified_attention_variants():
    """Generate variants for kernel_unified_attention_2d."""
    variants = []
    for model in ALL_MODELS.values():
        head_dim = model["head_dim"]
        head_dim_padded = _next_power_of_2(head_dim)
        num_query_heads = model["num_heads"]
        num_kv_heads = model.get("num_kv_heads", model["num_heads"])
        num_queries_per_kv = num_query_heads // num_kv_heads
        block_size = model.get("block_size", 16)

        # Decode config: ALL_DECODE=True, BLOCK_M based on num_queries_per_kv
        for all_decode in [True, False]:
            if all_decode:
                block_m = min(16, _next_power_of_2(num_queries_per_kv))
                tile_size = min(64, _next_power_of_2(block_size))
                num_warps = 2
                num_stages = 3
            else:
                block_m = 128
                tile_size = min(64, _next_power_of_2(block_size))
                num_warps = 4
                num_stages = 1

            block_q = max(1, block_m // num_queries_per_kv)

            for use_fp8 in [False, True]:
                for use_alibi in [False]:
                    for use_softcap in [False]:
                        for sliding_window in [0]:
                            variants.append(
                                KernelVariant(
                                    constexprs={
                                        "scale": 1.0 / math.sqrt(head_dim),
                                        "num_query_heads": num_query_heads,
                                        "num_queries_per_kv": num_queries_per_kv,
                                        "BLOCK_SIZE": block_size,
                                        "TILE_SIZE": tile_size,
                                        "HEAD_SIZE": head_dim,
                                        "HEAD_SIZE_PADDED": head_dim_padded,
                                        "USE_ALIBI_SLOPES": use_alibi,
                                        "USE_QQ_BIAS": False,
                                        "USE_SOFTCAP": use_softcap,
                                        "USE_SINKS": False,
                                        "SLIDING_WINDOW": sliding_window,
                                        "stride_k_cache_3": 1,
                                        "stride_v_cache_3": 1,
                                        "BLOCK_Q": block_q,
                                        "BLOCK_M": block_m,
                                        "USE_FP8": use_fp8,
                                        "FP8_MIN": -448.0,
                                        "FP8_MAX": 448.0,
                                        "ALL_DECODE": all_decode,
                                    },
                                    signature={
                                        "output_ptr": "*bf16",
                                        "query_ptr": "*bf16",
                                        "key_cache_ptr": "*bf16",
                                        "value_cache_ptr": "*bf16",
                                        "sink_ptr": "*i32",
                                        "block_tables_ptr": "*i32",
                                        "seq_lens_ptr": "*i32",
                                        "alibi_slopes_ptr": "*fp32",
                                        "qq_bias_ptr": "*fp32",
                                        "k_scale": "fp32",
                                        "v_scale": "fp32",
                                        "out_scale": "fp32",
                                        "softcap": "fp32",
                                        "block_table_stride": "i64",
                                        "query_stride_0": "i64",
                                        "query_stride_1": "i64",
                                        "output_stride_0": "i64",
                                        "output_stride_1": "i64",
                                        "qq_bias_stride_0": "i64",
                                        "stride_k_cache_0": "i64",
                                        "stride_k_cache_1": "i64",
                                        "stride_k_cache_2": "i64",
                                        "stride_v_cache_0": "i64",
                                        "stride_v_cache_1": "i64",
                                        "stride_v_cache_2": "i64",
                                        "query_start_len_ptr": "*i32",
                                        "num_seqs": "i32",
                                    },
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                    waves_per_eu=2,
                                )
                            )
    return variants


def _get_fused_rope_cache_variants():
    """Generate variants for _fused_qk_rope_reshape_and_cache_kernel."""
    variants = []
    for model in ALL_MODELS.values():
        head_dim = model["head_dim"]
        block_size = model.get("block_size", 16)
        num_heads = model["num_heads"]
        num_kv_heads = model.get("num_kv_heads", model["num_heads"])
        qh_per_kh = num_heads // num_kv_heads

        for is_neox in [True]:
            for flash_layout in [False, True]:
                for have_k_scale in [False, True]:
                    for have_v_scale in [False, True]:
                        variants.append(
                            KernelVariant(
                                constexprs={
                                    "QH_PER_KH": qh_per_kh,
                                    "QH": num_heads,
                                    "KH": num_kv_heads,
                                    "REUSE_FREQS_FRONT_PART": True,
                                    "IS_NEOX": is_neox,
                                    "BLOCK_D_pe": head_dim,
                                    "BLOCK_D_HALF_pe": head_dim // 2,
                                    "BLOCK_SIZE": block_size,
                                    "X_SIZE": 0 if flash_layout else 8,
                                    "FLASH_LAYOUT": flash_layout,
                                    "HAVE_POS": True,
                                    "HAVE_K_SCALE": have_k_scale,
                                    "HAVE_V_SCALE": have_v_scale,
                                    "HAVE_ZEROS": False,
                                },
                                signature={
                                    "q_ptr": "*bf16",
                                    "k_ptr": "*bf16",
                                    "v_ptr": "*bf16",
                                    "pos_ptr": "*i64",
                                    "cos_ptr": "*bf16",
                                    "sin_ptr": "*bf16",
                                    "offs_ptr": "*i64",
                                    "key_cache_ptr": "*bf16",
                                    "value_cache_ptr": "*bf16",
                                    "slot_mapping_ptr": "*i32",
                                    "q_out_ptr": "*bf16",
                                    "k_out_ptr": "*bf16",
                                    "zeros_out_ptr": "*bf16",
                                    "T": "i32",
                                    "T_slot": "i32",
                                    "q_stride_t": "i64",
                                    "q_stride_h": "i64",
                                    "q_stride_d": "i64",
                                    "k_stride_t": "i64",
                                    "k_stride_h": "i64",
                                    "k_stride_d": "i64",
                                    "v_stride_t": "i64",
                                    "v_stride_h": "i64",
                                    "v_stride_d": "i64",
                                    "cos_stride_t": "i64",
                                    "cos_stride_d": "i64",
                                    "q_out_stride_t": "i64",
                                    "q_out_stride_h": "i64",
                                    "q_out_stride_d": "i64",
                                    "k_out_stride_t": "i64",
                                    "k_out_stride_h": "i64",
                                    "k_out_stride_d": "i64",
                                    "key_cache_stride_t": "i64",
                                    "key_cache_stride_h": "i64",
                                    "key_cache_stride_d": "i64",
                                    "key_cache_stride_b": "i64",
                                    "key_cache_stride_x": "i64",
                                    "value_cache_stride_t": "i64",
                                    "value_cache_stride_h": "i64",
                                    "value_cache_stride_d": "i64",
                                    "value_cache_stride_b": "i64",
                                    "zeros_out_stride_t": "i64",
                                    "zeros_out_stride_h": "i64",
                                    "zeros_out_stride_d": "i64",
                                    "k_scale_ptr": "*fp32",
                                    "v_scale_ptr": "*fp32",
                                },
                                num_warps=1,
                            )
                        )
    return variants


def _get_fused_mla_rope_cache_variants():
    """Generate variants for _fused_qk_rope_cat_and_cache_mla_kernel (DeepSeek MLA)."""
    variants = []
    ds = ALL_MODELS.get("deepseek_v3")
    if ds is None:
        return variants

    d_nope = ds.get("qk_nope_head_dim", 128)
    dk_nope = ds.get("kv_lora_rank", 512)
    d_pe = ds.get("qk_rope_head_dim", 64)
    num_heads = ds["num_heads"]
    num_kv_heads = ds.get("num_kv_heads", 1)
    qh_per_kh = num_heads // num_kv_heads

    for is_neox in [True]:
        for have_k_scale in [False, True]:
            for output_zeros in [False, True]:
                variants.append(
                    KernelVariant(
                        constexprs={
                            "QH_PER_KH": qh_per_kh,
                            "QH": num_heads,
                            "KH": num_kv_heads,
                            "REUSE_FREQS_FRONT_PART": True,
                            "IS_NEOX": is_neox,
                            "BLOCK_D_nope": d_nope,
                            "BLOCK_DK_nope": dk_nope,
                            "BLOCK_D_pe": d_pe,
                            "BLOCK_D_HALF_pe": d_pe // 2,
                            "OUTPUT_Q_NOPE_ZEROS": output_zeros,
                            "HAVE_K_SCALE": have_k_scale,
                        },
                        signature={
                            "q_nope_ptr": "*bf16",
                            "q_pe_ptr": "*bf16",
                            "k_nope_ptr": "*bf16",
                            "k_pe_ptr": "*bf16",
                            "pos_ptr": "*i64",
                            "cos_ptr": "*bf16",
                            "sin_ptr": "*bf16",
                            "q_out_ptr": "*bf16",
                            "decode_q_pe_out_ptr": "*bf16",
                            "k_pe_out_ptr": "*bf16",
                            "q_nope_zeros_out_ptr": "*bf16",
                            "kv_cache_ptr": "*bf16",
                            "slot_mapping_ptr": "*i32",
                            "B": "i32",
                            "B_slot": "i32",
                            "num_decode_toks_for_zeros": "i32",
                            "q_nope_stride_b": "i64",
                            "q_nope_stride_h": "i64",
                            "q_nope_stride_d": "i64",
                            "q_pe_stride_b": "i64",
                            "q_pe_stride_h": "i64",
                            "q_pe_stride_d": "i64",
                            "k_nope_stride_b": "i64",
                            "k_nope_stride_h": "i64",
                            "k_nope_stride_d": "i64",
                            "k_pe_stride_b": "i64",
                            "k_pe_stride_h": "i64",
                            "k_pe_stride_d": "i64",
                            "pos_stride_b": "i64",
                            "cos_stride_b": "i64",
                            "cos_stride_d": "i64",
                            "q_out_stride_b": "i64",
                            "q_out_stride_h": "i64",
                            "q_out_stride_d": "i64",
                            "decode_q_pe_out_stride_b": "i64",
                            "decode_q_pe_out_stride_h": "i64",
                            "decode_q_pe_out_stride_d": "i64",
                            "k_pe_out_stride_b": "i64",
                            "k_pe_out_stride_h": "i64",
                            "k_pe_out_stride_d": "i64",
                            "q_nope_zeros_out_stride_b": "i64",
                            "q_nope_zeros_out_stride_h": "i64",
                            "q_nope_zeros_out_stride_d": "i64",
                            "kv_cache_stride_b": "i64",
                            "kv_cache_stride_h": "i64",
                            "kv_cache_stride_d": "i64",
                            "k_scale_ptr": "*fp32",
                        },
                        num_warps=1,
                    )
                )
    return variants


def _get_moe_routing_variants():
    """Generate variants for _routing_compute_indx."""
    variants = []
    block_ms = [32, 64, 128]
    n_expts_acts = [8, 16]

    for block_m in block_ms:
        for n_expts_act in n_expts_acts:
            for even_m in [False, True]:
                variants.append(
                    KernelVariant(
                        constexprs={
                            "BLOCK_M": block_m,
                            "EVEN_M": even_m,
                            "N_EXPTS_ACT": n_expts_act,
                        },
                        signature={
                            "pid_m": "i32",
                            "GatherIndx": "*i32",
                            "ScatterIndx": "*i32",
                            "GateScal": "*fp32",
                            "ExptScal": "*fp32",
                            "ExptIndx": "*i32",
                            "PartialOffs": "*i32",
                            "stride_pm": "i64",
                            "stride_pn": "i64",
                            "TokensStart": "*i32",
                            "n_gates": "i32",
                        },
                        num_warps=1,
                    )
                )
    return variants


# ---------------------------------------------------------------------------
# Registry: map name -> (kernel_fn_getter, variant_generator)
# Using lazy imports to avoid importing heavy modules at registry load time.
# ---------------------------------------------------------------------------

_KERNEL_REGISTRY = {
    "_rms_norm_kernel": {
        "module": "aiter.ops.triton._triton_kernels.normalization.rmsnorm",
        "fn_name": "_rms_norm_kernel",
        "get_variants": _get_rmsnorm_variants,
        "description": "RMSNorm forward kernel",
    },
    "_fused_add_rmsnorm_kernel": {
        "module": "aiter.ops.triton._triton_kernels.normalization.rmsnorm",
        "fn_name": "_fused_add_rmsnorm_kernel",
        "get_variants": _get_fused_add_rmsnorm_variants,
        "description": "Fused residual add + RMSNorm",
    },
    "_quant_rms_norm_kernel": {
        "module": "aiter.ops.triton._triton_kernels.normalization.rmsnorm",
        "fn_name": "_quant_rms_norm_kernel",
        "get_variants": _get_quant_rmsnorm_variants,
        "description": "RMSNorm + FP8 quantization",
    },
    "_fused_rms_fp8_per_tensor_static_quant_kernel": {
        "module": "aiter.ops.triton._triton_kernels.quant.fused_fp8_quant",
        "fn_name": "_fused_rms_fp8_per_tensor_static_quant_kernel",
        "get_variants": _get_fused_rms_fp8_quant_variants,
        "description": "Fused RMSNorm + FP8 per-tensor static quantization",
    },
    "_act_mul_and_dynamic_mxfp4_quant_kernel": {
        "module": "aiter.ops.triton._triton_kernels.activation",
        "fn_name": "_act_mul_and_dynamic_mxfp4_quant_kernel",
        "get_variants": _get_act_mxfp4_quant_variants,
        "description": "Activation (SiLU/GeLU) + MUL + MXFP4 quantization",
    },
    "_act_mul_and_dynamic_fp8_group_quant_kernel": {
        "module": "aiter.ops.triton._triton_kernels.activation",
        "fn_name": "_act_mul_and_dynamic_fp8_group_quant_kernel",
        "get_variants": _get_act_fp8_group_quant_variants,
        "description": "Activation + MUL + FP8 group quantization",
    },
    "kernel_unified_attention_2d": {
        "module": "aiter.ops.triton._triton_kernels.attention.unified_attention",
        "fn_name": "kernel_unified_attention_2d",
        "get_variants": _get_unified_attention_variants,
        "description": "Unified paged attention (decode + prefill)",
    },
    "_fused_qk_rope_reshape_and_cache_kernel": {
        "module": "aiter.ops.triton._triton_kernels.fusions.fused_kv_cache",
        "fn_name": "_fused_qk_rope_reshape_and_cache_kernel",
        "get_variants": _get_fused_rope_cache_variants,
        "description": "Fused QK RoPE + reshape + KV cache update",
    },
    "_fused_qk_rope_cat_and_cache_mla_kernel": {
        "module": "aiter.ops.triton._triton_kernels.fusions.fused_kv_cache",
        "fn_name": "_fused_qk_rope_cat_and_cache_mla_kernel",
        "get_variants": _get_fused_mla_rope_cache_variants,
        "description": "Fused MLA QK RoPE + concat + KV cache (DeepSeek)",
    },
    "_routing_compute_indx": {
        "module": "aiter.ops.triton._triton_kernels.moe.moe_routing.routing",
        "fn_name": "_routing_compute_indx",
        "get_variants": _get_moe_routing_variants,
        "description": "MoE routing index computation",
    },
}


def get_kernel_specs():
    """
    Return list of KernelSpec objects for all registered kernels.
    Uses lazy imports to load kernel functions only when needed.
    """
    import importlib

    specs = []
    for name, entry in _KERNEL_REGISTRY.items():
        try:
            mod = importlib.import_module(entry["module"])
            kernel_fn = getattr(mod, entry["fn_name"])
            # Unwrap @triton.heuristics / @triton.autotune wrappers
            kernel_fn = _unwrap_kernel(kernel_fn)
            specs.append(
                KernelSpec(
                    name=name,
                    kernel_fn=kernel_fn,
                    get_variants=entry["get_variants"],
                    description=entry.get("description", ""),
                )
            )
        except Exception as e:
            print(f"[prebuild_triton] Warning: could not load kernel {name}: {e}")
    return specs


def list_registered_kernels():
    """List all registered kernel names and descriptions."""
    for name, entry in _KERNEL_REGISTRY.items():
        print(f"  {name}: {entry.get('description', '')}")


def estimate_variant_count():
    """Estimate total number of variants without loading kernels."""
    total = 0
    for name, entry in _KERNEL_REGISTRY.items():
        try:
            variants = entry["get_variants"]()
            count = len(variants)
            total += count
            print(f"  {name}: {count} variants")
        except Exception as e:
            print(f"  {name}: ERROR - {e}")
    print(f"  Total: {total} variants per arch")
    return total
