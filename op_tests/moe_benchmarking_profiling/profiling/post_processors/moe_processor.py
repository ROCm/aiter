#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
MOE-specific post-processor for combining profiling results.

Handles:
- Kernel name matching for ASM fmoe and CK moe_ck2stages kernels
- Kernel name from input config
- MOE-specific columns (token, model_dim, inter_dim, expert, topk, etc.)
"""

from typing import Any, Dict, Optional

from .base import PostProcessor
from profiling.script_generators.moe_generator import MoeKernelConfig


class MoePostProcessor(PostProcessor):
    """
    Post-processor for MOE kernel profiling results.
    
    Extracts target MOE kernels from trace/counters and adds
    MOE-specific configuration columns to the output.
    """
    
    def get_kernel_search_pattern(self, config: MoeKernelConfig) -> str:
        """
        Return pattern to find MOE kernel in Kernel_Name column.
        
        MOE kernels have different naming patterns:
        - ASM kernels: mangled names starting with _ZN containing 'fmoe'
        - CK kernels: names starting with 'moe_ck2stages'
        
        The actual kernel names in trace are demangled/templated, so we
        search for the characteristic substring.
        """
        kernel_name = config.kernel_name
        
        # ASM fmoe kernels: _ZN5aiter...fmoe... -> search for 'aiter::fmoe'
        if kernel_name.startswith('_ZN') and 'fmoe' in kernel_name:
            return 'aiter::fmoe'
        
        # CK 2-stage kernels: moe_ck2stages_gemm1/gemm2 -> search for 'kernel_moe_gemm'
        if kernel_name.startswith('moe_ck2stages'):
            return 'kernel_moe_gemm'
        
        # Fallback: use the kernel name directly
        return kernel_name
    
    def get_kernel_name(self, config: MoeKernelConfig) -> Optional[str]:
        """
        Return kernel name from MOE config.
        
        Uses the kernel_name field from the input config CSV.
        """
        return config.kernel_name
    
    def get_cfg_idx(self, config: MoeKernelConfig) -> int:
        """
        Return configuration index from MOE config.
        
        Uses the config_idx field from the input config CSV.
        """
        return config.config_idx
    
    def get_config_columns(self, config: MoeKernelConfig) -> Dict[str, Any]:
        """
        Return MOE-specific configuration columns.
        
        Includes all problem dimensions and kernel configuration.
        Note: cfg_idx and kernel_name are handled by base class.
        """
        return {
            'kernel_type': config.kernel_type,
            'stage': config.stage,
            'token': config.token,
            'model_dim': config.model_dim,
            'inter_dim': config.inter_dim,
            'expert': config.expert,
            'topk': config.topk,
            'block_m': config.block_m,
            'dtype': config.dtype,
            'q_dtype_a': config.q_dtype_a,
            'q_dtype_w': config.q_dtype_w,
            'q_type': config.q_type,
            'act_type': config.act_type,
            'use_g1u1': config.use_g1u1,
            'doweight_stage1': config.doweight_stage1,
            'quant_time_us': config.quant_time_us,
            'error': config.error,
        }
