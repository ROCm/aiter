#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
GPU detection and architecture utilities.
"""

import subprocess
from typing import Optional


# Mapping from GPU model to GFX architecture for counter file selection
GPU_TO_GFX = {
    'MI250X': 'gfx90a',
    'MI250': 'gfx90a',
    'MI300A': 'gfx942',
    'MI300X': 'gfx942',
    'MI355X': 'gfx950',
}

# Supported GPU architectures
SUPPORTED_ARCHS = ['MI250X', 'MI300A', 'MI300X', 'MI355X']


def detect_gpu() -> Optional[str]:
    """
    Detect the AMD GPU model running on the system.
    
    Returns:
        One of: 'MI250X', 'MI300A', 'MI300X', 'MI355X', or None if detection fails.
    """
    try:
        # Try using rocm-smi to get GPU information
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Check for specific GPU models
            if 'mi355x' in output or 'instinct mi355x' in output:
                return 'MI355X'
            elif 'mi300x' in output or 'instinct mi300x' in output:
                return 'MI300X'
            elif 'mi300a' in output or 'instinct mi300a' in output:
                return 'MI300A'
            elif 'mi250x' in output or 'instinct mi250x' in output:
                return 'MI250X'
            elif 'mi250' in output or 'instinct mi250' in output:
                return 'MI250X'
        
        # Try rocminfo for more detailed info
        result = subprocess.run(
            ['rocminfo'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False
        )
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Check marketing name or device name
            for line in output.split('\n'):
                if 'marketing name' in line or 'name:' in line:
                    if 'mi355x' in line:
                        return 'MI355X'
                    elif 'mi300x' in line:
                        return 'MI300X'
                    elif 'mi300a' in line:
                        return 'MI300A'
                    elif 'mi250x' in line or 'mi250' in line:
                        return 'MI250X'
            
            # Match gfx architecture to GPU model
            if 'gfx950' in output:
                return 'MI355X'
            elif 'gfx942' in output:
                # gfx942 can be MI300A or MI300X, default to MI300X
                if 'mi300a' in output:
                    return 'MI300A'
                return 'MI300X'
            elif 'gfx90a' in output:
                return 'MI250X'
    
    except FileNotFoundError:
        pass
    except Exception:
        pass
    
    return None


def get_gfx_arch(gpu_model: str) -> str:
    """
    Get the GFX architecture string for a GPU model.
    
    Args:
        gpu_model: GPU model name (e.g., 'MI300X')
        
    Returns:
        GFX architecture string (e.g., 'gfx942')
    """
    return GPU_TO_GFX.get(gpu_model.upper(), 'gfx942')


def get_num_gpus() -> int:
    """
    Get the number of available GPUs.
    
    Returns:
        Number of GPUs, or 1 if detection fails
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    
    # Fallback: try rocm-smi
    try:
        result = subprocess.run(
            ['rocm-smi', '--showid'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False
        )
        if result.returncode == 0:
            # Count lines with GPU IDs
            lines = [l for l in result.stdout.split('\n') if 'GPU' in l]
            if lines:
                return len(lines)
    except Exception:
        pass
    
    return 1


def validate_arch(arch: str) -> str:
    """
    Validate and normalize architecture string.
    
    Args:
        arch: GPU architecture name
        
    Returns:
        Normalized architecture name
        
    Raises:
        ValueError: If architecture is not supported
    """
    arch_upper = arch.upper()
    if arch_upper not in SUPPORTED_ARCHS:
        raise ValueError(
            f"Unsupported architecture: {arch}. "
            f"Must be one of: {', '.join(SUPPORTED_ARCHS)}"
        )
    return arch_upper
