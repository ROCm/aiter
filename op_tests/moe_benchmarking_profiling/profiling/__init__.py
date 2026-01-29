#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Generic Kernel Profiling Package

This package provides a modular framework for profiling GPU kernels using
rocprofv3 with hardware performance counter collection.

Main components:
- profiler: Generic rocprofv3-based profiler for any Python script
- runner: Single/multi-GPU execution manager
- session: High-level profiling session orchestrator
- gpu_utils: GPU detection and architecture utilities
- script_generators: Operator-specific script generation factories
- post_processors: Result combination and cleanup
"""

from .profiler import Profiler, ProfileResult
from .runner import Runner, RunResult, WorkItem
from .session import ProfilingSession, SessionResult
from .gpu_utils import detect_gpu, get_gfx_arch, get_num_gpus

__all__ = [
    'Profiler',
    'ProfileResult',
    'Runner',
    'RunResult',
    'WorkItem',
    'ProfilingSession',
    'SessionResult',
    'detect_gpu',
    'get_gfx_arch',
    'get_num_gpus',
]
