# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Candidate entry point for the external dsv41_single_layer replay suite."""
import os
from pathlib import Path
from aiter.ops.mono_standard.loader import evidence
from . import runtime_artifacts

os.environ['MONO_ARTIFACT_DIR'] = str(evidence() / 'a05-artifacts')
runtime_artifacts.install()
from aiter.ops.mono_standard import KERNELS  # noqa: E402,F401
