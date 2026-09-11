# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys

# Under pytest, silence aiter's INFO chatter (e.g. checkAllclose "passed~") unless the caller
# set AITER_LOG_LEVEL; aiter/__init__.py reads it once, on the first import triggered below.
if "pytest" in sys.modules and "AITER_LOG_LEVEL" not in os.environ:
    os.environ["AITER_LOG_LEVEL"] = "WARNING"

from op_tests.triton_tests.utils import *
