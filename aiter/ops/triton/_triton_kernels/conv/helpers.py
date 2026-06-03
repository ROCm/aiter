# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

# Env-var escape hatch: set AITER_TRITON_CONV_AUTOTUNE=1 to bypass JSON-loaded
# configs and let @triton.autotune do a runtime search across each kernel file's
# AUTOTUNE_*_CONFIGS list. Default off — production / CI path uses JSON configs
# from configs/conv/.
CONV_AUTOTUNE_ENABLED = os.environ.get("AITER_TRITON_CONV_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
