// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Provides thread-local ctypes error storage for extensions that include
// AITER_CTYPES_ERROR_DECL (e.g. asm_fmoe.cu) without another TU that defines
// AITER_CTYPES_ERROR_DEF (e.g. asm_moe_2stage.cu in module_moe_fmoe_asm).
#include "aiter_tensor.h"
#include "aiter_ctypes_error.h"

AITER_CTYPES_ERROR_DEF
