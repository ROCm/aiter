// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// Lightweight torch header for aiter.
// Use this instead of <torch/all.h> or <torch/extension.h> to reduce
// compile time and minimize torch version coupling.

#pragma once

#include <torch/types.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
