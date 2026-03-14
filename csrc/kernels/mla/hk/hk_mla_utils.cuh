// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "kittens.cuh"

namespace hk     = kittens;
namespace hkdart = hk::ducks::art;
namespace hkm    = hk::macros;
namespace ckt    = ck_tile;

typedef uint32_t v2ui __attribute__((ext_vector_type(2)));
typedef uint32_t v4ui __attribute__((ext_vector_type(4)));
typedef uint32_t v8ui __attribute__((ext_vector_type(8)));
