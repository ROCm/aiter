// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "opus/opus.hpp"
#include "kittens.cuh"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace hk     = kittens;
namespace hkdart = hk::ducks::art;
namespace hkm    = hk::macros;

#ifndef HIP_CALL
#define HIP_CALL(call)                                                          \
    do                                                                          \
    {                                                                           \
        hipError_t err = (call);                                                \
        if(err != hipSuccess)                                                   \
        {                                                                       \
            std::fprintf(stderr,                                                \
                         "HIP error at %s:%d: %s\n",                            \
                         __FILE__, __LINE__, hipGetErrorString(err));           \
            std::abort();                                                       \
        }                                                                       \
    } while(0)
#endif

typedef uint32_t v2ui __attribute__((ext_vector_type(2)));
typedef uint32_t v4ui __attribute__((ext_vector_type(4)));
typedef uint32_t v8ui __attribute__((ext_vector_type(8)));
