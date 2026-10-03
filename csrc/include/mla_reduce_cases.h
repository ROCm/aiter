// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#define AITER_MLA_REDUCE_CASES(X, ...) \
    X(1, 128, __VA_ARGS__)             \
    X(2, 128, __VA_ARGS__)             \
    X(4, 128, __VA_ARGS__)             \
    X(8, 128, __VA_ARGS__)             \
    X(10, 128, __VA_ARGS__)            \
    X(16, 128, __VA_ARGS__)            \
    X(16, 512, __VA_ARGS__)            \
    X(24, 512, __VA_ARGS__)            \
    X(32, 128, __VA_ARGS__)            \
    X(32, 512, __VA_ARGS__)            \
    X(40, 128, __VA_ARGS__)            \
    X(48, 128, __VA_ARGS__)            \
    X(64, 64, __VA_ARGS__)             \
    X(64, 128, __VA_ARGS__)            \
    X(64, 512, __VA_ARGS__)            \
    X(96, 128, __VA_ARGS__)            \
    X(128, 128, __VA_ARGS__)           \
    X(128, 512, __VA_ARGS__)           \
    X(8, 512, __VA_ARGS__)             \
    X(12, 512, __VA_ARGS__)            \
    X(48, 512, __VA_ARGS__)            \
    X(80, 512, __VA_ARGS__)            \
    X(96, 512, __VA_ARGS__)            \
    X(112, 512, __VA_ARGS__)
