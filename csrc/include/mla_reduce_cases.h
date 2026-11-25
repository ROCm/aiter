// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// The (num_head, head_dim) support matrix. Single source of truth for
// MLA_REDUCE_ROUTER and the extern template block in mla_reduce_kernels.cuh; a row added here needs
// a matching TU in reduce_instances/. Without one the .so still links (a shared
// object may leave symbols undefined) but fails to import with an undefined
// dispatch_mla_reduce_v1 symbol.
// Pinned: clang-format reads each F(...) as a nested call and staircases it.
// clang-format off
#define _AITER_MLA_REDUCE_FOR_EACH_ROW(F, ...) \
    F(1, 128, __VA_ARGS__) F(2, 128, __VA_ARGS__) F(4, 128, __VA_ARGS__)                    \
    F(8, 128, __VA_ARGS__) F(10, 128, __VA_ARGS__) F(12, 128, __VA_ARGS__)                  \
    F(16, 128, __VA_ARGS__)                                                                 \
    F(16, 512, __VA_ARGS__) F(24, 512, __VA_ARGS__) F(32, 128, __VA_ARGS__)                 \
    F(32, 512, __VA_ARGS__) F(40, 128, __VA_ARGS__) F(48, 128, __VA_ARGS__)                 \
    F(64, 64, __VA_ARGS__) F(64, 128, __VA_ARGS__) F(64, 512, __VA_ARGS__)                  \
    F(96, 128, __VA_ARGS__) F(128, 128, __VA_ARGS__) F(128, 512, __VA_ARGS__)               \
    F(8, 512, __VA_ARGS__) F(12, 512, __VA_ARGS__) F(48, 512, __VA_ARGS__)                  \
    F(80, 512, __VA_ARGS__) F(96, 512, __VA_ARGS__) F(112, 512, __VA_ARGS__)
// clang-format on
