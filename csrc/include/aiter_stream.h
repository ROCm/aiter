// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_runtime.h>

/// Lightweight thread-local stream manager (pure HIP, no torch dependency).
///
/// Usage:
///   hipStream_t s = aiter::getCurrentHIPStream();
///   aiter::setCurrentHIPStream(stream);

namespace aiter {

namespace detail {

inline hipStream_t& threadLocalStream()
{
    thread_local hipStream_t stream = nullptr;
    return stream;
}

} // namespace detail

inline hipStream_t getCurrentHIPStream()
{
    return detail::threadLocalStream();
}

inline void setCurrentHIPStream(hipStream_t stream)
{
    detail::threadLocalStream() = stream;
}

} // namespace aiter
