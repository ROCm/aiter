// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
#include "aiter_hip_common.h"

// Stream-ordered scratch for the small-shape host launchers. The HIP memory
// pool reuses eager allocations only after their last stream use. During
// capture, allocation and free become graph nodes, so separate graphs do not
// share a cached pointer. Always use the asynchronous APIs, including on the
// default stream (AiterTensor::empty uses hipMalloc for that stream).
class OpusGfx950Scratch
{
public:
    OpusGfx950Scratch(size_t bytes, int device, hipStream_t stream)
        : device_(device), stream_(stream)
    {
        if(bytes > 0)
        {
            HipDeviceGuard guard(device_);
            HIP_CALL(hipMallocAsync(&ptr_, bytes, stream_));
        }
    }

    ~OpusGfx950Scratch() noexcept
    {
        if(ptr_)
        {
            HipDeviceGuard guard(device_);
            hipFreeAsync(ptr_, stream_);
        }
    }

    void* data_ptr() const { return ptr_; }

    OpusGfx950Scratch(const OpusGfx950Scratch&) = delete;
    OpusGfx950Scratch& operator=(const OpusGfx950Scratch&) = delete;

private:
    void* ptr_ = nullptr;
    int device_;
    hipStream_t stream_;
};
#endif
