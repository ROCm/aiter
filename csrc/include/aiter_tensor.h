// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_enum.h"
#include <cstdint>

struct AiterTensor
{
    void* ptr;          // data_ptr, pointer to GPU memory
    size_t numel_;      // total number of elements
    int ndim;           // number of dimensions
    int64_t shape[8];   // size of each dimension, up to 8 dims (PyTorch limit)
    int64_t strides[8]; // stride of each dimension
    AiterDtype dtype_;  // data type
    int device_id;      // GPU device index: 0, 1, 2, ...

    // torch::Tensor-compatible accessors
    int64_t size(int i) const { return (i < 0) ? shape[ndim + i] : shape[i]; }
    int64_t stride(int i) const { return (i < 0) ? strides[ndim + i] : strides[i]; }
    void* data_ptr() const { return ptr; }
    size_t numel() const { return numel_; }
    int dim() const { return ndim; }
    AiterDtype dtype() const { return dtype_; }
    size_t element_size() const { return AiterDtype_element_size(dtype_); }

    bool is_contiguous() const
    {
        int64_t expected = 1;
        for(int d = ndim - 1; d >= 0; --d)
        {
            if(shape[d] != 1 && strides[d] != expected)
                return false;
            expected *= shape[d];
        }
        return true;
    }
};
