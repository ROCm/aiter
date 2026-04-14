// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_hip_common.h"
#include <cstring>
#include <initializer_list>

struct aiter_tensor_t
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

/// RAII C++ class for GPU tensor, inherits aiter_tensor_t (POD).
/// - Factory methods: AiterTensor::empty(), AiterTensor::zeros()
/// - Auto hipFree on destruction
/// - Move-only (no copy)
/// - AiterTensor* is implicitly convertible to aiter_tensor_t*
class AiterTensor : public aiter_tensor_t
{
public:
    /// Allocate uninitialized GPU memory.
    static AiterTensor empty(std::initializer_list<int64_t> dims,
                             AiterDtype dtype,
                             int device_id,
                             hipStream_t stream = nullptr)
    {
        (void)stream; // reserved for future async alloc
        AiterTensor t;
        t.init_shape(dims, dtype, device_id);

        size_t nbytes = t.numel_ * AiterDtype_element_size(dtype);
        if(nbytes > 0)
        {
            HipDeviceGuard guard(device_id);
            HIP_CALL(hipMalloc(&t.ptr, nbytes));
        }
        t.owns_memory_ = true;
        return t;
    }

    /// Allocate uninitialized GPU memory with same shape/strides/dtype/device as `other`.
    /// Preserves the original strides of `other`.
    /// Allocates enough storage span to cover the full positive-stride layout.
    static AiterTensor empty_like(const aiter_tensor_t* other,
                                  hipStream_t stream = nullptr)
    {
        (void)stream; // reserved for future async alloc
        AITER_CHECK(other != nullptr, __func__, ": other must not be null");
        AITER_CHECK(other->ndim <= 8, __func__, ": ndim ", other->ndim, " exceeds max 8");
        AiterTensor t;
        t.ndim = other->ndim;
        t.numel_ = other->numel_;
        t.dtype_ = other->dtype_;
        t.device_id = other->device_id;

        size_t storage_nelem = (t.numel_ == 0) ? 0 : 1;
        for(int i = 0; i < other->ndim; ++i)
        {
            t.shape[i] = other->shape[i];
            t.strides[i] = other->strides[i];

            AITER_CHECK(other->strides[i] >= 0,
                        __func__,
                        ": negative strides are not supported");
            if(storage_nelem > 0 && other->shape[i] > 1)
                storage_nelem += static_cast<size_t>(other->shape[i] - 1) *
                                 static_cast<size_t>(other->strides[i]);
        }

        size_t nbytes = storage_nelem * AiterDtype_element_size(t.dtype_);
        if(nbytes > 0)
        {
            HipDeviceGuard guard(t.device_id);
            HIP_CALL(hipMalloc(&t.ptr, nbytes));
        }
        t.owns_memory_ = true;
        return t;
    }

    /// Allocate zero-initialized GPU memory.
    static AiterTensor zeros(std::initializer_list<int64_t> dims,
                             AiterDtype dtype,
                             int device_id,
                             hipStream_t stream = nullptr)
    {
        AiterTensor t = empty(dims, dtype, device_id, stream);
        size_t nbytes = t.numel_ * AiterDtype_element_size(dtype);
        if(nbytes > 0)
        {
            HipDeviceGuard guard(device_id);
            if(stream)
                HIP_CALL(hipMemsetAsync(t.ptr, 0, nbytes, stream));
            else
                HIP_CALL(hipMemset(t.ptr, 0, nbytes));
        }
        return t;
    }

    /// Wrap an existing tensor descriptor without taking ownership.
    explicit AiterTensor(const aiter_tensor_t& other)
        : aiter_tensor_t(other),
          owns_memory_(false)
    {
    }

    ~AiterTensor()
    {
        if(owns_memory_ && ptr)
        {
            HipDeviceGuard guard(device_id);
            hipFree(ptr);
            ptr = nullptr;
        }
    }

    // Move constructor
    AiterTensor(AiterTensor&& other) noexcept
        : aiter_tensor_t(static_cast<aiter_tensor_t&>(other)),
          owns_memory_(other.owns_memory_)
    {
        other.owns_memory_ = false;
        other.ptr = nullptr;
    }

    // Move assignment
    AiterTensor& operator=(AiterTensor&& other) noexcept
    {
        if(this != &other)
        {
            if(owns_memory_ && ptr)
            {
                HipDeviceGuard guard(device_id);
                hipFree(ptr);
            }
            static_cast<aiter_tensor_t&>(*this) = static_cast<aiter_tensor_t&>(other);
            owns_memory_ = other.owns_memory_;
            other.owns_memory_ = false;
            other.ptr = nullptr;
        }
        return *this;
    }

    // No copy
    AiterTensor(const AiterTensor&) = delete;
    AiterTensor& operator=(const AiterTensor&) = delete;

private:
    bool owns_memory_ = false;

    AiterTensor()
    {
        // Zero-init the POD base
        std::memset(static_cast<aiter_tensor_t*>(this), 0, sizeof(aiter_tensor_t));
    }

    void init_shape(std::initializer_list<int64_t> dims, AiterDtype dt, int dev)
    {
        AITER_CHECK(dims.size() <= 8, "AiterTensor supports at most 8 dims, got ", dims.size());
        ndim = static_cast<int>(dims.size());
        int i = 0;
        for(auto d : dims)
            shape[i++] = d;

        // Row-major contiguous strides
        if(ndim > 0)
        {
            strides[ndim - 1] = 1;
            for(int d = ndim - 2; d >= 0; --d)
                strides[d] = strides[d + 1] * shape[d + 1];
        }

        numel_ = 1;
        for(int d = 0; d < ndim; ++d)
            numel_ *= shape[d];

        dtype_ = dt;
        device_id = dev;
    }
};
