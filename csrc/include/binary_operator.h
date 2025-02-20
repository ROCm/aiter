#pragma once
/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/all.h>
namespace aiter
{
    template <typename T, typename Operation>
    inline __device__ T performOperation(T a, T b);

    template <typename Operation>
    torch::Tensor aten_compute(torch::Tensor &input, torch::Tensor &other);

    struct AddOp
    {
        template <typename T>
        inline __device__ static T apply(T a, T b) { return a + b; }

        static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
        {
            return torch::add(input, other);
        }
    };

    struct SubOp
    {
        template <typename T>
        inline __device__ static T apply(T a, T b)
        {
            return a - b;
        }

        static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
        {
            return torch::sub(input, other);
        }
    };

    struct MulOp
    {
        template <typename T>
        inline __device__ static T apply(T a, T b) { return a * b; }

        static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
        {
            return torch::mul(input, other);
        }
    };

    struct DivOp
    {
        template <typename T>
        inline __device__ static T apply(T a, T b)
        {
            // assert(b == static_cast<T>(0));
            return a / b;
        }

        static torch::Tensor compute(torch::Tensor &input, torch::Tensor &other)
        {
            return torch::div(input, other);
        }
    };

    template <typename T, typename Operation, bool order_flag>
    inline __device__ T performOperation(T a, T b)
    {
        if constexpr (std::is_same_v<Operation, AddOp>)
        {
            return Operation::apply(a, b);
        }
        else if constexpr (std::is_same_v<Operation, SubOp>)
        {
            if constexpr (!order_flag)
            {
                return Operation::apply(b, a);
            }
            else
            {
                return Operation::apply(a, b);
            }
        }
        else if constexpr (std::is_same_v<Operation, MulOp>)
        {
            return Operation::apply(a, b);
        }
        else if constexpr (std::is_same_v<Operation, DivOp>)
        {
            if constexpr (!order_flag)
            {
                return Operation::apply(b, a);
            }
            else
            {
                return Operation::apply(a, b);
            }
        }
        else
        {
            static_assert(false, "Unsupported operation");
        }
    }

    template <typename Operation>
    torch::Tensor aten_compute(torch::Tensor &input, torch::Tensor &other)
    {
        if constexpr (std::is_same_v<Operation, AddOp>)
        {
            return Operation::compute(input, other);
        }
        else if constexpr (std::is_same_v<Operation, SubOp>)
        {
            return Operation::compute(input, other);
        }
        else if constexpr (std::is_same_v<Operation, MulOp>)
        {
            return Operation::compute(input, other);
        }
        else if constexpr (std::is_same_v<Operation, DivOp>)
        {
            return Operation::compute(input, other);
        }
        else
        {
            static_assert(false, "Unsupported operation");
        }
    }
}