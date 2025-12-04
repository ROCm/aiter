#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include <torch/extension.h>

void topk_plain(torch::Tensor& values,
                torch::Tensor& topk_ids,
                int topk_num,
                bool largest);
