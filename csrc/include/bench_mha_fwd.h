#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include "mha_fwd.h"
#include <vector>

int bench_mha_fwd(int argc, std::vector<std::string> argv);