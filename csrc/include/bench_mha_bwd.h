#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include "mha_bwd.h"
#include <vector>

int bench_mha_bwd(int argc, std::vector<std::string> argv);