#pragma once
#include <torch/extension.h>

torch::Tensor ater_add(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_mul(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_sub(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_div(torch::Tensor &input0, torch::Tensor &input1);