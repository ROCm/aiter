#pragma once
#include <torch/extension.h>

torch::Tensor ater_add(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_mul(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_sub(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_div(torch::Tensor &input0, torch::Tensor &input1);
torch::Tensor ater_sigmoid(torch::Tensor &input);
torch::Tensor ater_tanh(torch::Tensor &input);
