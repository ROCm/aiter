#include "binary_operator.cuh"
#include "binary_op_api_common.hpp"

torch::Tensor initOutput(torch::Tensor &input, torch::Tensor &other)
{
  torch::ScalarType out_dtype = torch::promote_types(input.scalar_type(), other.scalar_type());
  std::vector<int64_t> out_shape = broadcastShapes(input, other);
  auto device = input.device();
  auto options = torch::TensorOptions().dtype(out_dtype).device(input.device());
  torch::Tensor output = torch::empty(out_shape, options);
  return output;
}

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other)
{
  torch::Tensor output = initOutput(input, other);
  binary_op_dispatch("add", input, other, output);
  return output;
  // return binary_operation<aiter::AddOp, false>(input, other);
}

torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other)
{
  torch::Tensor output = initOutput(input, other);
  binary_op_dispatch("sub", input, other, output);
  return output;
}

torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other)
{
  torch::Tensor output = initOutput(input, other);
  binary_op_dispatch("mul", input, other, output);
  return output;
}

torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other)
{
  torch::Tensor output = initOutput(input, other);
  binary_op_dispatch("div", input, other, output);
  return output;
}

// inp interface
torch::Tensor aiter_add_(torch::Tensor &input, torch::Tensor &other)
{
  binary_op_dispatch("add", input, other, input);
  return input;
}

torch::Tensor aiter_sub_(torch::Tensor &input, torch::Tensor &other)
{
  binary_op_dispatch("sub", input, other, input);
  return input;
}

torch::Tensor aiter_mul_(torch::Tensor &input, torch::Tensor &other)
{
  binary_op_dispatch("mul", input, other, input);
  return input;
}

torch::Tensor aiter_div_(torch::Tensor &input, torch::Tensor &other)
{
  binary_op_dispatch("div", input, other, input);
  return input;
}
